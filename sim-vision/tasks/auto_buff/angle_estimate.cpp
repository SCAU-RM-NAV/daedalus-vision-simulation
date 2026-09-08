#include "angle_estimate.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <limits>


#include "tools/logger.hpp"

namespace auto_buff
{
namespace
{
std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

template <typename T>
bool read_optional(
  const YAML::Node & root, const YAML::Node & group, const std::string & root_key,
  const std::string & group_key, T & value)
{
  if (group && group[group_key]) {
    value = group[group_key].as<T>();
    return true;
  }
  if (root[root_key]) {
    value = root[root_key].as<T>();
    return true;
  }
  return false;
}

// 将角度范围映射在[−π,π]
double wrap_angle(double value) { return std::atan2(std::sin(value), std::cos(value)); }

// 返回从 from 转到 to 的最短有符号角距离
double shortest_angular_distance(double from, double to) { return wrap_angle(to - from); }

constexpr double kLeafAngleStep = 2.0 * M_PI / 5.0;
// 相邻确认帧的真实转动远小于该门限;只用它剔除 ID 误匹配/重复帧造成的整叶跳变
constexpr double kDirectionOutlierGateRad = 0.1;
// 望远镜求和:中间噪声相互抵消,累积到该阈值即可锁定方向(小符约 0.15s)
constexpr double kDirectionLockThresholdRad = 0.15;

// 对历史队列降采样，最多返回约 100 个样本索引
std::vector<std::size_t> sample_indices(const std::deque<LeafData> & leaf_datas)
{
  const auto leaf_count = leaf_datas.size();
  constexpr std::size_t dense_tail = 65; // 最新数据最多取65个
  constexpr std::size_t sparse_head = 35; // 旧数据最大均匀取35个
  const std::size_t dense_n = std::min(dense_tail, leaf_count);
  const std::size_t history_pool = leaf_count - dense_n;
  const std::size_t sparse_n = std::min(sparse_head, history_pool);

  std::vector<std::size_t> indices;
  indices.reserve(dense_n + sparse_n);
  if (sparse_n > 0 && history_pool > 0) {
    for (std::size_t i = 0; i < sparse_n; ++i) {
      const std::size_t idx =
        (sparse_n == 1)
          ? 0
          : static_cast<std::size_t>(std::round(
              static_cast<double>(i) * (history_pool - 1) / static_cast<double>(sparse_n - 1)));
      indices.push_back(idx);
    }
  }
  for (std::size_t i = leaf_count - dense_n; i < leaf_count; ++i) indices.push_back(i);
  return indices;
}

// ceres通用配置
ceres::Solver::Options make_solver_options(
  const AngleEstimateConfig & config, ceres::LinearSolverType linear_solver_type)
{
  ceres::Solver::Options options;
  options.linear_solver_type = linear_solver_type;
  options.minimizer_progress_to_stdout = false;
  options.function_tolerance = 1e-5;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-7;
  options.num_threads = config.ceres_threads;
  return options;
}

void record_solver_summary(
  const ceres::Solver::Summary & summary, std::chrono::steady_clock::time_point fit_start,
  AngleEstimate::AngleFitResult & result)
{
  result.solve_time_ms = summary.total_time_in_seconds * 1000.0;
  result.ceres_iterations = static_cast<int>(summary.iterations.size());
  result.ceres_successful_steps = summary.num_successful_steps;
  result.ceres_unsuccessful_steps = summary.num_unsuccessful_steps;
  result.total_fit_wall_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start).count();
}

// ===== Big rune Ceres (大符 Ceres) =====

// 相对时间形式的大符基础角度。tau = t - t_ref, delta_t0 = t0 - t_ref。
// base(tau) = -a/w * cos(w*(tau - delta_t0)) + (2.09 - a)*tau
// 完整角度 = direction * (base(tau) + c_ref) + leaf_id * kLeafAngleStep
double big_base_angle(double tau, double a, double w, double delta_t0)
{
  return -a / w * std::cos(w * (tau - delta_t0)) + (2.09 - a) * tau;
}

double observation_weight(double leaf_t, double first_time, double current_time)
{
  const double tau = std::max(current_time - first_time, 1e-3);
  return std::exp(-(current_time - leaf_t) / tau);
}

double huber_cost(double residual, double delta)
{
  const double absolute_residual = std::abs(residual);
  if (absolute_residual <= delta) return residual * residual;
  return 2.0 * delta * absolute_residual - delta * delta;
}

double huber_irls_weight(double residual, double delta)
{
  const double absolute_residual = std::abs(residual);
  return absolute_residual <= delta || absolute_residual <= 1e-12 ? 1.0
                                                                    : delta / absolute_residual;
}

struct ColdStartCandidate
{
  double a = 0.0;
  double w = 0.0;
  double c = 0.0;
  double t0 = 0.0;
  double cost = std::numeric_limits<double>::infinity();
};

ColdStartCandidate evaluate_cold_start_candidate(
  const std::deque<LeafData> & leaf_datas, const std::vector<std::size_t> & indices, int direction,
  double a, double w, double test_delta_t0, double huber_delta)
{
  ColdStartCandidate candidate;
  candidate.a = a;
  candidate.w = w;
  candidate.t0 = test_delta_t0;  // 内部存储为 Δt0
  const double first_time = leaf_datas.front().t;
  const double t_ref = leaf_datas.back().t;
  auto circular_mean = [&](double current_c_ref, bool robust, double & out_c_ref) {
    double sum_sin = 0.0;
    double sum_cos = 0.0;
    for (const auto idx : indices) {
      const auto & leaf = leaf_datas[idx];
      const double leaf_tau = leaf.t - t_ref;
      const double base = big_base_angle(leaf_tau, a, w, test_delta_t0);
      const double phase = wrap_angle(direction * leaf.zero_angle - base);
      const double weight = observation_weight(leaf.t, first_time, t_ref);
      const double residual = std::sqrt(weight) * wrap_angle(phase - current_c_ref);
      const double effective_weight =
        robust ? weight * huber_irls_weight(residual, huber_delta) : weight;
      sum_sin += effective_weight * std::sin(phase);
      sum_cos += effective_weight * std::cos(phase);
    }
    if (std::hypot(sum_sin, sum_cos) < 1e-6) return false;
    out_c_ref = std::atan2(sum_sin, sum_cos);
    return true;
  };

  if (!circular_mean(0.0, false, candidate.c)) return candidate;
  for (int iteration = 0; iteration < 2; ++iteration) {
    if (!circular_mean(candidate.c, true, candidate.c)) return candidate;
  }

  double cost = 0.0;
  double weight_sum = 0.0;
  for (const auto idx : indices) {
    const auto & leaf = leaf_datas[idx];
    const double leaf_tau = leaf.t - t_ref;
    const double base = big_base_angle(leaf_tau, a, w, test_delta_t0);
    const double pred_zero = direction * (base + candidate.c);
    const double err = wrap_angle(leaf.zero_angle - pred_zero);
    const double weight = observation_weight(leaf.t, first_time, t_ref);
    cost += huber_cost(std::sqrt(weight) * err, huber_delta);
    weight_sum += weight;
  }

  candidate.cost = cost / std::max(weight_sum, 1e-9);
  return candidate;
}

// 在相角 φ = w*Δt0 ∈ [0, 2π) 上网格搜索最优 Δt0 和 c_ref。
// 相比直接搜索 Δt0，相角搜索更均匀地覆盖一个完整周期，可用更少点数达到同等精度。
ColdStartCandidate search_cold_start_t0_c(
  const std::deque<LeafData> & leaf_datas, int direction, double a, double w, int coarse_steps,
  int refine_steps, double huber_delta)
{
  ColdStartCandidate candidate;
  candidate.a = a;
  candidate.w = w;
  if (leaf_datas.empty() || std::abs(w) < 1e-6) {
    candidate.t0 = 0.0;
    candidate.c = leaf_datas.empty() ? 0.0 : leaf_datas.front().zero_angle;
    return candidate;
  }

  const auto indices = sample_indices(leaf_datas);
  const int coarse_count = std::max(4, coarse_steps);
  const int refine_count = std::max(2, refine_steps);
  const double inv_w = 1.0 / w;
  const double dphi_coarse = 2.0 * M_PI / static_cast<double>(coarse_count);

  // 粗搜：在相角 [0, 2π) 上均匀采样
  for (int i = 0; i < coarse_count; ++i) {
    const double phi = static_cast<double>(i) * dphi_coarse;
    const double test_delta_t0 = phi * inv_w;
    const auto test = evaluate_cold_start_candidate(
      leaf_datas, indices, direction, a, w, test_delta_t0, huber_delta);
    if (test.cost < candidate.cost) candidate = test;
  }

  // 精搜：在最佳相角附近细化搜索
  const double best_phi = w * candidate.t0;
  const double dphi_refine = 2.0 * dphi_coarse;  // 搜索窗口 ± 一个粗搜步长
  for (int i = 0; i < refine_count; ++i) {
    const double alpha = static_cast<double>(i) / static_cast<double>(refine_count - 1);
    const double phi = best_phi - dphi_coarse + alpha * dphi_refine;
    const double test_delta_t0 = phi * inv_w;
    const auto test = evaluate_cold_start_candidate(
      leaf_datas, indices, direction, a, w, test_delta_t0, huber_delta);
    if (test.cost < candidate.cost) candidate = test;
  }

  return candidate;
}

// 大符正弦模型残差（相对时间形式）。这里手写雅可比，减少 Ceres 自动求导的开销。
class BigTrigCostFunction final : public ceres::SizedCostFunction<1, 1, 1, 1, 1>
{
public:
  BigTrigCostFunction(
    double tau, int leaf_id, double measured_angle, int direction, double weight)
  : tau_(tau),
    measured_angle_(measured_angle),
    leaf_id_(leaf_id),
    direction_(direction),
    weight_(weight)
  {
  }

  bool Evaluate(
    double const * const * parameters, double * residuals, double ** jacobians) const override
  {
    const double a = parameters[0][0];
    const double w = parameters[1][0];
    const double c_ref = parameters[2][0];
    const double delta_t0 = parameters[3][0];
    const double dt = tau_ - delta_t0;
    const double u = w * dt;
    const double sin_u = std::sin(u);
    const double cos_u = std::cos(u);
    const double inv_w = 1.0 / w;
    const double base = big_base_angle(tau_, a, w, delta_t0) + c_ref;
    const double value = static_cast<double>(direction_) * base + leaf_id_ * kLeafAngleStep;
    const double error = wrap_angle(measured_angle_ - value);
    residuals[0] = weight_ * error;

    if (jacobians != nullptr) {
      const double d_residual_d_base = -weight_ * static_cast<double>(direction_);
      // 关键改动：∂base/∂a 中 time_ → tau_，消除了 -t_ref 项
      const double d_base_da = -cos_u * inv_w - tau_;
      const double d_base_dw = a * cos_u * inv_w * inv_w + a * dt * sin_u * inv_w;
      const double d_base_dc = 1.0;
      const double d_base_dt0 = -a * sin_u;

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = d_residual_d_base * d_base_da;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = d_residual_d_base * d_base_dw;
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = d_residual_d_base * d_base_dc;
      }
      if (jacobians[3] != nullptr) {
        jacobians[3][0] = d_residual_d_base * d_base_dt0;
      }
    }
    return true;
  }

private:
  double tau_;
  double measured_angle_;
  int leaf_id_;
  int direction_;
  double weight_;
};

// 参数先验残差
struct ParameterPriorCostFunctor
{
  explicit ParameterPriorCostFunctor(double initial_value) : initial_value_(initial_value) {}

  template <typename T>
  bool operator()(const T * const param, T * residual) const
  {
    residual[0] = param[0] - T(initial_value_);
    return true;
  }

  double initial_value_;
};

struct BigFitVariables
{
  double a = 0.0;
  double w = 0.0;
  double c = 0.0;
  double t0 = 0.0;
};

std::vector<BigFitVariables> make_big_cold_start_seeds(
  const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  AngleEstimate::State state_snapshot, bool ever_fit_snapshot, const AngleEstimateConfig & config,
  double & cold_start_time_ms)
{
  const double t_ref = leaf_datas.empty() ? 0.0 : leaf_datas.back().t;
  // 把 EnergyTri 的绝对坐标转成 Ceres 内部的相对坐标
  const double c_ref_warm = tri_snapshot.c + (2.09 - tri_snapshot.a) * t_ref;
  const double delta_t0_warm = tri_snapshot.t0 - t_ref;
  BigFitVariables current{tri_snapshot.a, tri_snapshot.w, c_ref_warm, delta_t0_warm};
  if (state_snapshot == AngleEstimate::State::FITTED || ever_fit_snapshot) return {current};

  const auto cold_start_begin = std::chrono::steady_clock::now();
  const std::array<double, 3> a_values{
    config.big_a_min, 0.5 * (config.big_a_min + config.big_a_max), config.big_a_max};
  const std::array<double, 3> w_values{
    config.big_w_min, 0.5 * (config.big_w_min + config.big_w_max), config.big_w_max};
  std::vector<ColdStartCandidate> candidates;
  candidates.reserve(a_values.size() * w_values.size());
  for (const double a : a_values) {
    for (const double w : w_values) {
      candidates.push_back(search_cold_start_t0_c(
        leaf_datas, tri_snapshot.direction, a, w, config.big_cold_start_coarse_steps,
        config.big_cold_start_refine_steps, config.big_huber_delta));
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
    return lhs.cost < rhs.cost;
  });
  cold_start_time_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cold_start_begin)
      .count();

  constexpr std::size_t max_seed_count = 2;
  std::vector<BigFitVariables> seeds;
  seeds.reserve(std::min(max_seed_count, candidates.size()));
  for (std::size_t index = 0; index < candidates.size() && seeds.size() < max_seed_count; ++index) {
    const auto & candidate = candidates[index];
    if (!std::isfinite(candidate.cost)) continue;
    seeds.push_back({candidate.a, candidate.w, candidate.c, candidate.t0});
  }
  return seeds.empty() ? std::vector<BigFitVariables>{current} : seeds;
}

void add_big_observation_residuals(
  ceres::Problem & problem, const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  const AngleEstimateConfig & config, BigFitVariables & variables,
  AngleEstimate::AngleFitResult & result)
{
  const double first_time = leaf_datas.front().t;
  const double t_ref = leaf_datas.back().t;
  const double decay_scale = std::max(t_ref - first_time, 1e-3);

  // 近期样本更影响未来预测，历史样本只保留稀疏约束。
  for (const auto idx : sample_indices(leaf_datas)) {
    const auto & leaf = leaf_datas[idx];
    const double weight = std::exp(-(t_ref - leaf.t) / decay_scale);
    const double leaf_tau = leaf.t - t_ref;  // 相对时间，最新帧 τ=0
    auto * cost = new BigTrigCostFunction(
      leaf_tau, leaf.leaf_id, leaf.angle, tri_snapshot.direction, std::sqrt(weight));
    problem.AddResidualBlock(
      cost, new ceres::HuberLoss(config.big_huber_delta), &variables.a, &variables.w, &variables.c,
      &variables.t0);
    ++result.residual_count;
  }
}

void add_big_parameter_priors(
  ceres::Problem & problem, const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  const AngleEstimateConfig & config, BigFitVariables & variables, bool use_history_priors,
  AngleEstimate::AngleFitResult & result)
{
  // 1. 物理中枢约束：始终生效，弱锚定防止参数漂移到物理边界
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<ParameterPriorCostFunctor, 1, 1>(
      new ParameterPriorCostFunctor(config.big_a_center)),
    new ceres::ScaledLoss(
      new ceres::HuberLoss(1.0), config.big_center_prior_weight, ceres::TAKE_OWNERSHIP),
    &variables.a);
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<ParameterPriorCostFunctor, 1, 1>(
      new ParameterPriorCostFunctor(config.big_w_center)),
    new ceres::ScaledLoss(
      new ceres::HuberLoss(1.0), config.big_center_prior_weight, ceres::TAKE_OWNERSHIP),
    &variables.w);
  result.residual_count += 2;

  // 2. 历史连续性约束：仅在已有历史拟合结果时生效，拉住上一帧的值防止跳变
  if (!use_history_priors) return;
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<ParameterPriorCostFunctor, 1, 1>(
      new ParameterPriorCostFunctor(tri_snapshot.a)),
    new ceres::ScaledLoss(
      new ceres::HuberLoss(1.0), config.big_omega_a, ceres::TAKE_OWNERSHIP),
    &variables.a);
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<ParameterPriorCostFunctor, 1, 1>(
      new ParameterPriorCostFunctor(tri_snapshot.w)),
    new ceres::ScaledLoss(
      new ceres::HuberLoss(1.0), config.big_omega_w, ceres::TAKE_OWNERSHIP),
    &variables.w);
  result.residual_count += 2;
}

void apply_big_parameter_bounds(
  ceres::Problem & problem, const AngleEstimateConfig & config, BigFitVariables & variables)
{
  problem.SetParameterLowerBound(&variables.a, 0, config.big_a_min);
  problem.SetParameterUpperBound(&variables.a, 0, config.big_a_max);
  problem.SetParameterLowerBound(&variables.w, 0, config.big_w_min);
  problem.SetParameterUpperBound(&variables.w, 0, config.big_w_max);
}

bool solve_big_candidate(
  const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  const AngleEstimateConfig & config, const BigFitVariables & seed,
  bool use_history_priors, std::chrono::steady_clock::time_point fit_start,
  AngleEstimate::AngleFitResult & result)
{
  result = {};
  result.mode = tri_snapshot.mode;
  result.energy_tri = tri_snapshot;
  BigFitVariables variables = seed;

  const auto build_start = std::chrono::steady_clock::now();
  ceres::Problem problem;
  add_big_observation_residuals(problem, leaf_datas, tri_snapshot, config, variables, result);
  add_big_parameter_priors(
    problem, leaf_datas, tri_snapshot, config, variables, use_history_priors, result);
  apply_big_parameter_bounds(problem, config, variables);
  result.problem_build_time_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start)
      .count();

  auto options = make_solver_options(config, ceres::DENSE_NORMAL_CHOLESKY);
  options.max_num_iterations = config.max_iterations;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  record_solver_summary(summary, fit_start, result);

  result.average_cost =
    summary.final_cost / static_cast<double>(std::max(result.residual_count, 1));
  // 判断条件
  if (
    summary.termination_type == ceres::CONVERGENCE &&
    result.average_cost < config.big_cost_threshold) {
    // 把 Ceres 内部的相对坐标 (Δt0, c_ref) 转回 EnergyTri 的绝对坐标 (t0, c)
    const double t_ref = leaf_datas.back().t;
    result.energy_tri.a = variables.a;
    result.energy_tri.w = variables.w;
    result.energy_tri.t0 = variables.t0 + t_ref;  // t0 = Δt0 + t_ref
    result.energy_tri.c = variables.c - (2.09 - variables.a) * t_ref;  // c = c_ref - (2.09-a)·t_ref
    result.success = true;
    return true;
  }
  return false;
}

// Big rune Ceres entry point.
bool fit_big_on_snapshot(
  const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  AngleEstimate::State state_snapshot, bool ever_fit_snapshot, const AngleEstimateConfig & config,
  AngleEstimate::AngleFitResult & result)
{
  const auto fit_start = std::chrono::steady_clock::now();
  double cold_start_time_ms = 0.0;
  const auto seeds = make_big_cold_start_seeds(
    leaf_datas, tri_snapshot, state_snapshot, ever_fit_snapshot, config, cold_start_time_ms);
  const bool use_history_priors =
    state_snapshot == AngleEstimate::State::FITTED || ever_fit_snapshot;

  AngleEstimate::AngleFitResult best_attempt;
  bool have_attempt = false;
  bool have_success = false;
  for (const auto & seed : seeds) {
    AngleEstimate::AngleFitResult candidate;
    const bool success = solve_big_candidate(
      leaf_datas, tri_snapshot, config, seed, use_history_priors, fit_start, candidate);
    const double candidate_cost =
      std::isfinite(candidate.average_cost) ? candidate.average_cost
                                            : std::numeric_limits<double>::infinity();
    const double best_cost =
      std::isfinite(best_attempt.average_cost) ? best_attempt.average_cost
                                                : std::numeric_limits<double>::infinity();
    if (
      !have_attempt || (success && (!have_success || candidate_cost < best_cost)) ||
      (!have_success && candidate_cost < best_cost)) {
      best_attempt = candidate;
      have_attempt = true;
    }
    have_success = have_success || success;
  }

  result = best_attempt;
  result.cold_start_time_ms = cold_start_time_ms;
  result.total_fit_wall_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fit_start).count();
  return have_success;
}

// ===== Small rune Ceres (小符 Ceres) =====

// 小符线性模型残差
struct LinearResidual
{
  LinearResidual(double x, int id, double y, int direction)
  : x_(x), y_(y), id_(id), direction_(direction)
  {
  }

  template <typename T>
  bool operator()(const T * const k, const T * const b, T * residual) const
  {
    T value = k[0] * T(x_) + b[0];
    value = T(direction_) * value + T(id_) * T(kLeafAngleStep);
    const T error = T(y_) - value;
    residual[0] = ceres::atan2(ceres::sin(error), ceres::cos(error));
    return true;
  }

  double x_;
  double y_;
  int id_;
  int direction_;
};

// 小符速度先验残差
struct SmallSpeedPriorFunctor
{
  explicit SmallSpeedPriorFunctor(double target_k) : target_k_(target_k) {}

  template <typename T>
  bool operator()(const T * const k, T * residual) const
  {
    residual[0] = k[0] - T(target_k_);
    return true;
  }

  double target_k_;
};

struct SmallFitVariables
{
  double k = 0.0;
  double b = 0.0;
  double t0 = 0.0;
};

void add_small_observation_residuals(
  ceres::Problem & problem, const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  const AngleEstimateConfig & config, SmallFitVariables & variables,
  AngleEstimate::AngleFitResult & result)
{
  // 小符队列最大 120 个样本，规模远小于大符，无需降采样，全量送入 Ceres。
  for (const auto & leaf : leaf_datas) {
    auto * cost = new ceres::AutoDiffCostFunction<LinearResidual, 1, 1, 1>(
      new LinearResidual(leaf.t - variables.t0, leaf.leaf_id, leaf.angle, tri_snapshot.direction));
    problem.AddResidualBlock(
      cost, new ceres::HuberLoss(config.small_huber_delta), &variables.k, &variables.b);
    ++result.residual_count;
  }
}

// Small rune Ceres entry point.
bool fit_small_on_snapshot(
  const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot,
  const AngleEstimateConfig & config, AngleEstimate::AngleFitResult & result)
{
  const auto fit_start = std::chrono::steady_clock::now();
  const double t_ref = leaf_datas.back().t;
  const double b_at_t_ref = tri_snapshot.b + tri_snapshot.k * (t_ref-tri_snapshot.t0);
  SmallFitVariables variables{tri_snapshot.k, b_at_t_ref, t_ref};
  // SmallFitVariables variables{tri_snapshot.k, tri_snapshot.b, tri_snapshot.t0};

  const auto build_start = std::chrono::steady_clock::now();
  ceres::Problem problem;
  add_small_observation_residuals(problem, leaf_datas, tri_snapshot, config, variables, result);

  // 添加速度先验约束 (小符理论角速度约为 PI / 3)
  constexpr double kSmallRuneTheoreticalSpeed = M_PI / 3.0;
  constexpr double omega_k = 60.0; // 速度约束权重，可根据噪声情况调整
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<SmallSpeedPriorFunctor, 1, 1>(
      new SmallSpeedPriorFunctor(kSmallRuneTheoreticalSpeed)),
    new ceres::ScaledLoss(new ceres::HuberLoss(1.0), omega_k, ceres::TAKE_OWNERSHIP),
    &variables.k);
  result.residual_count += 1;

  // 限制转速 k 在理论值的 [0.8, 1.2] 倍范围内
  problem.SetParameterLowerBound(&variables.k, 0, 0.8 * kSmallRuneTheoreticalSpeed);
  problem.SetParameterUpperBound(&variables.k, 0, 1.2 * kSmallRuneTheoreticalSpeed);

  result.problem_build_time_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start)
      .count();
 
  auto options = make_solver_options(config, ceres::DENSE_QR);
  options.max_num_iterations = config.max_iterations;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  record_solver_summary(summary, fit_start, result);

  result.energy_tri.k = variables.k;
  result.energy_tri.b = variables.b;
  result.energy_tri.t0 = t_ref;

  std::cout << "k:" << result.energy_tri.k << std::endl;
  std::cout << "b:" << result.energy_tri.b << std::endl;


  result.average_cost =
    summary.final_cost / static_cast<double>(std::max(result.residual_count, 1));
  if (
    summary.termination_type == ceres::CONVERGENCE &&
    result.average_cost < config.small_cost_threshold) {
    result.success = true;
    return true;
  }
  return false;
}
}  // namespace

// ===== Model and configuration =====

double EnergyTri::getAngle(double time) const
{
  if (mode == BuffMode::BIG) {
    return direction * (-a / w * std::cos(w * (time - t0)) + (2.09 - a) * time + c);
  }
  if (mode == BuffMode::SMALL) return direction * (k * (time - t0) + b);
  return 0.0;
}

double EnergyTri::getSpeed(double time) const
{
  if (mode == BuffMode::BIG) {
    return direction * (a * std::sin(w * (time - t0)) + (2.09 - a));
  }
  if (mode == BuffMode::SMALL) return direction * k;
  return 0.0;
}

AngleEstimateConfig AngleEstimateConfig::from_yaml(const std::string & config_path)
{
  AngleEstimateConfig config;
  auto yaml = YAML::LoadFile(config_path);
  auto predictor = yaml["predictor"];
  auto id_tracking = yaml["id_tracking"];
  auto fitting = yaml["fitting"];

  std::string small_predictor;
  if (read_optional(yaml, predictor, "small_predictor", "small_predictor", small_predictor)) {
    config.small_predictor = lower(small_predictor) == "target_ekf" ? SmallRunePredictor::TARGET_EKF
                                                                    : SmallRunePredictor::CERES;
  }

  read_optional(yaml, predictor, "max_match_angle", "max_match_angle", config.max_match_angle);
  read_optional(yaml, predictor, "min_time_change", "min_time_change", config.min_time_change);
  read_optional(
    yaml, predictor, "outliers_threshold", "outliers_threshold", config.outliers_threshold);
  read_optional(yaml, predictor, "sigma_multiple", "sigma_multiple", config.sigma_multiple);
  read_optional(yaml, predictor, "big_queue_size", "big_queue_size", config.big_queue_size);
  read_optional(yaml, predictor, "small_queue_size", "small_queue_size", config.small_queue_size);

  read_optional(yaml, fitting, "big_min_fit_size", "big_min_fit_size", config.big_min_fit_size);
  read_optional(
    yaml, fitting, "small_min_fit_size", "small_min_fit_size", config.small_min_fit_size);
  read_optional(
    yaml, fitting, "big_direction_lock_threshold", "big_direction_lock_threshold",
    config.big_direction_lock_threshold);
  read_optional(
    yaml, fitting, "small_direction_lock_threshold", "small_direction_lock_threshold",
    config.small_direction_lock_threshold);
  read_optional(yaml, fitting, "fitting_timeout_sec", "timeout_sec", config.fitting_timeout_sec);
  read_optional(yaml, fitting, "big_a_min", "big_a_min", config.big_a_min);
  read_optional(yaml, fitting, "big_a_max", "big_a_max", config.big_a_max);
  read_optional(yaml, fitting, "big_w_min", "big_w_min", config.big_w_min);
  read_optional(yaml, fitting, "big_w_max", "big_w_max", config.big_w_max);
  read_optional(yaml, fitting, "big_huber_delta", "huber_delta", config.big_huber_delta);
  read_optional(yaml, fitting, "big_omega_a", "omega_a", config.big_omega_a);
  read_optional(yaml, fitting, "big_omega_w", "omega_w", config.big_omega_w);
  read_optional(
    yaml, fitting, "big_a_center", "big_a_center", config.big_a_center);
  read_optional(
    yaml, fitting, "big_w_center", "big_w_center", config.big_w_center);
  read_optional(
    yaml, fitting, "big_center_prior_weight", "big_center_prior_weight",
    config.big_center_prior_weight);
  read_optional(
    yaml, fitting, "big_cold_start_coarse_steps", "big_cold_start_coarse_steps",
    config.big_cold_start_coarse_steps);
  read_optional(
    yaml, fitting, "big_cold_start_refine_steps", "big_cold_start_refine_steps",
    config.big_cold_start_refine_steps);
  read_optional(yaml, fitting, "max_iterations", "max_iterations", config.max_iterations);
  read_optional(yaml, fitting, "ceres_threads", "ceres_threads", config.ceres_threads);
  read_optional(
    yaml, fitting, "big_cost_threshold", "big_cost_threshold", config.big_cost_threshold);
  read_optional(
    yaml, fitting, "small_cost_threshold", "small_cost_threshold", config.small_cost_threshold);
  read_optional(yaml, fitting, "small_huber_delta", "small_huber_delta", config.small_huber_delta);

  read_optional(
    yaml, predictor, "small_ekf_p0_angle", "small_ekf_p0_angle", config.small_ekf_p0_angle);
  read_optional(
    yaml, predictor, "small_ekf_p0_speed", "small_ekf_p0_speed", config.small_ekf_p0_speed);
  read_optional(
    yaml, predictor, "small_ekf_q_accel", "small_ekf_q_accel", config.small_ekf_q_accel);
  read_optional(
    yaml, predictor, "small_ekf_r_angle", "small_ekf_r_angle", config.small_ekf_r_angle);

  read_optional(yaml, id_tracking, "max_match_angle", "max_match_angle", config.max_match_angle);
  read_optional(
    yaml, id_tracking, "min_residual_margin", "min_residual_margin", config.min_residual_margin);
  read_optional(
    yaml, id_tracking, "max_id_angular_speed", "max_id_angular_speed", config.max_id_angular_speed);
  read_optional(
    yaml, id_tracking, "max_zero_phase_innovation_rad", "max_zero_phase_innovation_rad",
    config.max_zero_phase_innovation_rad);
  read_optional(
    yaml, id_tracking, "coast_timeout_sec", "coast_timeout_sec", config.id_coast_timeout_sec);
  read_optional(
    yaml, id_tracking, "reset_timeout_sec", "reset_timeout_sec", config.id_reset_timeout_sec);
  read_optional(yaml, id_tracking, "use_model_prior", "use_model_prior", config.use_model_prior);
  read_optional(
    yaml, id_tracking, "model_min_confirmed_frames", "model_min_confirmed_frames",
    config.model_min_confirmed_frames);
  read_optional(yaml, id_tracking, "model_max_nis", "model_max_nis", config.model_max_nis);

  config.max_match_angle = std::clamp(config.max_match_angle, 0.01, M_PI / 5.0);
  config.min_residual_margin = std::max(0.0, config.min_residual_margin);
  config.max_id_angular_speed = std::max(0.1, config.max_id_angular_speed);
  config.max_zero_phase_innovation_rad = std::max(0.0, config.max_zero_phase_innovation_rad);
  config.id_coast_timeout_sec = std::max(0.0, config.id_coast_timeout_sec);
  config.id_reset_timeout_sec = std::max(config.id_coast_timeout_sec, config.id_reset_timeout_sec);
  config.model_min_confirmed_frames = std::max(1, config.model_min_confirmed_frames);
  config.model_max_nis = std::max(0.0, config.model_max_nis);
  config.small_huber_delta = std::max(1e-6, config.small_huber_delta);
  config.big_a_center = std::clamp(config.big_a_center, config.big_a_min, config.big_a_max);
  config.big_w_center = std::clamp(config.big_w_center, config.big_w_min, config.big_w_max);
  config.big_center_prior_weight = std::max(0.0, config.big_center_prior_weight);
  config.big_huber_delta = std::max(1e-6, config.big_huber_delta);
  config.big_omega_a = std::max(0.0, config.big_omega_a);
  config.big_omega_w = std::max(0.0, config.big_omega_w);

  return config;
}

// ===== Lifetime =====

AngleEstimate::AngleEstimate(const AngleEstimateConfig & config) : config_(config)
{
  fit_worker_ = std::thread(&AngleEstimate::fitWorkerLoop, this);
}

AngleEstimate::AngleEstimate(
  double max_match_angle, double min_time_change, int outliers_threshold, double sigma_multiple)
{
  config_.max_match_angle = max_match_angle;
  config_.min_time_change = min_time_change;
  config_.outliers_threshold = outliers_threshold;
  config_.sigma_multiple = sigma_multiple;
  fit_worker_ = std::thread(&AngleEstimate::fitWorkerLoop, this);
}

AngleEstimate::~AngleEstimate()
{
  {
    std::lock_guard<std::mutex> lock(fit_mutex_);
    fit_worker_quit_ = true;
    pending_fit_job_.reset();
  }
  fit_cv_.notify_all();
  if (fit_worker_.joinable()) fit_worker_.join();
}

// ===== Main update flow =====

void AngleEstimate::init(double leaf_angle, double time, BuffMode mode)
{
  resetTrackingState(leaf_angle, time, mode, false);
}

void AngleEstimate::resetTrackingState(
  double leaf_angle, double time, BuffMode mode, bool preserve_locked_direction)
{
  {
    std::lock_guard<std::mutex> lock(fit_mutex_);
    ++fit_epoch_;
    pending_fit_job_.reset();
    pending_fit_result_.reset();
  }
  fit_cv_.notify_all();

  // ID重置以及模型重置，ceres拟合的c、b设为第一帧角度，保留已经判断的旋转方向
  ++id_epoch_;
  detect_leaf_id_ = 0;
  last_confirmed_time_abs_ = time;
  last_observation_time_abs_ = time;
  model_epoch_ = 0;
  last_model_sample_time_abs_ = std::numeric_limits<double>::quiet_NaN();
  confirmed_observations_in_epoch_ = 1;
  last_id_match_debug_ = {};
  last_ceres_fit_debug_ = {};
  lost_count_ = 0;
  estimate_state_ = State::WAITTING;
  start_time_ = time;
  const int previous_direction = energy_tri_.direction;
  const bool previous_locked = direction_locked_;
  energy_tri_ = EnergyTri{};
  energy_tri_.mode = mode;
  energy_tri_.c = leaf_angle;
  energy_tri_.b = leaf_angle;
  energy_tri_.t0 = 0.0;
  if (preserve_locked_direction && previous_locked) {
    energy_tri_.direction = previous_direction;
    direction_locked_ = true;
    tools::logger()->info(
      "[DirectionLock] resetTracking preserved: direction={} preserve_flag={}",
      previous_direction, preserve_locked_direction);
  } else {
    direction_locked_ = false;
    tools::logger()->info(
      "[DirectionLock] resetTracking cleared: prev_direction={} prev_locked={} preserve_flag={}",
      previous_direction, previous_locked, preserve_locked_direction);
  }
  direction_sum_ = 0.0;
  leaf_datas_.clear(); // 清空队列
  leaf_datas_.push_back(
    {leaf_angle, leaf_angle, 0.0, detect_leaf_id_});  // 将第一帧目标扇叶作为0号扇叶
  ++latest_sample_version_;
  submitted_sample_version_ = 0;
  committed_fit_version_ = 0;
  last_fit_submit_t_ = std::chrono::steady_clock::time_point::min();
  deterministic_last_fit_time_abs_ = std::numeric_limits<double>::quiet_NaN();
  fitted_ = false;
  ever_fit_ = false;
  average_cost_ = 0.0;
  last_ceres_solve_time_ms_ = 0.0;
  last_small_ekf_update_time_ms_ = 0.0;
  has_ceres_solve_time_ = false;
  fitting_long_time_count_ = 0;
  fitting_start_rel_time_ = 0.0;
  if (mode == BuffMode::SMALL && useTargetEkf()) {
    const bool can_preserve_model =
      preserve_locked_direction && previous_locked;

    if (can_preserve_model) {
      // ID epoch重建，但旋转方向仍然可信。
      // 如果原EKF存在，就只重置角度和时间原点，保留速度与协方差。
      if (small_ekf_.initialized()) {
        small_ekf_.reanchor(leaf_angle, 0.0);
      } else {
        // 理论上较少出现，但做防御性处理：
        // 方向已经确定而EKF不存在，则按正确方向重新初始化。
        small_ekf_.reset(
          leaf_angle, 0.0, config_, energy_tri_.direction);
      }

      fitted_ = true;
      ever_fit_ = true;
      estimate_state_ = State::FITTED;
      model_epoch_ = id_epoch_;
      last_model_sample_time_abs_ = time;
    } else {
      // 首次进入小符，或方向不能继续沿用。
      // 此时绝对不能用默认direction初始化EKF。
      small_ekf_ = SmallRuneEkf{};

      fitted_ = false;
      ever_fit_ = false;
      estimate_state_ = State::WAITTING;
      model_epoch_ = 0;
      last_model_sample_time_abs_ =
        std::numeric_limits<double>::quiet_NaN();
    }
  } else {
    small_ekf_ = SmallRuneEkf{};
  }
}

// 用于帧间隔过长时重置ID
AngleEstimate::IdMatchResult AngleEstimate::reinitializeIdEpoch(
  double leaf_angle, double time, BuffMode mode)
{
  // 短时间（掉帧/遮挡）方向可信，保留；长时间（切模式/停转）方向可能已反转，强制重新判断
  constexpr double kDirectionPreserveMaxSec = 5.0;
  const double gap = time - last_confirmed_time_abs_;
  const bool preserve = direction_locked_ && std::isfinite(gap) && gap > 0.0 &&
                        gap <= kDirectionPreserveMaxSec;
  if (direction_locked_ && !preserve) {
    tools::logger()->info(
      "[DirectionLock] reinitializeIdEpoch direction DROPPED: gap={:.1f}s > {:.1f}s, was direction={}",
      gap, kDirectionPreserveMaxSec, energy_tri_.direction);
  }
  resetTrackingState(leaf_angle, time, mode, preserve);

  IdMatchResult result;
  result.status = IdMatchStatus::INITIALIZED;
  result.leaf_id = detect_leaf_id_;
  result.aligned_angle = leaf_angle;
  result.best_residual = 0.0;
  result.epoch = id_epoch_;
  return result;
}

AngleEstimate::IdMatchResult AngleEstimate::update(double leaf_angle, double time)
{
  return update(leaf_angle, time, energy_tri_.mode);
}

AngleEstimate::IdMatchResult AngleEstimate::update(double leaf_angle, double time, BuffMode mode)
{
  last_ceres_fit_debug_ = {};
  if (estimate_state_ == State::LOST || energy_tri_.mode != mode) {
    tools::logger()->info(
      "[DirectionLock] update triggered INIT: state={} old_mode={} new_mode={} direction_locked={}",
      estimate_state_ == State::LOST ? "LOST" : "FITTING/FITTED",
      energy_tri_.mode == BuffMode::BIG ? "BIG" : (energy_tri_.mode == BuffMode::SMALL ? "SMALL" : "LOST"),
      mode == BuffMode::BIG ? "BIG" : (mode == BuffMode::SMALL ? "SMALL" : "LOST"),
      direction_locked_);
    init(leaf_angle, time, mode);
    IdMatchResult result;
    result.status = IdMatchStatus::INITIALIZED;
    result.leaf_id = detect_leaf_id_;
    result.aligned_angle = leaf_angle;
    result.best_residual = 0.0;
    result.epoch = id_epoch_;
    return result;
  }

  // 检查绝对时间，不能比之前早
  if (std::isfinite(last_observation_time_abs_) && time < last_observation_time_abs_ - 1e-6) {
    IdMatchResult result;
    result.status = IdMatchStatus::OUT_OF_ORDER;
    result.epoch = id_epoch_;
    return result;
  }
  last_observation_time_abs_ = time;

  // 检查相对时间，因为拟合的时间是用相对时间来的
  const double relative_time = time - start_time_;
  if (!leaf_datas_.empty() && relative_time < leaf_datas_.back().t - 1e-6) {
    IdMatchResult result;
    result.status = IdMatchStatus::OUT_OF_ORDER;
    result.epoch = id_epoch_;
    return result;
  }

  // 检查时间间隔，过长则重置ID epoch
  if (
    std::isfinite(last_confirmed_time_abs_) && time >= last_confirmed_time_abs_ &&
    time - last_confirmed_time_abs_ > config_.id_reset_timeout_sec) {
    tools::logger()->info(
      "[DirectionLock] reinitializeIdEpoch: dt={:.3f}s timeout={:.3f}s direction_locked={}",
      time - last_confirmed_time_abs_, config_.id_reset_timeout_sec, direction_locked_);
    return reinitializeIdEpoch(leaf_angle, time, mode);
  }

  // ID匹配
  auto match = updateLeafId(leaf_angle, relative_time);
  if (!match.confirmed()) {
    ++lost_count_;
    return match;
  }
  lost_count_ = 0;
  last_confirmed_time_abs_ = time;
  ++confirmed_observations_in_epoch_;

  // 生成零相位
  const double aligned_angle = match.aligned_angle;
  const double curr_zero = aligned_angle - detect_leaf_id_ * 2.0 * M_PI / 5.0;
  last_id_match_debug_.current_zero_angle = curr_zero;
  if (!leaf_datas_.empty()) {
    last_id_match_debug_.zero_innovation =
      shortest_angular_distance(leaf_datas_.back().zero_angle, curr_zero);
  }
  if (!leaf_datas_.empty() && std::abs(relative_time - leaf_datas_.back().t) < 1e-6) {
    // 同时间戳融合（大符有多扇叶情况）
    last_id_match_debug_.fused_same_timestamp = true;
    auto & last = leaf_datas_.back();
    const double fused_zero =
      last.zero_angle + 0.5 * shortest_angular_distance(last.zero_angle, curr_zero);
    last.zero_angle = fused_zero;
    last.angle = fused_zero + last.leaf_id * 2.0 * M_PI / 5.0;
    ++latest_sample_version_;
  } else {
    // 进入队列

    std::cout << "zero_angle:" << curr_zero << std::endl;

    leaf_datas_.push_back({aligned_angle, curr_zero, relative_time, detect_leaf_id_});
    ++latest_sample_version_;
    const auto max_len = static_cast<std::size_t>(
      mode == BuffMode::BIG ? config_.big_queue_size : config_.small_queue_size);
    while (leaf_datas_.size() > max_len) leaf_datas_.pop_front();
  }

  // EKF OR CERES，只有小符才有EKF
  if (mode == BuffMode::SMALL && useTargetEkf()) {
    updateDirectionLock();  // EKF 与 Ceres 共用同一判据
    const auto ekf_update_start = std::chrono::steady_clock::now();
    if (!small_ekf_.initialized() && direction_locked_) {
      // 方向刚锁定：速度先验符号已知，此时初始化
      small_ekf_.reset(curr_zero, relative_time, config_, energy_tri_.direction);
    } else if (small_ekf_.initialized()) {
      small_ekf_.update(curr_zero, relative_time, config_);
    }
    last_small_ekf_update_time_ms_ =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ekf_update_start)
        .count();
    if (small_ekf_.initialized()) {
      fitted_ = true;
      ever_fit_ = true; 
      estimate_state_ = State::FITTED;
      model_epoch_ = id_epoch_;
      last_model_sample_time_abs_ = time;
    }
    return match;
  }
  updateState(relative_time);
  return match;
}

// 扇叶 id 匹配：闭式最近邻 + 连续性门 + 可选模型先验。
// 写入 detect_leaf_id_ 与 last_id_match_debug_；leaf_datas_ 由 update() 负责写入。
AngleEstimate::IdMatchResult AngleEstimate::updateLeafId(double angle, double time)
{
  // debug信息
  IdMatchResult result;
  result.epoch = id_epoch_;
  IdMatchDebug debug;
  debug.attempted = true;
  debug.previous_leaf_id = detect_leaf_id_;
  debug.time = time;
  debug.observed_angle = angle;
  debug.expected_angles.fill(std::numeric_limits<double>::infinity());
  debug.residuals.fill(std::numeric_limits<double>::infinity());
  debug.model_expected_angles.fill(std::numeric_limits<double>::infinity());
  debug.model_residuals.fill(std::numeric_limits<double>::infinity());
  debug.model_epoch = model_epoch_;
  debug.confirmed_observations_in_epoch = confirmed_observations_in_epoch_;
  debug.history_size = static_cast<int>(leaf_datas_.size());
  if (!leaf_datas_.empty()) {
    const auto & last = leaf_datas_.back();
    debug.history_last_leaf_id = last.leaf_id;
    debug.history_last_angle = last.angle;
    debug.history_last_zero_angle = last.zero_angle;
    debug.extrapolation_dt = std::max(time - last.t, 0.0);
  }
  if (leaf_datas_.size() >= 2) {
    const auto & prev = leaf_datas_[leaf_datas_.size() - 2];
    debug.history_previous_leaf_id = prev.leaf_id;
    debug.history_previous_angle = prev.angle;
    debug.history_previous_zero_angle = prev.zero_angle;
  }

  if (leaf_datas_.empty()) {
    last_id_match_debug_ = debug;
    return result;
  }

  // ---- 步骤 1：用两帧差分推算当前帧的零号扇叶相位 ----
  const auto & last = leaf_datas_.back();
  const double dt2 = std::max(time - last.t, 0.0);
  double predicted_zero = last.zero_angle;

  if (leaf_datas_.size() < 2) {
    debug.short_history = true;
  } else {
    const auto & prev = leaf_datas_[leaf_datas_.size() - 2];
    const double dt1 = last.t - prev.t;
    const double zero_diff = shortest_angular_distance(prev.zero_angle, last.zero_angle);
    debug.history_dt = dt1;
    debug.estimated_zero_delta = zero_diff;

    if (dt1 >= config_.min_time_change) {
      const double angular_speed = zero_diff / dt1;
      // 小符约 1.05 rad/s，大符最大约 2.09 rad/s，4 rad/s 留足余量
      if (std::isfinite(angular_speed) && std::abs(angular_speed) <= config_.max_id_angular_speed) {
        predicted_zero += angular_speed * dt2;
      }
    }
  }

  // ---- 步骤 2：闭式最近邻匹配 ----
  // 给定预测零相位，用「观测偏差 / 格距四舍五入」直接得到最近 id，
  // 数学上与穷举 5 候选取最小残差等价，但只需一次除法+取整。
  const double innov_limit = config_.max_zero_phase_innovation_rad
                             + config_.max_id_angular_speed * dt2;

  // 返回 (id, aligned_angle, |residual|, zero_innov)，同时填 debug 数组
  const auto match_from_zero = [&](
    double zero_angle,
    std::array<double, 5> & dbg_expected,
    std::array<double, 5> & dbg_residuals)
    -> std::tuple<int, double, double, double>
  {
    for (int i = 0; i < 5; ++i) {
      dbg_expected[static_cast<std::size_t>(i)] = zero_angle + i * kLeafAngleStep;
    }

    // 偏差折算成格数，四舍五入即为最近 id
    const double phase = shortest_angular_distance(zero_angle, angle);
    int id = static_cast<int>(std::lround(phase / kLeafAngleStep)) % 5;
    if (id < 0) id += 5;

    const double expected = zero_angle + id * kLeafAngleStep;
    const double signed_residual = shortest_angular_distance(expected, angle);
    const double aligned_angle = expected + signed_residual;
    const double zero_innov = shortest_angular_distance(
      last.zero_angle, aligned_angle - id * kLeafAngleStep);

    for (int i = 0; i < 5; ++i) {
      const double r = shortest_angular_distance(dbg_expected[static_cast<std::size_t>(i)], angle);
      dbg_residuals[static_cast<std::size_t>(i)] = std::abs(r);
    }

    return {id, aligned_angle, std::abs(signed_residual), zero_innov};
  };

  const auto [kin_id, kin_aligned, kin_residual, kin_innov] =
    match_from_zero(predicted_zero, debug.expected_angles, debug.residuals);
  debug.kinematic_leaf_id = kin_id;

  // 连续性门：零相位跳变超限则拒绝（防止掉帧后错认 id）
  const bool kin_continuous = std::abs(kin_innov) <= innov_limit;

  // 次优残差的解析值：相邻候选恰好差一格，所以 second = 72° − |residual|
  const double kin_second = std::abs(kLeafAngleStep - kin_residual);

  // ---- 步骤 3：可选模型先验（use_model_prior = true 时启用） ----
  // 模型健康检查通过且结果更可信时，用模型预测的零相位覆盖运动学结果。
  int sel_id = kin_continuous ? kin_id : -1;
  double sel_aligned = kin_aligned;
  double sel_residual = kin_residual;
  double sel_second = kin_second;
  double sel_innov = kin_innov;

  if (config_.use_model_prior) {
    const double absolute_time = start_time_ + time;
    bool model_ok = false;

    if (model_epoch_ != id_epoch_) {
      debug.model_prior_status = IdModelPriorStatus::EPOCH_MISMATCH;
    } else if (!std::isfinite(last_model_sample_time_abs_)) {
      debug.model_prior_status = IdModelPriorStatus::STALE;
    } else {
      debug.model_age_sec = absolute_time - last_model_sample_time_abs_;
      if (!std::isfinite(debug.model_age_sec) || debug.model_age_sec < 0.0 ||
          debug.model_age_sec > config_.id_coast_timeout_sec) {
        debug.model_prior_status = IdModelPriorStatus::STALE;
      } else if (energy_tri_.mode == BuffMode::SMALL && useTargetEkf()) {
        if (confirmed_observations_in_epoch_ < config_.model_min_confirmed_frames) {
          debug.model_prior_status = IdModelPriorStatus::WARMING_UP;
        } else if (!small_ekf_.initialized()) {
          debug.model_prior_status = IdModelPriorStatus::EKF_STATE_INVALID;
        } else {
          const auto ekf = small_ekf_.debug();
          // P 矩阵正定检查（防止数值发散后给出伪置信预测）
          const bool cov_ok = std::isfinite(ekf.p00) && std::isfinite(ekf.p11) &&
                              ekf.p00 > 0.0 && ekf.p11 > 0.0 &&
                              ekf.p00 * ekf.p11 - ekf.p01 * ekf.p10 > 0.0;
          if (!cov_ok) {
            debug.model_prior_status = IdModelPriorStatus::EKF_STATE_INVALID;
          } else if (!std::isfinite(ekf.nis) || ekf.nis > config_.model_max_nis) {
            debug.model_prior_status = IdModelPriorStatus::EKF_NIS_INVALID;
          } else {
            model_ok = true;
          }
        }
      } else {
        const double cost_thresh = energy_tri_.mode == BuffMode::BIG
                                     ? config_.big_cost_threshold
                                     : config_.small_cost_threshold;
        if (estimate_state_ != State::FITTED || !fitted_ ||
            !std::isfinite(average_cost_) || average_cost_ > cost_thresh) {
          debug.model_prior_status = IdModelPriorStatus::CERES_COST_INVALID;
        } else {
          model_ok = true;
        }
      }
    }

    if (model_ok) {
      const double model_zero = predictZero(absolute_time);
      if (!std::isfinite(model_zero)) {
        debug.model_prior_status = IdModelPriorStatus::PREDICTION_INVALID;
      } else {
        const auto [mod_id, mod_aligned, mod_residual, mod_innov] =
          match_from_zero(model_zero, debug.model_expected_angles, debug.model_residuals);
        const double mod_second = std::abs(kLeafAngleStep - mod_residual);
        const bool mod_acceptable =
          mod_residual <= config_.max_match_angle &&
          mod_second - mod_residual >= config_.min_residual_margin &&
          std::abs(mod_innov) <= innov_limit;

        debug.model_available = true;
        debug.model_prior_status = IdModelPriorStatus::ACTIVE;
        debug.model_zero_angle = model_zero;
        debug.model_leaf_id = mod_id;
        debug.model_used = mod_acceptable;
        debug.model_overrode_kinematic = mod_acceptable && kin_continuous && mod_id != kin_id;

        if (mod_acceptable) {
          sel_id = mod_id;
          sel_aligned = mod_aligned;
          sel_residual = mod_residual;
          sel_second = mod_second;
          sel_innov = mod_innov;
        }
      }
    }
  }

  // ---- 步骤 4：最终门控与结果输出 ----
  debug.raw_best_leaf_id = kin_id;
  debug.raw_best_residual = kin_residual;
  debug.continuity_leaf_id = kin_continuous ? kin_id : -1;
  debug.zero_innovation_limit = innov_limit;
  debug.zero_innovation = sel_innov;

  const bool no_candidate = sel_id < 0;
  const bool residual_too_large = sel_residual > config_.max_match_angle;
  const bool ambiguous = sel_second - sel_residual < config_.min_residual_margin;

  // 无候选、最佳匹配的残差过大、最佳与次佳的区别不大
  if (no_candidate || residual_too_large || ambiguous) {
    result.status = ambiguous ? IdMatchStatus::AMBIGUOUS : IdMatchStatus::OUTLIER;
    result.leaf_id = -1;
    debug.continuity_rejected = no_candidate && !kin_continuous;
    debug.selected_leaf_id = -1;
    last_id_match_debug_ = debug;
    return result;
  }

  result.status = IdMatchStatus::MATCHED;
  result.leaf_id = sel_id;
  result.aligned_angle = sel_aligned;
  result.best_residual = sel_residual;
  result.second_best_residual = sel_second;

  debug.selected_leaf_id = sel_id;
  debug.best_residual = sel_residual;
  debug.accepted = true;
  debug.aligned_angle = sel_aligned;
  last_id_match_debug_ = debug;

  detect_leaf_id_ = sel_id;
  return result;
}

// 累计一定角度就锁定方向，当epoch重建时清零重新判断
void AngleEstimate::updateDirectionLock()
{
  if (direction_locked_ || leaf_datas_.size() < 2) return;

  const double direction_threshold = energy_tri_.mode == BuffMode::BIG
                                       ? config_.big_direction_lock_threshold
                                       : config_.small_direction_lock_threshold;
  const auto & latest = leaf_datas_.back();
  const auto & previous = leaf_datas_[leaf_datas_.size() - 2];
  const double dt = latest.t - previous.t;
  const double max_speed = (energy_tri_.mode == BuffMode::BIG) ? 2.2 : 1.2;  // rad/s，留余量
  const double max_expected_delta = max_speed * dt + 0.05;
  const double delta = shortest_angular_distance(previous.zero_angle, latest.zero_angle);
  // 排除异常帧间跳变，避免方向锁定被误触发
  if (!std::isfinite(delta) || std::abs(delta) > max_expected_delta) {
    tools::logger()->warn(
      "[DirectionLock] delta filtered: delta={:.4f} max_expected={:.4f} dt={:.4f} queue_size={}",
      delta, max_expected_delta, dt, leaf_datas_.size());
    return;
  }

  direction_sum_ += delta;
  tools::logger()->debug(
    "[DirectionLock] accumulating: delta={:.4f} sum={:.4f} threshold={:.2f} queue_size={}",
    delta, direction_sum_, direction_threshold, leaf_datas_.size());

  if (std::abs(direction_sum_) < direction_threshold) return;

  energy_tri_.direction = direction_sum_ > 0.0 ? 1 : -1;
  direction_locked_ = true;
  tools::logger()->info(
    "[DirectionLock] LOCKED: direction={} sum={:.4f} threshold={:.2f} queue_size={} mode={}",
    energy_tri_.direction, direction_sum_, direction_threshold, leaf_datas_.size(),
    energy_tri_.mode == BuffMode::BIG ? "BIG" : "SMALL");
}

void AngleEstimate::updateState(double relative_time)
{
  if (energy_tri_.mode == BuffMode::SMALL && useTargetEkf()) return;

  const int min_len =
    energy_tri_.mode == BuffMode::BIG ? config_.big_min_fit_size : config_.small_min_fit_size;
  updateDirectionLock();
  if (static_cast<int>(leaf_datas_.size()) < min_len || !direction_locked_) {
    estimate_state_ = State::WAITTING;
    return;
  }

  if (estimate_state_ == State::WAITTING) {
    estimate_state_ = State::FITTING;
    fitting_start_rel_time_ = relative_time;
  } else if (estimate_state_ == State::FITTING) {
    if (fitted_) {
      estimate_state_ = State::FITTED;
    } else if (relative_time - fitting_start_rel_time_ > config_.fitting_timeout_sec) {
      const int keep_len = energy_tri_.mode == BuffMode::BIG ? 200 : 120;
      while (static_cast<int>(leaf_datas_.size()) > keep_len) leaf_datas_.pop_front();
      fitting_long_time_count_++;
      if (fitting_long_time_count_ > 10 && !leaf_datas_.empty()) {
        const auto latest = leaf_datas_.back();
        leaf_datas_.clear();
        leaf_datas_.push_back(latest);
        estimate_state_ = State::WAITTING;
        fitting_long_time_count_ = 0;
      }
      fitting_start_rel_time_ = relative_time;
    }
  } else if (estimate_state_ == State::FITTED && !fitted_) {
    estimate_state_ = State::FITTING;
  }
}

// ===== Small rune EKF (小符 EKF) =====

void AngleEstimate::SmallRuneEkf::reset(
  double angle, double time, const AngleEstimateConfig & config, int direction)
{
  const bool initialized_before = initialized_;
  initialized_ = true;
  last_time_ = time;

  Eigen::Vector2d x0;
  x0 << angle, direction * SMALL_W;
  Eigen::Matrix2d P0;
  P0 << config.small_ekf_p0_angle, 0.0, 0.0, config.small_ekf_p0_speed;
  H_ << 1.0, 0.0;
  R_ << config.small_ekf_r_angle;

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = wrap_angle(c[0]);
    return c;
  };
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  last_debug_ = {};
  last_debug_.initialized_before = initialized_before;
  last_debug_.reset = true;
  last_debug_.input_angle = angle;
  last_debug_.time = time;
  last_debug_.aligned_measurement = angle;
  last_debug_.state_angle_after = ekf_.x[0];
  last_debug_.state_speed_after = ekf_.x[1];
  last_debug_.p00 = ekf_.P(0, 0);
  last_debug_.p01 = ekf_.P(0, 1);
  last_debug_.p10 = ekf_.P(1, 0);
  last_debug_.p11 = ekf_.P(1, 1);
}

// 用于ID epoch重建但方向和速度仍可信
void AngleEstimate::SmallRuneEkf::reanchor(double angle, double time)
{
  if (!initialized_) return;

  ekf_.x[0] = angle;
  last_time_ = time;
  last_debug_ = {};
  last_debug_.initialized_before = true;
  last_debug_.input_angle = angle;
  last_debug_.time = time;
  last_debug_.aligned_measurement = angle;
  last_debug_.state_angle_after = ekf_.x[0];
  last_debug_.state_speed_after = ekf_.x[1];
  last_debug_.p00 = ekf_.P(0, 0);
  last_debug_.p01 = ekf_.P(0, 1);
  last_debug_.p10 = ekf_.P(1, 0);
  last_debug_.p11 = ekf_.P(1, 1);
}

void AngleEstimate::SmallRuneEkf::update(
  double angle, double time, const AngleEstimateConfig & config)
{
  last_debug_ = {};
  last_debug_.initialized_before = initialized_;
  last_debug_.input_angle = angle;
  last_debug_.time = time;

  const double dt = time - last_time_;
  last_debug_.dt = dt;
  last_debug_.state_angle_before = ekf_.x[0];
  last_debug_.state_speed_before = ekf_.x[1];
  if (dt < config.min_time_change) {
    ekf_.x[0] += 0.5 * shortest_angular_distance(ekf_.x[0], angle);
    ekf_.x[0] = wrap_angle(ekf_.x[0]);
    last_time_ = time;
    last_debug_.short_dt_fusion = true;
    last_debug_.updated = true;
    last_debug_.aligned_measurement = angle;
    last_debug_.state_angle_after = ekf_.x[0];
    last_debug_.state_speed_after = ekf_.x[1];
    last_debug_.p00 = ekf_.P(0, 0);
    last_debug_.p01 = ekf_.P(0, 1);
    last_debug_.p10 = ekf_.P(1, 0);
    last_debug_.p11 = ekf_.P(1, 1);
    return;
  }

  double aligned_angle = angle;
  last_debug_.raw_innovation = shortest_angular_distance(ekf_.x[0], angle);
  if (std::abs(shortest_angular_distance(ekf_.x[0], aligned_angle)) > M_PI / 12.0) {
    for (int i = -5; i <= 5; ++i) {
      const double candidate = ekf_.x[0] + static_cast<double>(i) * 2.0 * M_PI / 5.0;
      if (std::abs(shortest_angular_distance(candidate, angle)) < M_PI / 5.0) {
        aligned_angle = candidate + shortest_angular_distance(candidate, angle);
        last_debug_.phase_realigned = i != 0;
        last_debug_.phase_shift = i;
        break;
      }
    }
  }
  last_debug_.aligned_measurement = aligned_angle;

  F_ << 1.0, dt, 0.0, 1.0;
  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  Q_ << dt4 / 4.0 * config.small_ekf_q_accel, dt3 / 2.0 * config.small_ekf_q_accel,
    dt3 / 2.0 * config.small_ekf_q_accel, dt2 * config.small_ekf_q_accel;

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F_ * x;
    x_prior[0] = wrap_angle(x_prior[0]);
    return x_prior;
  };
  ekf_.predict(F_, Q_, f);
  last_debug_.predicted_angle = ekf_.x[0];
  last_debug_.predicted_speed = ekf_.x[1];

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = wrap_angle(c[0]);
    return c;
  };

  Eigen::Matrix<double, 1, 1> z;
  z << aligned_angle;
  ekf_.update(z, H_, R_, z_subtract);
  last_time_ = time;
  last_debug_.updated = true;
  last_debug_.state_angle_after = ekf_.x[0];
  last_debug_.state_speed_after = ekf_.x[1];
  last_debug_.p00 = ekf_.P(0, 0);
  last_debug_.p01 = ekf_.P(0, 1);
  last_debug_.p10 = ekf_.P(1, 0);
  last_debug_.p11 = ekf_.P(1, 1);
  last_debug_.nis = ekf_.last_nis;
}

double AngleEstimate::SmallRuneEkf::predict(double time) const
{
  if (!initialized_) return 0.0;
  return ekf_.x[0] + ekf_.x[1] * (time - last_time_);
}

double AngleEstimate::SmallRuneEkf::speed() const
{
  if (!initialized_) return 0.0;
  return ekf_.x[1];
}

// ===== Ceres dispatch and async fitting =====

// 表示这次调用是否用了新结果
bool AngleEstimate::angleFit()
{
  if (!usesAsyncCeres()) return is_fitted();
  if (deterministic_replay_) return runDeterministicFitIfReady();

  const bool applied = collectFitResult();
  submitFitJobIfReady();
  return applied;
}

// 设置确定性回放开关，开启时不会收取已经取消的异步任务的结果，关闭时可能会收取结果
void AngleEstimate::setDeterministicReplay(bool enabled)
{
  deterministic_replay_ = enabled;
  deterministic_last_fit_time_abs_ = std::numeric_limits<double>::quiet_NaN();
  {
    std::lock_guard<std::mutex> lock(fit_mutex_);
    pending_fit_job_.reset();
    pending_fit_result_.reset();
  }
  if (enabled) {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_.ceres_threads = 1;
  }
}

// 拟合核心函数
bool AngleEstimate::angleFitOnSnapshot(
  const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot, State state_snapshot,
  AngleFitResult & result) const
{
  return angleFitOnSnapshot(leaf_datas, tri_snapshot, state_snapshot, ever_fit_, result);
}

// 保存最新样本时间和零相位，然后把数据发给ceres拟合
bool AngleEstimate::angleFitOnSnapshot(
  const std::deque<LeafData> & leaf_datas, const EnergyTri & tri_snapshot, State state_snapshot,
  bool ever_fit_snapshot, AngleFitResult & result) const
{
  AngleEstimateConfig config_snapshot;
  {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_snapshot = config_;
  }

  result = {};
  result.mode = tri_snapshot.mode;
  result.energy_tri = tri_snapshot;
  if (leaf_datas.empty()) return false;
  if (
    tri_snapshot.mode == BuffMode::SMALL &&
    config_snapshot.small_predictor == SmallRunePredictor::TARGET_EKF) {
    return false;
  }

  if (tri_snapshot.mode == BuffMode::BIG) {
    const bool ok = fit_big_on_snapshot(
      leaf_datas, tri_snapshot, state_snapshot, ever_fit_snapshot, config_snapshot, result);
    result.latest_t = leaf_datas.back().t;
    result.latest_zero_angle = leaf_datas.back().zero_angle;
    return ok;
  }

  if (tri_snapshot.mode == BuffMode::SMALL) {
    const bool ok = fit_small_on_snapshot(leaf_datas, tri_snapshot, config_snapshot, result);
    result.latest_t = leaf_datas.back().t;
    result.latest_zero_angle = leaf_datas.back().zero_angle;
    return ok;
  }

  return false;
}

// 拟合成功时复制拟合模型
void AngleEstimate::applyAngleFitResult(const AngleFitResult & result)
{
  if (!result.success) return;
  const bool had_previous_fit = fitted_;
  EnergyTri fitted_model = result.energy_tri;
  // 方向由 updateDirectionLock 统一管理;拟合结果必须服从已锁定方向
  const int fit_direction = result.energy_tri.direction;
  if (direction_locked_) {
    fitted_model.direction = energy_tri_.direction;
    if (fit_direction != energy_tri_.direction) {
      tools::logger()->warn(
        "[DirectionLock] applyFit OVERRIDE: fit_direction={} overridden_by_locked={}",
        fit_direction, energy_tri_.direction);
    }
  } else {
    fitted_model.direction = result.energy_tri.direction >= 0 ? 1 : -1;
    tools::logger()->debug(
      "[DirectionLock] applyFit from result: direction={}", fitted_model.direction);
  }

  // leaf-snap: 修正整叶偏移，修正大符和小符 Ceres 拟合时扇叶整体偏移
  double phase_reference_time = result.latest_t;
  double phase_reference_zero_angle = result.latest_zero_angle;

  if (!leaf_datas_.empty()) {
    phase_reference_time = leaf_datas_.back().t;
    phase_reference_zero_angle = leaf_datas_.back().zero_angle;
  }

  if (
    std::isfinite(phase_reference_time) &&
    std::isfinite(phase_reference_zero_angle)) {
    const double model_zero = fitted_model.getAngle(phase_reference_time);
    const double diff = wrap_angle(phase_reference_zero_angle - model_zero);
    const int k = static_cast<int>(std::round(diff / kLeafAngleStep));

    if (fitted_model.mode == BuffMode::BIG && k != 0) {
      fitted_model.c +=
        static_cast<double>(k) * kLeafAngleStep / static_cast<double>(fitted_model.direction);
    } else if (fitted_model.mode == BuffMode::SMALL && k != 0) {
      fitted_model.b +=
        static_cast<double>(k) * kLeafAngleStep / static_cast<double>(fitted_model.direction);
    }

    fitted_model = alignFitModelPhase(
      fitted_model, phase_reference_time, phase_reference_zero_angle);
  }

  energy_tri_ = fitted_model;
  average_cost_ = result.average_cost;
  fitted_ = true;
  ever_fit_ = true;
  estimate_state_ = State::FITTED;
  model_epoch_ = id_epoch_;
  last_model_sample_time_abs_ = std::isfinite(result.latest_t)
                                  ? start_time_ + result.latest_t
                                  : std::numeric_limits<double>::quiet_NaN();
}

// 新一轮拟合可能改变模型参数，导致最新时刻预测角跳变，用于滤波，防止跳变（未使用）
EnergyTri AngleEstimate::alignFitModelPhase(
  EnergyTri fitted_model, double reference_time, double reference_zero_angle) const
{
  if (!std::isfinite(reference_time) || !std::isfinite(reference_zero_angle))
    return fitted_model;

  const double fitted_phase = fitted_model.getAngle(reference_time);
  const double phase_fix = wrap_angle(reference_zero_angle - fitted_phase);

  //只修正亚叶级别的噪声；整叶偏移由leaf-snap 处理
  if (std::abs(phase_fix) >= kLeafAngleStep * 0.5) return fitted_model;

  if (fitted_model.mode == BuffMode::BIG) {
    fitted_model.c += phase_fix / static_cast<double>(fitted_model.direction);
  } else if (fitted_model.mode == BuffMode::SMALL) {
    fitted_model.b += phase_fix / static_cast<double>(fitted_model.direction);
  }
  return fitted_model;
}

// 获取拟合结果
bool AngleEstimate::collectFitResult()
{
  std::optional<FitResult> fit_result;
  {
    std::lock_guard<std::mutex> lock(fit_mutex_);
    if (!pending_fit_result_) return false;
    fit_result = pending_fit_result_;
    pending_fit_result_.reset();
  }

  last_ceres_fit_debug_.result_received = true;
  last_ceres_fit_debug_.result_epoch = fit_result->epoch;
  last_ceres_fit_debug_.result_sample_version = fit_result->sample_version;
  last_ceres_fit_debug_.result_sample_count = fit_result->sample_count;
  last_ceres_fit_debug_.result_first_sample_time = fit_result->first_sample_time;
  last_ceres_fit_debug_.result_last_sample_time = fit_result->last_sample_time;
  last_ceres_fit_debug_.previous_k = energy_tri_.k;
  last_ceres_fit_debug_.previous_b = energy_tri_.b;
  last_ceres_fit_debug_.result_k = fit_result->result.energy_tri.k;
  last_ceres_fit_debug_.result_b = fit_result->result.energy_tri.b;
  last_ceres_fit_debug_.average_cost = fit_result->result.average_cost;
  last_ceres_fit_debug_.solve_time_ms = fit_result->result.solve_time_ms;
  last_ceres_fit_debug_.total_fit_wall_ms = fit_result->result.total_fit_wall_ms;
  last_ceres_fit_debug_.residual_count = fit_result->result.residual_count;

  if (
    fit_result->epoch != fit_epoch_ || fit_result->result.mode != energy_tri_.mode ||
    estimate_state_ == State::LOST || fit_result->sample_version <= committed_fit_version_) {
    last_ceres_fit_debug_.result_rejected = true;
    return false;
  }

  last_ceres_solve_time_ms_ = fit_result->result.solve_time_ms;
  has_ceres_solve_time_ = true;
  if (!fit_result->fit_ok || !fit_result->result.success) {
    last_ceres_fit_debug_.result_rejected = true;
    tools::logger()->warn(
      "[DirectionLock] fitResult REJECTED: fit_ok={} success={} cost={:.4f} fit_direction={} locked_direction={}",
      fit_result->fit_ok, fit_result->result.success, fit_result->result.average_cost,
      fit_result->result.energy_tri.direction, energy_tri_.direction);
    return false;
  }

  tools::logger()->debug(
    "[DirectionLock] fitResult applying: fit_direction={} current_locked_dir={} locked={} cost={:.4f}",
    fit_result->result.energy_tri.direction, energy_tri_.direction, direction_locked_,
    fit_result->result.average_cost);

  applyAngleFitResult(fit_result->result);
  committed_fit_version_ = fit_result->sample_version;
  last_ceres_fit_debug_.result_applied = true;
  return true;
}

// 提交异步任务前检查，保证每次只保留最新一份待处理数据
bool AngleEstimate::submitFitJobIfReady()
{
  if (!usesAsyncCeres()) return false;
  if (leaf_datas_.empty()) return false;
  if (estimate_state_ != State::FITTING && estimate_state_ != State::FITTED) return false;
  if (latest_sample_version_ <= submitted_sample_version_) return false;

  const auto now = std::chrono::steady_clock::now();
  if (!shouldSubmitFitJob(now)) return false;

  FitJob job;
  {
    std::lock_guard<std::mutex> lock(fit_mutex_);
    if (fit_worker_busy_ && pending_fit_job_) {
      pending_fit_job_->epoch = fit_epoch_;
      pending_fit_job_->sample_version = latest_sample_version_;
      pending_fit_job_->leaf_datas = leaf_datas_;
      pending_fit_job_->energy_tri = energy_tri_;
      pending_fit_job_->state = estimate_state_;
      pending_fit_job_->ever_fit = ever_fit_;
    } else {
      pending_fit_job_ = FitJob{fit_epoch_,  latest_sample_version_, leaf_datas_,
                                energy_tri_, estimate_state_,        ever_fit_};
    }
  }

  submitted_sample_version_ = latest_sample_version_;
  last_fit_submit_t_ = now;
  last_ceres_fit_debug_.job_submitted = true;
  last_ceres_fit_debug_.submitted_epoch = fit_epoch_;
  last_ceres_fit_debug_.submitted_sample_version = latest_sample_version_;
  last_ceres_fit_debug_.submitted_sample_count = static_cast<int>(leaf_datas_.size());
  last_ceres_fit_debug_.submitted_first_sample_time = leaf_datas_.front().t;
  last_ceres_fit_debug_.submitted_last_sample_time = leaf_datas_.back().t;
  last_ceres_fit_debug_.previous_k = energy_tri_.k;
  last_ceres_fit_debug_.previous_b = energy_tri_.b;
  fit_cv_.notify_one();
  tools::logger()->debug(
    "[DirectionLock] fitJob submitted: direction={} locked={} sample_count={}",
    energy_tri_.direction, direction_locked_, leaf_datas_.size());
  return true;
}

// 同步时用，不是ceres和ekf使用，可以回放调试用
bool AngleEstimate::runDeterministicFitIfReady()
{
  if (leaf_datas_.empty()) return false;
  if (estimate_state_ != State::FITTING && estimate_state_ != State::FITTED) return false;
  if (latest_sample_version_ <= submitted_sample_version_) return false;
  if (!std::isfinite(last_observation_time_abs_)) return false;

  const double period_sec = ever_fit_ ? 0.05 : 0.01;
  if (
    std::isfinite(deterministic_last_fit_time_abs_) &&
    last_observation_time_abs_ - deterministic_last_fit_time_abs_ + 1e-9 < period_sec) {
    return false;
  }

  deterministic_last_fit_time_abs_ = last_observation_time_abs_;
  submitted_sample_version_ = latest_sample_version_;
  last_ceres_fit_debug_.job_submitted = true;
  last_ceres_fit_debug_.submitted_epoch = fit_epoch_;
  last_ceres_fit_debug_.submitted_sample_version = latest_sample_version_;
  last_ceres_fit_debug_.submitted_sample_count = static_cast<int>(leaf_datas_.size());
  last_ceres_fit_debug_.submitted_first_sample_time = leaf_datas_.front().t;
  last_ceres_fit_debug_.submitted_last_sample_time = leaf_datas_.back().t;
  last_ceres_fit_debug_.previous_k = energy_tri_.k;
  last_ceres_fit_debug_.previous_b = energy_tri_.b;

  AngleFitResult result;
  const bool fit_ok =
    angleFitOnSnapshot(leaf_datas_, energy_tri_, estimate_state_, ever_fit_, result);
  last_ceres_fit_debug_.result_received = true;
  last_ceres_fit_debug_.result_epoch = fit_epoch_;
  last_ceres_fit_debug_.result_sample_version = latest_sample_version_;
  last_ceres_fit_debug_.result_sample_count = static_cast<int>(leaf_datas_.size());
  last_ceres_fit_debug_.result_first_sample_time = leaf_datas_.front().t;
  last_ceres_fit_debug_.result_last_sample_time = leaf_datas_.back().t;
  last_ceres_fit_debug_.result_k = result.energy_tri.k;
  last_ceres_fit_debug_.result_b = result.energy_tri.b;
  last_ceres_fit_debug_.average_cost = result.average_cost;
  last_ceres_fit_debug_.solve_time_ms = result.solve_time_ms;
  last_ceres_fit_debug_.total_fit_wall_ms = result.total_fit_wall_ms;
  last_ceres_fit_debug_.residual_count = result.residual_count;
  last_ceres_solve_time_ms_ = result.solve_time_ms;
  has_ceres_solve_time_ = true;

  if (!fit_ok || !result.success || result.mode != energy_tri_.mode) {
    last_ceres_fit_debug_.result_rejected = true;
    return false;
  }

  applyAngleFitResult(result);
  committed_fit_version_ = latest_sample_version_;
  last_ceres_fit_debug_.result_applied = true;
  return true;
}

// 设置异步频率
bool AngleEstimate::shouldSubmitFitJob(std::chrono::steady_clock::time_point now) const
{
  constexpr auto kBeforeFirstFitPeriod =
    std::chrono::milliseconds(10);  // 100 Hz（第一次拟合时频率高一点，更快得到拟合结果）
  constexpr auto kAfterFirstFitPeriod =
    std::chrono::milliseconds(20);  // 20 Hz（第一次拟合成功后将频率降低，降低CPU占用）
  const auto period = ever_fit_ ? kAfterFirstFitPeriod : kBeforeFirstFitPeriod;
  return last_fit_submit_t_ == std::chrono::steady_clock::time_point::min() ||
         now - last_fit_submit_t_ >= period;
}

// ceres拟合线程函数
void AngleEstimate::fitWorkerLoop()
{
  while (true) {
    FitJob job;
    {
      std::unique_lock<std::mutex> lock(fit_mutex_);
      fit_cv_.wait(lock, [this] { return fit_worker_quit_ || pending_fit_job_.has_value(); });
      if (fit_worker_quit_) return;
      job = std::move(*pending_fit_job_);
      pending_fit_job_.reset();
      fit_worker_busy_ = true;
    }

    // 耗时长的拟合主函数放在互斥锁外
    AngleFitResult result;
    const bool fit_ok =
      angleFitOnSnapshot(job.leaf_datas, job.energy_tri, job.state, job.ever_fit, result);

    {
      std::lock_guard<std::mutex> lock(fit_mutex_);
      pending_fit_result_ = FitResult{
        job.epoch,
        job.sample_version,
        static_cast<int>(job.leaf_datas.size()),
        job.leaf_datas.empty() ? std::numeric_limits<double>::quiet_NaN()
                               : job.leaf_datas.front().t,
        job.leaf_datas.empty() ? std::numeric_limits<double>::quiet_NaN() : job.leaf_datas.back().t,
        result,
        fit_ok};
      fit_worker_busy_ = false;
    }
  }
}

// ===== Prediction and configuration helpers =====

double AngleEstimate::predict(double time) const { return predict(time, detect_leaf_id_); }

double AngleEstimate::predict(double time, int id) const
{
  if (estimate_state_ == State::LOST) return 0.0;
  return predictZero(time) + static_cast<double>(id) * 2.0 * M_PI / 5.0;
}

double AngleEstimate::predictDelta(double from_time, double to_time) const
{
  if (estimate_state_ == State::LOST) return 0.0;
  return predictZero(to_time) - predictZero(from_time);
}

double AngleEstimate::predictZero(double time) const
{
  const double relative_time = time - start_time_;
  if (energy_tri_.mode == BuffMode::SMALL && useTargetEkf()) {
    return small_ekf_.predict(relative_time);
  }
  return energy_tri_.getAngle(relative_time);
}

double AngleEstimate::predictSpeed(double time) const
{
  const double relative_time = time - start_time_;
  if (energy_tri_.mode == BuffMode::SMALL && useTargetEkf()) {
    return small_ekf_.speed();
  }
  return energy_tri_.getSpeed(relative_time);
}

AngleEstimate::Snapshot AngleEstimate::snapshot() const
{
  AngleEstimateConfig config_snapshot;
  {
    std::lock_guard<std::mutex> lock(config_mutex_);
    config_snapshot = config_;
  }
  return {
    estimate_state_,
    energy_tri_,
    average_cost_,
    last_ceres_solve_time_ms_,
    last_small_ekf_update_time_ms_,
    leaf_datas_.size(),
    fitted_,
    has_ceres_solve_time_,
    currentPredictorBackend(),
    config_snapshot.small_predictor};
}

AngleEstimate::DebugSnapshot AngleEstimate::debug_snapshot() const
{
  DebugSnapshot debug;
  debug.summary = snapshot();
  debug.id_epoch = id_epoch_;
  debug.model_epoch = model_epoch_;
  debug.last_model_sample_time_abs = last_model_sample_time_abs_;
  debug.confirmed_observations_in_epoch = confirmed_observations_in_epoch_;
  debug.current_leaf_id = detect_leaf_id_;
  debug.lost_count = lost_count_;
  debug.direction_locked = direction_locked_;
  debug.direction = direction_locked_ ? energy_tri_.direction : 0;
  debug.direction_sum = direction_sum_;
  debug.start_time = start_time_;
  debug.latest_sample_version = latest_sample_version_;
  debug.submitted_sample_version = submitted_sample_version_;
  debug.committed_fit_version = committed_fit_version_;
  debug.id_match = last_id_match_debug_;
  debug.small_ekf = small_ekf_.debug();
  debug.ceres_fit = last_ceres_fit_debug_;
  debug.recent_leaf_data_count = std::min<int>(
    static_cast<int>(leaf_datas_.size()), static_cast<int>(debug.recent_leaf_datas.size()));
  for (int i = 0; i < debug.recent_leaf_data_count; ++i) {
    debug.recent_leaf_datas[static_cast<std::size_t>(i)] =
      leaf_datas_[leaf_datas_.size() - static_cast<std::size_t>(debug.recent_leaf_data_count) + i];
  }
  return debug;
}

bool AngleEstimate::updateConfig(const AngleEstimateConfig & config)
{
  auto sanitized = config;
  sanitized.max_match_angle = std::clamp(sanitized.max_match_angle, 0.01, M_PI / 5.0);
  sanitized.min_residual_margin = std::max(0.0, sanitized.min_residual_margin);
  sanitized.max_id_angular_speed = std::max(0.1, sanitized.max_id_angular_speed);
  sanitized.max_zero_phase_innovation_rad = std::max(0.0, sanitized.max_zero_phase_innovation_rad);
  sanitized.id_coast_timeout_sec = std::max(0.0, sanitized.id_coast_timeout_sec);
  sanitized.id_reset_timeout_sec =
    std::max(sanitized.id_coast_timeout_sec, sanitized.id_reset_timeout_sec);
  sanitized.model_min_confirmed_frames = std::max(1, sanitized.model_min_confirmed_frames);
  sanitized.model_max_nis = std::max(0.0, sanitized.model_max_nis);
  sanitized.big_direction_lock_threshold =
    std::clamp(sanitized.big_direction_lock_threshold, 0.05, M_PI);
  sanitized.small_direction_lock_threshold =
    std::clamp(sanitized.small_direction_lock_threshold, 0.05, M_PI);
  sanitized.small_huber_delta = std::max(1e-6, sanitized.small_huber_delta);
  sanitized.big_a_center =
    std::clamp(sanitized.big_a_center, sanitized.big_a_min, sanitized.big_a_max);
  sanitized.big_w_center =
    std::clamp(sanitized.big_w_center, sanitized.big_w_min, sanitized.big_w_max);
  sanitized.big_center_prior_weight = std::max(0.0, sanitized.big_center_prior_weight);
  sanitized.big_huber_delta = std::max(1e-6, sanitized.big_huber_delta);
  sanitized.big_omega_a = std::max(0.0, sanitized.big_omega_a);
  sanitized.big_omega_w = std::max(0.0, sanitized.big_omega_w);

  std::lock_guard<std::mutex> lock(config_mutex_);
  const bool changed =
    std::abs(config_.max_match_angle - sanitized.max_match_angle) > 1e-12 ||
    std::abs(config_.min_residual_margin - sanitized.min_residual_margin) > 1e-12 ||
    std::abs(config_.max_id_angular_speed - sanitized.max_id_angular_speed) > 1e-12 ||
    std::abs(config_.max_zero_phase_innovation_rad - sanitized.max_zero_phase_innovation_rad) >
      1e-12 ||
    std::abs(config_.id_coast_timeout_sec - sanitized.id_coast_timeout_sec) > 1e-12 ||
    std::abs(config_.id_reset_timeout_sec - sanitized.id_reset_timeout_sec) > 1e-12 ||
    config_.model_min_confirmed_frames != sanitized.model_min_confirmed_frames ||
    std::abs(config_.model_max_nis - sanitized.model_max_nis) > 1e-12 ||
    std::abs(config_.big_direction_lock_threshold - sanitized.big_direction_lock_threshold) >
      1e-12 ||
    std::abs(config_.small_direction_lock_threshold - sanitized.small_direction_lock_threshold) >
      1e-12 ||
    std::abs(config_.big_huber_delta - sanitized.big_huber_delta) > 1e-12 ||
    std::abs(config_.big_omega_a - sanitized.big_omega_a) > 1e-12 ||
    std::abs(config_.big_omega_w - sanitized.big_omega_w) > 1e-12 ||
    std::abs(config_.big_a_center - sanitized.big_a_center) > 1e-12 ||
    std::abs(config_.big_w_center - sanitized.big_w_center) > 1e-12 ||
    std::abs(config_.big_center_prior_weight - sanitized.big_center_prior_weight) > 1e-12 ||
    std::abs(config_.small_ekf_q_accel - sanitized.small_ekf_q_accel) > 1e-12 ||
    std::abs(config_.small_ekf_r_angle - sanitized.small_ekf_r_angle) > 1e-12 ||
    std::abs(config_.small_ekf_p0_angle - sanitized.small_ekf_p0_angle) > 1e-12 ||
    std::abs(config_.small_ekf_p0_speed - sanitized.small_ekf_p0_speed) > 1e-12 ||
    std::abs(config_.small_huber_delta - sanitized.small_huber_delta) > 1e-12;
  if (!changed) return false;

  config_.max_match_angle = sanitized.max_match_angle;
  config_.min_residual_margin = sanitized.min_residual_margin;
  config_.max_id_angular_speed = sanitized.max_id_angular_speed;
  config_.max_zero_phase_innovation_rad = sanitized.max_zero_phase_innovation_rad;
  config_.id_coast_timeout_sec = sanitized.id_coast_timeout_sec;
  config_.id_reset_timeout_sec = sanitized.id_reset_timeout_sec;
  config_.model_min_confirmed_frames = sanitized.model_min_confirmed_frames;
  config_.model_max_nis = sanitized.model_max_nis;
  config_.big_direction_lock_threshold = sanitized.big_direction_lock_threshold;
  config_.small_direction_lock_threshold = sanitized.small_direction_lock_threshold;
  config_.big_huber_delta = sanitized.big_huber_delta;
  config_.big_omega_a = sanitized.big_omega_a;
  config_.big_omega_w = sanitized.big_omega_w;
  config_.big_a_center = sanitized.big_a_center;
  config_.big_w_center = sanitized.big_w_center;
  config_.big_center_prior_weight = sanitized.big_center_prior_weight;
  config_.small_ekf_q_accel = sanitized.small_ekf_q_accel;
  config_.small_ekf_r_angle = sanitized.small_ekf_r_angle;
  config_.small_ekf_p0_angle = sanitized.small_ekf_p0_angle;
  config_.small_ekf_p0_speed = sanitized.small_ekf_p0_speed;
  config_.small_huber_delta = sanitized.small_huber_delta;
  return true;
}

bool AngleEstimate::useTargetEkf() const
{
  return config_.small_predictor == SmallRunePredictor::TARGET_EKF;
}

bool AngleEstimate::usesAsyncCeres() const
{
  return !(energy_tri_.mode == BuffMode::SMALL && useTargetEkf());
}

PredictorBackend AngleEstimate::currentPredictorBackend() const
{
  if (energy_tri_.mode == BuffMode::BIG) return PredictorBackend::BIG_CERES;
  if (energy_tri_.mode == BuffMode::SMALL && useTargetEkf()) {
    return PredictorBackend::SMALL_TARGET_EKF;
  }
  return PredictorBackend::SMALL_CERES;
}
}  // namespace auto_buff
