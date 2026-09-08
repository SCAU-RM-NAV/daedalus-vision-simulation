#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, const EkfNoiseConfig & ekf_noise_config)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0),
  ekf_noise_config_(ekf_noise_config),
  current_v1_(ekf_noise_config.normal_v1),
  current_v2_(ekf_noise_config.normal_v2)
{
  auto r = radius;
  priority = armor.priority;
  last_armor_pnp_xyz_in_camera_ = armor.pnp_xyz_in_camera;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  apply_z_observation(armor, 0, true);
  update_ekf_debug_data(current_v1_, current_v2_, 0.0, 0.0);
}

Target::Target(double x, double vyaw, double radius, double h)
: name(ArmorName::not_armor),
  armor_type(ArmorType::small),
  priority(ArmorPriority::fifth),
  jumped(true),
  last_id(0),
  armor_num_(4),
  switch_count_(0),
  update_count_(0),
  is_switch_(false),
  is_converged_(true)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
  ekf_.data["observed_z"] = last_observed_z_;
  ekf_.data["observed_base_z"] = observed_base_z_;
  ekf_.data["z_observation_only"] = 0.0;
  update_ekf_debug_data(current_v1_, current_v2_, 0.0, 0.0);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  const bool hold_observed_z = z_observation_only() && std::isfinite(observed_base_z_);
  if (hold_observed_z) {
    // Keep the latest observed height while X/Y/yaw continue to use the normal prediction model.
    F(4, 5) = 0.0;
  }

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = ekf_noise_config_.outpost_v1;
    v2 = ekf_noise_config_.outpost_v2;
  } else {
    v1 = current_v1_;
    v2 = current_v2_;
  }
  update_ekf_debug_data(v1, v2, 0.0, 0.0);

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 防止夹角求和出现异常值
  if (hold_observed_z) {
    // Adaptive translational noise must not turn this channel back into a Z predictor.
    Q.block<2, 2>(4, 4).setZero();
  }

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    if (hold_observed_z) {
      x_prior[4] = observed_base_z_;
      x_prior[5] = 0.0;
    }
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
  decay_adaptive_process_noise();
}

void Target::update(const Armor & armor, bool record_z_observation)
{
  last_armor_pnp_xyz_in_camera_ = armor.pnp_xyz_in_camera;

  // 装甲板匹配
  int id = 0;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  int candidates_count = std::min<int>(3, static_cast<int>(xyza_i_list.size()));
  for (int i = 0; i < candidates_count; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
    if (is_outpost()) {
      angle_error += std::abs(tools::limit_rad(armor.ypd_in_world[1] - ypd[1]));
    } else {
      angle_error += std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    }

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  update_count_++;

  if (is_outpost()) init_outpost_height(armor, id);

  update_ypda(armor, id, record_z_observation);
  update_adaptive_process_noise();
}

void Target::update_ypda(const Armor & armor, int id, bool record_z_observation)
{
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  // Eigen::VectorXd R_dig{{4e-3, 4e-3, 1, 9e-2}};
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
    {4e-3, 4e-3, log(std::abs(delta_angle) + 1) + 1,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};

  //测量过程噪声偏差的方差
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 定义非线性转换函数h: x -> z
  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 防止夹角求差出现异常值
  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};  //获得观测量

  ekf_.update(z, H, R, h, z_subtract);

  apply_z_observation(armor, id, record_z_observation);

  if (is_outpost()) ekf_.x[8] = 0.2765;
}

void Target::apply_z_observation(const Armor & armor, int id, bool record_z_observation)
{
  last_observed_z_ = armor.xyz_in_world.z();
  if (record_z_observation) update_z_observation_only_state(last_observed_z_);
  ekf_.data["observed_z"] = last_observed_z_;
  ekf_.data["z_observation_only"] = z_observation_only() ? 1.0 : 0.0;

  if (!z_observation_only() || !std::isfinite(last_observed_z_)) {
    ekf_.data["observed_base_z"] = observed_base_z_;
    return;
  }

  // Preserve modeled height differences between plates, but anchor their common/base height to the
  // latest raw world-Z observation. The same rule covers outpost's per-plate height offsets.
  const double armor_z_offset = h_armor_xyz(ekf_.x, id).z() - ekf_.x[4];
  observed_base_z_ = last_observed_z_ - armor_z_offset;
  ekf_.x[4] = observed_base_z_;
  ekf_.x[5] = 0.0;
  ekf_.data["observed_base_z"] = observed_base_z_;
}

void Target::update_z_observation_only_state(double observed_z)
{
  if (is_outpost()) {
    z_observation_only_active_ = false;
    return;
  }

  switch (ekf_noise_config_.z_observation_only_mode) {
    case ZObservationOnlyMode::disabled:
      z_observation_only_active_ = false;
      return;
    case ZObservationOnlyMode::enabled:
      z_observation_only_active_ = true;
      return;
    case ZObservationOnlyMode::automatic:
      break;
  }

  if (!std::isfinite(observed_z)) {
    z_observation_history_.clear();
    z_observation_only_active_ = false;
    return;
  }

  const auto window_size = static_cast<std::size_t>(
    std::max(1, ekf_noise_config_.z_observation_auto_window_frames));
  z_observation_history_.push_back(observed_z);
  while (z_observation_history_.size() > window_size) {
    z_observation_history_.pop_front();
  }

  const auto [min_z, max_z] =
    std::minmax_element(z_observation_history_.begin(), z_observation_history_.end());
  const double range = *max_z - *min_z;
  z_observation_only_active_ =
    range > std::max(0.0, ekf_noise_config_.z_observation_auto_range_threshold);

  if (z_observation_only_active_) {
    tools::logger()->info(
      "[Target] Z observation-only auto triggered: range={:.3f} m, min_z={:.3f} m, "
      "max_z={:.3f} m, samples={}",
      range, *min_z, *max_z, z_observation_history_.size());
  }
}

void Target::update_adaptive_process_noise()
{
  if (name == ArmorName::outpost) {
    update_ekf_debug_data(ekf_noise_config_.outpost_v1, ekf_noise_config_.outpost_v2, 0.0, 0.0);
    return;
  }

  bool v1_boost = false;
  bool v2_boost = false;

  if (ekf_noise_config_.adaptive_v1) {
    auto residual_yaw = std::abs(ekf_.data.at("residual_yaw"));
    auto residual_pitch = std::abs(ekf_.data.at("residual_pitch"));
    auto residual_distance = std::abs(ekf_.data.at("residual_distance"));
    auto yaw_mismatch = residual_yaw > ekf_noise_config_.adaptive_v1_residual_yaw;
    auto pitch_mismatch = residual_pitch > ekf_noise_config_.adaptive_v1_residual_pitch;
    auto distance_mismatch = residual_distance > ekf_noise_config_.adaptive_v1_residual_distance;

  auto v1_mismatch = yaw_mismatch || pitch_mismatch || distance_mismatch;

  if (v1_mismatch) {
    adaptive_v1_hit_count_ = std::min(adaptive_v1_hit_count_ + 1, 100);
  } else {
    adaptive_v1_hit_count_ = 0;
  }

  v1_boost = adaptive_v1_hit_count_ >= 3;
  
    if (v1_boost) {
      if (!adaptive_v1_boosting_) {
        tools::logger()->info(
          "[Target] Adaptive v1 boost {:.0f}->{:.0f}, residual_yaw={:.4f}, "
          "residual_pitch={:.4f}, residual_distance={:.3f}",
          current_v1_, ekf_noise_config_.adaptive_v1_high, residual_yaw, residual_pitch,
          residual_distance);
      }
      adaptive_v1_boosting_ = true;
      current_v1_ = std::max(current_v1_, ekf_noise_config_.adaptive_v1_high);
    } else {
      adaptive_v1_boosting_ = false;
    }
  } else {
    adaptive_v1_boosting_ = false;
    current_v1_ = ekf_noise_config_.normal_v1;
  }

  if (ekf_noise_config_.adaptive_v2) {
    auto residual_angle = std::abs(ekf_.data.at("residual_angle"));
    auto nis = ekf_.data.at("nis");
    auto angle_mismatch = residual_angle > ekf_noise_config_.adaptive_v2_residual_angle;
    auto nis_mismatch = nis > ekf_noise_config_.adaptive_v2_nis;

    v2_boost = angle_mismatch || nis_mismatch;
    if (v2_boost) {
      if (!adaptive_v2_boosting_) {
        tools::logger()->info(
          "[Target] Adaptive v2 boost {:.0f}->{:.0f}, residual_angle={:.3f}, nis={:.3f}",
          current_v2_, ekf_noise_config_.adaptive_v2_high, residual_angle, nis);
      }
      adaptive_v2_boosting_ = true;
      current_v2_ = std::max(current_v2_, ekf_noise_config_.adaptive_v2_high);
    } else {
      adaptive_v2_boosting_ = false;
    }
  } else {
    adaptive_v2_boosting_ = false;
    current_v2_ = ekf_noise_config_.normal_v2;
  }

  update_ekf_debug_data(
    current_v1_, current_v2_, v1_boost ? 1.0 : 0.0, v2_boost ? 1.0 : 0.0);
}

void Target::decay_adaptive_process_noise()
{
  if (name == ArmorName::outpost) return;

  if (ekf_noise_config_.adaptive_v1) {
    auto decay = std::clamp(ekf_noise_config_.adaptive_v1_decay, 0.0, 1.0);
    current_v1_ = std::max(
      ekf_noise_config_.normal_v1, current_v1_ * decay + ekf_noise_config_.normal_v1 * (1 - decay));
  }

  if (ekf_noise_config_.adaptive_v2) {
    auto decay = std::clamp(ekf_noise_config_.adaptive_v2_decay, 0.0, 1.0);
    current_v2_ = std::max(
      ekf_noise_config_.normal_v2, current_v2_ * decay + ekf_noise_config_.normal_v2 * (1 - decay));
  }
}

void Target::update_ekf_debug_data(double v1, double v2, double v1_boost, double v2_boost)
{
  ekf_.data["process_v1"] = v1;
  ekf_.data["process_v2"] = v2;
  ekf_.data["adaptive_v1_boost"] = v1_boost;
  ekf_.data["adaptive_v2_boost"] = v2_boost;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

const TrackingInfo & Target::tracking_info() const { return tracking_info_; }

void Target::set_tracking_info(const TrackingInfo & tracking_info)
{
  tracking_info_ = tracking_info;
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

const Eigen::Vector3d & Target::last_armor_pnp_xyz_in_camera() const
{
  return last_armor_pnp_xyz_in_camera_;
}

double Target::last_observed_z() const { return last_observed_z_; }

bool Target::z_observation_only() const
{
  // Outpost keeps the original Z/vz EKF and impact-time prediction even when the robot config
  // enables observation-only Z for normal targets.
  return z_observation_only_active_ && !is_outpost();
}

void Target::reset_z_observation_history()
{
  if (ekf_noise_config_.z_observation_only_mode != ZObservationOnlyMode::automatic) return;

  z_observation_history_.clear();
  z_observation_only_active_ = false;
  last_observed_z_ = 0.0;
  observed_base_z_ = std::numeric_limits<double>::quiet_NaN();
  ekf_.data["observed_z"] = last_observed_z_;
  ekf_.data["observed_base_z"] = observed_base_z_;
  ekf_.data["z_observation_only"] = 0.0;
}

bool Target::diverged() const
{
  if (is_outpost()) {
    auto r_ok = ekf_.x[8] > 0.12 && ekf_.x[8] < 0.4;
    if (r_ok) return false;

    tools::logger()->debug("[Target] outpost r={:.3f}", ekf_.x[8]);
    return true;
  }

  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

bool Target::is_outpost() const { return name == ArmorName::outpost && armor_num_ == 3; }

void Target::init_outpost_height(const Armor & armor, int id)
{
  if (id < 0 || id >= static_cast<int>(outpost_height_initialized_.size())) return;
  if (outpost_height_initialized_[id]) return;

  if (id == 0) {
    ekf_.x[4] = armor.xyz_in_world[2];
    ekf_.x[5] = 0.0;
  } else if (id == 1) {
    ekf_.x[9] = armor.xyz_in_world[2] - ekf_.x[4];
  } else {
    ekf_.x[10] = armor.xyz_in_world[2] - ekf_.x[4];
  }

  outpost_height_initialized_[id] = true;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];
  if (is_outpost()) {
    if (id == 1) armor_z = x[4] + x[9];
    if (id == 2) armor_z = x[4] + x[10];
  }

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh1 = 0.0;
  auto dz_dh2 = (use_l_h) ? 1.0 : 0.0;
  if (is_outpost()) {
    dz_dh1 = (id == 1) ? 1.0 : 0.0;
    dz_dh2 = (id == 2) ? 1.0 : 0.0;
  }

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,      0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,      0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0, dz_dh1, dz_dh2},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,      0,      0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
