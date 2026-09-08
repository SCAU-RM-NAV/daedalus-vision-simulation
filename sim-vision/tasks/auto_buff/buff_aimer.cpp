#include "buff_aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

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

BuffMode read_buff_mode(const YAML::Node & yaml)
{
  std::string mode = "big";
  if (yaml["predictor"] && yaml["predictor"]["mode"]) {
    mode = yaml["predictor"]["mode"].as<std::string>();
  } else if (yaml["buff_mode"]) {
    mode = yaml["buff_mode"].as<std::string>();
  }
  return lower(mode) == "small" ? BuffMode::SMALL : BuffMode::BIG;
}

double read_tuned_bias_ms(const YAML::Node & yaml, const std::string & mode_key, double fallback)
{
  const auto ballistic = yaml["ballistic"];
  if (ballistic && ballistic[mode_key]) return ballistic[mode_key].as<double>();
  if (yaml[mode_key]) return yaml[mode_key].as<double>();
  return fallback;
}

template <typename T>
void read_target_selection_param(
  const YAML::Node & yaml, const std::string & key, const std::string & root_key, T & value)
{
  const auto selection = yaml["target_selection"];
  if (selection && selection[key]) {
    value = selection[key].as<T>();
    return;
  }
  if (yaml[root_key]) value = yaml[root_key].as<T>();
}

template <typename T>
void read_command_guard_param(
  const YAML::Node & yaml, const std::string & key, const std::string & root_key, T & value)
{
  const auto guard = yaml["command_guard"];
  if (guard && guard[key]) {
    value = guard[key].as<T>();
    return;
  }
  if (yaml[root_key]) value = yaml[root_key].as<T>();
}

std::vector<int> collect_attack_candidates(const PowerRune & rune)
{
  std::vector<int> candidates;
  candidates.reserve(rune.fanblades.size());
  for (const auto & blade : rune.fanblades) {
    if (blade.type != _unlight && blade.leaf_id >= 0) candidates.push_back(blade.leaf_id);
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
  return candidates;
}

AngleEstimate::IdMatchResult update_estimator(
  AngleEstimate & estimator, const PowerRune & rune, double observed_time_abs, BuffMode mode)
{
  auto match = estimator.update(rune.physical_angle, observed_time_abs, mode);
  estimator.angleFit();
  return match;
}

io::Command make_prefit_r_center_command(
  const BuffBallisticResult & result, BuffBallisticResult & last_result)
{
  last_result = result;
  io::Command command = {false, false, 0, 0};
  if (!result.valid) return command;
  command.control = true;
  command.shoot = false;
  command.yaw = result.yaw;
  command.pitch = -result.pitch;
  return command;
}

auto_aim::Plan make_prefit_r_center_plan(
  const BuffBallisticResult & result, BuffBallisticResult & last_result)
{
  last_result = result;
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  if (!result.valid) return plan;
  plan.control = true;
  plan.fire = false;
  plan.yaw = result.yaw;
  plan.pitch = -result.pitch;
  return plan;
}

constexpr double kLeafSpacing = 2.0 * M_PI / 5.0;
constexpr double kDefaultLegacyBiasTimeMs = 0.0; // 默认时间偏置
constexpr double kBigSingleLeafFireWindowSec = 1.0;

}  // namespace

Aimer::Aimer(const std::string & config_path)
: angle_estimator_(AngleEstimateConfig::from_yaml(config_path)),
  pose_filter_(
    PoseFilterConfig::from_yaml(config_path),
    BallisticParams::from_yaml(config_path).target_radius),
  ballistic_(BallisticParams::from_yaml(config_path))
{
  auto yaml = YAML::LoadFile(config_path);
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  double legacy_bias_time_ms = kDefaultLegacyBiasTimeMs;
  if (yaml["ballistic"] && yaml["ballistic"]["bias_time_ms"]) {
    legacy_bias_time_ms = yaml["ballistic"]["bias_time_ms"].as<double>();
  } else if (yaml["bias_time_ms"]) {
    legacy_bias_time_ms = yaml["bias_time_ms"].as<double>();
  }
  small_bias_time_ms_ = read_tuned_bias_ms(yaml, "small_bias_time_ms", legacy_bias_time_ms);
  big_bias_time_ms_ = read_tuned_bias_ms(yaml, "big_bias_time_ms", legacy_bias_time_ms);
  prediction_mode_ = read_buff_mode(yaml);
  read_target_selection_param(yaml, "w_yaw", "sel_w_yaw", sel_w_yaw_);
  read_target_selection_param(yaml, "w_pitch", "sel_w_pitch", sel_w_pitch_);
  read_target_selection_param(yaml, "switch_margin", "sel_switch_margin", sel_switch_margin_);
  read_target_selection_param(yaml, "switch_frames", "sel_switch_frames", sel_switch_frames_);
  read_command_guard_param(yaml, "hold_last_leaf_sec", "hold_last_leaf_sec", hold_last_leaf_sec_);
  read_command_guard_param(
    yaml, "fire_rearm_confirmed_frames", "fire_rearm_confirmed_frames",
    fire_rearm_confirmed_frames_);
  read_command_guard_param(yaml, "max_yaw_rate", "max_yaw_rate", max_yaw_rate_);
  read_command_guard_param(yaml, "max_pitch_rate", "max_pitch_rate", max_pitch_rate_);
  read_command_guard_param(yaml, "max_slew_dt_sec", "max_slew_dt_sec", max_slew_dt_sec_);
  read_command_guard_param(yaml, "max_fire_yaw_error", "max_fire_yaw_error", max_fire_yaw_error_);
  read_command_guard_param(yaml, "max_fire_pitch_error", "max_fire_pitch_error", max_fire_pitch_error_);
  read_command_guard_param(
    yaml, "big_single_leaf_fire_guard_enabled", "big_single_leaf_fire_guard_enabled",
    big_single_leaf_fire_guard_enabled_);

  // 法向量滑动窗口滤波配置
  {
    const auto nf = yaml["normal_filter"];
    if (nf && nf["enabled"]) normal_avg_enabled_ = nf["enabled"].as<bool>();
    if (nf && nf["window_size"]) normal_avg_window_size_ = nf["window_size"].as<int>();
    if (nf && nf["change_threshold"])
      normal_avg_change_threshold_ = nf["change_threshold"].as<double>();
    if (yaml["normal_avg_enabled"]) normal_avg_enabled_ = yaml["normal_avg_enabled"].as<bool>();
    if (yaml["normal_avg_window_size"]) normal_avg_window_size_ = yaml["normal_avg_window_size"].as<int>();
  }
  normal_avg_window_size_ = std::max(5, normal_avg_window_size_);

  sel_switch_frames_ = std::max(1, sel_switch_frames_);
  hold_last_leaf_sec_ = std::max(0.0, hold_last_leaf_sec_);
  fire_rearm_confirmed_frames_ = std::max(1, fire_rearm_confirmed_frames_);
  max_yaw_rate_ = std::max(0.1, max_yaw_rate_);
  max_pitch_rate_ = std::max(0.1, max_pitch_rate_);
  max_slew_dt_sec_ = std::max(1e-3, max_slew_dt_sec_);

  last_fire_t_ = std::chrono::steady_clock::now();
}

io::Command Aimer::aim(
  PowerRune & rune, double observed_time_abs, double now_time_abs, double bullet_speed,
  BuffMode mode)
{
  const auto solve_start = std::chrono::steady_clock::now();
  last_ballistic_result_ = {};
  last_command_guard_debug_ = {};
  last_attack_selection_debug_ = {};
  io::Command command = {false, false, 0, 0};
  if (bullet_speed < 10.0) bullet_speed = 24.0;
  bullet_speed = filter_bullet_speed(bullet_speed);
  const BuffMode used_mode = mode == BuffMode::LOST ? prediction_mode_ : mode;
  if (!rune.pnp_valid) {
    annotate_observation(rune, std::nullopt);
    reset_fire_rearm();
    if (tracking_ready()) return hold_last_leaf_command(observed_time_abs);
    if (!can_aim_cached_r_center(rune, observed_time_abs)) return command;

    const double runtime_solve_delay_ms = resolve_runtime_solve_delay_ms(solve_start);
    const BuffBallisticTiming timing{
      observed_time_abs, now_time_abs, runtime_solve_delay_ms, tuned_bias_time_ms(used_mode)};
    const auto result = solve_r_center(rune, timing, bullet_speed);
    return make_prefit_r_center_command(result, last_ballistic_result_);
  }
  const auto match = update_observation(rune, observed_time_abs, used_mode);
  if (!match.confirmed()) {
    reset_fire_rearm();
    if (tracking_ready()) return hold_last_leaf_command(observed_time_abs);
    return command;
  }
  note_confirmed_observation();
  const double runtime_solve_delay_ms = resolve_runtime_solve_delay_ms(solve_start);
  const BuffBallisticTiming timing{
    observed_time_abs, now_time_abs, runtime_solve_delay_ms, tuned_bias_time_ms(used_mode)};

  if (!angle_estimator_.has_successful_fit() || angle_estimator_.is_lost()) {
    const auto result = solve_r_center(rune, timing, bullet_speed);
    return make_prefit_r_center_command(result, last_ballistic_result_);
  }

  const auto & pose = pose_filter_.snapshot();
  const Eigen::Vector3d & rune_center_world =
    pose.valid ? pose.filtered_xyz_world : rune.xyz_in_world;
  const Eigen::Matrix3d & rotation_world =
    pose.valid ? pose.filtered_rotation_world : rune.rotation_world;
  const double base_angle = angle_estimator_.predict(timing.observed_time_abs);
  // 从平均法向量提取 yaw/pitch（对绕 X 轴旋转不变），消除 PnP 共面退化导致的朝向抖动
  const double angle_now = std::isfinite(base_angle) ? base_angle : rune.physical_angle;
  const Eigen::Vector3d n = get_averaged_normal(rotation_world);
  const Eigen::Matrix3d rotation_world_aim =
    tools::rotation_matrix(Eigen::Vector3d(
      std::atan2(n.y(), n.x()),
      -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y())),
      angle_now + CV_PI / 2.0));
  auto result = ballistic_.solve_with_iteration(
    rune_center_world, rotation_world_aim, angle_now,
    timing, bullet_speed,
    [&](double predict_time_abs) { return angle_estimator_.predict(predict_time_abs); });
  result.aim_source = BuffAimSource::OBSERVED_LEAF;
  result.used_filtered_pose = pose.valid;
  result.observed_leaf_id = angle_estimator_.current_leaf_id();
  result.attack_leaf_id = angle_estimator_.current_leaf_id();
  last_ballistic_result_ = result;

  // ── debug: 追踪 blade 形状预测的关键变量，帧间变化过大时输出 ──
  if (result.valid) {
    const Eigen::Vector3d n = get_averaged_normal(rotation_world);
    const double normal_yaw = std::atan2(n.y(), n.x());
    const double normal_pitch = -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y()));
    const double r_center_dist = rune_center_world.norm();
    const double pred_angle =
      angle_estimator_.has_successful_fit()
        ? angle_estimator_.predict(timing.observed_time_abs, result.attack_leaf_id)
        : std::numeric_limits<double>::quiet_NaN();

    struct PrevStateCmd {
      double normal_yaw = 0, normal_pitch = 0, r_dist = 0, pred_angle = 0;
      double out_yaw = 0, out_pitch = 0;
      int attack_leaf = -1;
      bool init = false;
    };
    static PrevStateCmd prev_cmd;
    if (prev_cmd.init) {
      const double dyaw_n = std::abs(tools::limit_rad(normal_yaw - prev_cmd.normal_yaw));
      const double dpitch_n = std::abs(tools::limit_rad(normal_pitch - prev_cmd.normal_pitch));
      const double ddist = std::abs(r_center_dist - prev_cmd.r_dist);
      const double dangle = std::abs(tools::limit_rad(pred_angle - prev_cmd.pred_angle));
      const double dyaw_out = std::abs(tools::limit_rad(result.yaw - prev_cmd.out_yaw));
      const double dpitch_out = std::abs(result.pitch - prev_cmd.out_pitch);
      const bool leaf_changed = result.attack_leaf_id != prev_cmd.attack_leaf;

      constexpr double kDYawThresh = 0.01;
      constexpr double kDPitchThresh = 0.01;
      constexpr double kDDistThresh = 0.10;
      constexpr double kDAngleThresh = 0.03;
      constexpr double kDOutThresh = 0.01;

      if (dyaw_n > kDYawThresh || dpitch_n > kDPitchThresh || ddist > kDDistThresh ||
          dangle > kDAngleThresh || leaf_changed || dyaw_out > kDOutThresh ||
          dpitch_out > kDOutThresh) {
        tools::logger()->debug(
          "[blade_shape_cmd] Δnormal_yaw={:.4f} Δnormal_pitch={:.4f} ΔR_dist={:.3f}m "
          "Δpred_angle={:.4f}rad leaf={}->{} Δout_yaw={:.4f} Δout_pitch={:.4f}",
          dyaw_n, dpitch_n, ddist, dangle,
          prev_cmd.attack_leaf, result.attack_leaf_id, dyaw_out, dpitch_out);
      }
    }
    prev_cmd = {normal_yaw, normal_pitch, r_center_dist, pred_angle,
                result.yaw, result.pitch, result.attack_leaf_id, true};
  }

  if (!result.valid) return command;

  command.control = true;
  command.yaw = result.yaw;
  command.pitch = -result.pitch;  // 世界坐标系下 pitch 向上为负
  limit_and_record_command(command, observed_time_abs);
  command.shoot = !suppress_fire_ &&
                  confirmed_frames_since_guard_ >= fire_rearm_confirmed_frames_ && should_fire();

  return command;
}

// 主函数入口
auto_aim::Plan Aimer::mpc_aim(
  PowerRune & rune, double observed_time_abs, double now_time_abs, io::GimbalState gs,
  BuffMode mode)
{
  const auto solve_start = std::chrono::steady_clock::now();
  last_ballistic_result_ = {};
  last_command_guard_debug_ = {};
  last_attack_selection_debug_ = {};
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  tools::logger()->info("gs speed:{}",gs.bullet_speed);
  const double raw_bullet_speed = gs.bullet_speed < 10.0 ? 23.5 : gs.bullet_speed;
  const double bullet_speed = filter_bullet_speed(raw_bullet_speed);
  const BuffMode used_mode = mode == BuffMode::LOST ? prediction_mode_ : mode;
  update_big_single_leaf_fire_guard(rune, used_mode);
  if (!rune.pnp_valid) {
    tools::logger()->info("pnp unvalid!!!");
    annotate_observation(rune, std::nullopt); // 标记观测无效
    reset_fire_rearm();
    // 如果还在跟踪，保持上一帧目标
    if (tracking_ready()) return hold_last_leaf_plan(observed_time_abs);
    // 如果没有缓存，看能不能瞄R标
    if (!can_aim_cached_r_center(rune, observed_time_abs)) return plan;

    const double runtime_solve_delay_ms = resolve_runtime_solve_delay_ms(solve_start);
    const BuffBallisticTiming timing{
      observed_time_abs, now_time_abs, runtime_solve_delay_ms, tuned_bias_time_ms(used_mode)};
    const auto result = solve_r_center(rune, timing, bullet_speed);
    plan = make_prefit_r_center_plan(result, last_ballistic_result_);
    return plan;
  }

  // 更新观测+id匹配
  const auto match = update_observation(rune, observed_time_abs, used_mode);
  if (!match.confirmed()) {
    tools::logger()->info("id no match!!!");
    reset_fire_rearm();
    if (tracking_ready()) return hold_last_leaf_plan(observed_time_abs);
    return plan;
  }
  note_confirmed_observation();
  const double runtime_solve_delay_ms = resolve_runtime_solve_delay_ms(solve_start);
  const BuffBallisticTiming timing{
    observed_time_abs, now_time_abs, runtime_solve_delay_ms, tuned_bias_time_ms(used_mode)};

    // 还没拟合好，先瞄R标
  if (!angle_estimator_.has_successful_fit() || angle_estimator_.is_lost()) {
    const auto result = solve_r_center(rune, timing, bullet_speed);
    plan = make_prefit_r_center_plan(result, last_ballistic_result_);
    return plan;
  }

  // 对世界位姿进行EKF
  const auto & pose = pose_filter_.snapshot();
  // R标在世界坐标系的位置
  const Eigen::Vector3d & rune_center_world =
    pose.valid ? pose.filtered_xyz_world : rune.xyz_in_world;
  // buff坐标系到世界坐标系的旋转
    const Eigen::Matrix3d & rotation_world =
    pose.valid ? pose.filtered_rotation_world : rune.rotation_world;
  BuffBallisticResult result;
  // 大小符两个分支，大符除了弹道解算外，多加了击打扇叶选取的逻辑
  if (used_mode == BuffMode::BIG) {
    select_attack_leaf(rune, timing, rune_center_world, rotation_world, bullet_speed, gs, result);
  } else {
    const double base_angle_small = angle_estimator_.predict(timing.observed_time_abs);
    // const double base_angle_small = rune.physical_angle;
    const double angle_now_small =
      std::isfinite(base_angle_small) ? base_angle_small : rune.physical_angle;
    const Eigen::Vector3d n = get_averaged_normal(rotation_world);
    const Eigen::Matrix3d rotation_world_small =
      tools::rotation_matrix(Eigen::Vector3d(
        std::atan2(n.y(), n.x()),
        -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y())),
        angle_now_small + CV_PI / 2.0));
    result = ballistic_.solve_with_iteration(
      rune_center_world, rotation_world_small, angle_now_small,
      timing, bullet_speed,
      [&](double predict_time_abs) { return angle_estimator_.predict(predict_time_abs); });
    const int observed_leaf_id = angle_estimator_.current_leaf_id();
    result.aim_source = BuffAimSource::OBSERVED_LEAF;
    result.attack_leaf_id = observed_leaf_id;
    result.observed_leaf_id = observed_leaf_id;
    attack_leaf_id_ = observed_leaf_id;
    pending_attack_leaf_id_ = -1;
    switch_streak_ = 0;
  }
  last_ballistic_result_ = result;
  last_ballistic_result_.used_filtered_pose = pose.valid;

  // ── debug: 追踪 blade 形状预测的关键变量，帧间变化过大时输出 ──
  if (result.valid) {
    const Eigen::Vector3d n = get_averaged_normal(rotation_world);
    const double normal_yaw = std::atan2(n.y(), n.x());
    const double normal_pitch = -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y()));
    const double r_center_dist = rune_center_world.norm();
    const double pred_angle =
      angle_estimator_.has_successful_fit()
        ? angle_estimator_.predict(timing.observed_time_abs, result.attack_leaf_id)
        : std::numeric_limits<double>::quiet_NaN();

    struct PrevState {
      double normal_yaw = 0, normal_pitch = 0, r_dist = 0, pred_angle = 0;
      double out_yaw = 0, out_pitch = 0;
      int attack_leaf = -1;
      bool init = false;
    };
    static PrevState prev;
    if (prev.init) {
      const double dyaw_n = std::abs(tools::limit_rad(normal_yaw - prev.normal_yaw));
      const double dpitch_n = std::abs(tools::limit_rad(normal_pitch - prev.normal_pitch));
      const double ddist = std::abs(r_center_dist - prev.r_dist);
      const double dangle = std::abs(tools::limit_rad(pred_angle - prev.pred_angle));
      const double dyaw_out = std::abs(tools::limit_rad(result.yaw - prev.out_yaw));
      const double dpitch_out = std::abs(result.pitch - prev.out_pitch);
      const bool leaf_changed = result.attack_leaf_id != prev.attack_leaf;

      constexpr double kDYawThresh = 0.01;     // rad
      constexpr double kDPitchThresh = 0.01;
      constexpr double kDDistThresh = 0.10;     // m
      constexpr double kDAngleThresh = 0.03;    // rad
      constexpr double kDOutThresh = 0.01;      // rad

      if (dyaw_n > kDYawThresh || dpitch_n > kDPitchThresh || ddist > kDDistThresh ||
          dangle > kDAngleThresh || leaf_changed || dyaw_out > kDOutThresh ||
          dpitch_out > kDOutThresh) {
        tools::logger()->debug(
          "[blade_shape] Δnormal_yaw={:.4f} Δnormal_pitch={:.4f} ΔR_dist={:.3f}m "
          "Δpred_angle={:.4f}rad leaf={}->{} Δout_yaw={:.4f} Δout_pitch={:.4f}",
          dyaw_n, dpitch_n, ddist, dangle,
          prev.attack_leaf, result.attack_leaf_id, dyaw_out, dpitch_out);
      }
    }
    prev = {normal_yaw, normal_pitch, r_center_dist, pred_angle,
            result.yaw, result.pitch, result.attack_leaf_id, true};
  }

  // 弹道解算无效，重置开火恢复计数，保持上一帧目标（不让视觉控开火并不影响，后续可作为自动开火的逻辑升级）
  if (!result.valid) {
    reset_fire_rearm();
    return hold_last_leaf_plan(observed_time_abs);
  }

  plan.control = true;
  plan.yaw = result.yaw;
  plan.pitch = -result.pitch;

// 计算平滑的解析前馈速度
  double analytical_yaw_vel = 0.0;
  double analytical_pitch_vel = 0.0;
  calculate_analytical_velocities(
    timing, rune_center_world, rotation_world, bullet_speed, result.attack_leaf_id,
    analytical_yaw_vel, analytical_pitch_vel);
  
  limit_and_record_plan(plan, observed_time_abs, gs, analytical_yaw_vel, analytical_pitch_vel); // 角速度限幅

  
  const double fire_yaw_err = tools::limit_rad(result.yaw - static_cast<double>(gs.yaw));
  const double fire_pitch_err = static_cast<double>(plan.pitch) - static_cast<double>(gs.pitch);
  plan.fire = should_fire(result.attack_leaf_id, fire_yaw_err, fire_pitch_err, result.fly_time);

  return plan;
}

void Aimer::notify_observation_missed()
{
  last_ballistic_result_ = {};
  reset_fire_rearm();
  normal_history_.clear();
  normal_sum_ = Eigen::Vector3d::Zero();
  normal_sum_sq_ = Eigen::Vector3d::Zero();
}

void Aimer::reset()
{
  notify_observation_missed();
  angle_estimator_.reset();
  pose_filter_.reset();
  reset_attack_selection();
  selection_epoch_ = 0;
  has_last_plan_sample_ = false;
  last_plan_observed_time_abs_ = 0.0;
  last_yaw_ = 0.0;
  last_pitch_ = 0.0;
  last_raw_yaw_ = 0.0;
  last_yaw_delta_ = 0.0;
  last_yaw_vel_ = 0.0;
  last_pitch_delta_ = 0.0;
  big_single_leaf_fire_guard_active_ = false;
  previous_big_detected_leaf_count_ = -1;
  last_fire_key_ = {};
  last_raw_bullet_speed_ = 0.0;
  bullet_speed_buf_idx_ = 0;
  bullet_speed_buf_count_ = 0;
}

AngleEstimate::Snapshot Aimer::prediction_snapshot() const { return angle_estimator_.snapshot(); }

AngleEstimate::DebugSnapshot Aimer::prediction_debug_snapshot() const
{
  return angle_estimator_.debug_snapshot();
}

AngleEstimate::IdMatchDebug Aimer::id_match_debug() const
{
  return angle_estimator_.last_id_match_debug();
}

const AngleEstimateConfig & Aimer::prediction_config() const { return angle_estimator_.config(); }

bool Aimer::tune_prediction_config(const AngleEstimateConfig & config)
{
  return angle_estimator_.updateConfig(config);
}

double Aimer::prediction_angle(double abs_time) const { return angle_estimator_.predict(abs_time); }

double Aimer::prediction_angle(double abs_time, int leaf_id) const
{
  return angle_estimator_.predict(abs_time, leaf_id);
}

double Aimer::prediction_speed(double abs_time) const
{
  return angle_estimator_.predictSpeed(abs_time);
}

std::optional<Eigen::Matrix3d> Aimer::prediction_rotation_reference(double abs_time) const
{
  const auto & pose = pose_filter_.snapshot();
  if (!pose.valid || !angle_estimator_.has_successful_fit() || angle_estimator_.is_lost()) {
    return std::nullopt;
  }

  const double predicted_angle = angle_estimator_.predict(abs_time);
  if (!std::isfinite(predicted_angle)) return std::nullopt;

  const Eigen::Vector3d n = get_averaged_normal(pose.filtered_rotation_world);
  if (!n.allFinite()) return std::nullopt;
  return tools::rotation_matrix(Eigen::Vector3d(
    std::atan2(n.y(), n.x()),
    -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y())),
    predicted_angle + CV_PI / 2.0));
}

const BuffBallisticResult & Aimer::last_ballistic_result() const { return last_ballistic_result_; }

const BuffPoseSnapshot & Aimer::pose_snapshot() const { return pose_filter_.snapshot(); }

const BuffPoseFilterDebug & Aimer::pose_debug_snapshot() const
{
  return pose_filter_.debug_snapshot();
}

const CommandGuardDebug & Aimer::command_guard_debug() const { return last_command_guard_debug_; }

const AttackSelectionDebug & Aimer::attack_selection_debug() const
{
  return last_attack_selection_debug_;
}

double Aimer::tuned_bias_time_ms(BuffMode mode) const
{
  return mode == BuffMode::SMALL ? small_bias_time_ms_ : big_bias_time_ms_;
}

double Aimer::filter_bullet_speed(double raw_speed)
{
  // 仅当弹速发生变化时才注入窗口，避免相同值反复灌入导致滤波失效
  if (raw_speed != last_raw_bullet_speed_) {
    last_raw_bullet_speed_ = raw_speed;
    bullet_speed_buffer_[bullet_speed_buf_idx_] = raw_speed;
    bullet_speed_buf_idx_ = (bullet_speed_buf_idx_ + 1) % kBulletSpeedFilterWindow;
    if (bullet_speed_buf_count_ < kBulletSpeedFilterWindow) {
      ++bullet_speed_buf_count_;
    }
  }

  // 窗口为空（理论上不会走到这里，但保留兜底）
  if (bullet_speed_buf_count_ == 0) return raw_speed;

  double sum = 0.0;
  for (int i = 0; i < bullet_speed_buf_count_; ++i) {
    sum += bullet_speed_buffer_[i];
  }
  return sum / static_cast<double>(bullet_speed_buf_count_);
}

double Aimer::resolve_runtime_solve_delay_ms(
  const std::chrono::steady_clock::time_point & solve_start) const
{
  if (runtime_solve_delay_override_ms_.has_value()) {
    return *runtime_solve_delay_override_ms_;
  }
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solve_start)
    .count();
}

void Aimer::set_deterministic_replay(bool enabled)
{
  suppress_fire_ = enabled;
  angle_estimator_.setDeterministicReplay(enabled);
}

void Aimer::set_runtime_solve_delay_override_ms(std::optional<double> delay_ms)
{
  if (delay_ms.has_value() && (!std::isfinite(*delay_ms) || *delay_ms < 0.0)) {
    delay_ms.reset();
  }
  runtime_solve_delay_override_ms_ = delay_ms;
}

void Aimer::set_mpc_fire_gap_time_override(std::optional<double> fire_gap_time)
{
  if (fire_gap_time.has_value() &&
      (!std::isfinite(*fire_gap_time) || *fire_gap_time < 0.0)) {
    fire_gap_time.reset();
  }
  mpc_fire_gap_time_override_ = fire_gap_time;
}

AngleEstimate::IdMatchResult Aimer::update_observation(
  PowerRune & rune, double observed_time_abs, BuffMode mode)
{
  auto match = update_estimator(angle_estimator_, rune, observed_time_abs, mode); // 更新拟合器
  const bool epoch_changed = selection_epoch_ != match.epoch;
  if (epoch_changed) {
    tools::logger()->info(
      "[aimer] epoch_change {}->{} confirmed={} status={} leaf_id={}",
      selection_epoch_, match.epoch, match.confirmed(),
      static_cast<int>(match.status), match.leaf_id);
    reset_attack_selection();
    selection_epoch_ = match.epoch;
  }

  const int current_leaf_id = angle_estimator_.current_leaf_id();
  annotate_observation(
    rune, current_leaf_id >= 0 ? std::optional<int>{current_leaf_id} : std::nullopt);

  // 检测 confirmed 跳变（进入/离开）—— 这是 pose_filter 断更新的直接原因
  {
    static bool prev_confirmed = false;
    if (match.confirmed() != prev_confirmed) {
      tools::logger()->info(
        "[aimer] confirmed {}->{} epoch={} status={}",
        prev_confirmed, match.confirmed(), match.epoch, static_cast<int>(match.status));
      prev_confirmed = match.confirmed();
    }
  }

  if (match.confirmed()) {
    pose_filter_.update(rune, observed_time_abs);
  }
  return match;
}

void Aimer::annotate_observation(PowerRune & rune, std::optional<int> target_leaf_id)
{
  rune.target_leaf_id = -1;

  for (auto & blade : rune.fanblades) {
    blade.leaf_id = -1;
    blade.leaf_angle = 0.0;
    blade.leaf_angle_valid = false;
  }

  if (!target_leaf_id.has_value() || !rune.pnp_valid || rune.fanblades.empty()) return;
  rune.target_leaf_id = *target_leaf_id;

  for (std::size_t i = 0; i < rune.fanblades.size(); ++i) {
    auto & blade = rune.fanblades[i];
    if (blade.type == _unlight) continue;

    blade.leaf_id = (*target_leaf_id + static_cast<int>(i)) % 5;
    blade.leaf_angle =
      tools::limit_rad(rune.physical_angle + static_cast<double>(i) * kLeafSpacing);
    blade.leaf_angle_valid = true;
  }
}

void Aimer::reset_attack_selection()
{
  attack_leaf_id_ = -1;
  pending_attack_leaf_id_ = -1;
  switch_streak_ = 0;
  last_fire_key_ = {};  // epoch 变了，旧的开火记录作废
}

bool Aimer::tracking_ready() const
{
  return angle_estimator_.has_successful_fit() && !angle_estimator_.is_lost();
}

void Aimer::reset_fire_rearm() { confirmed_frames_since_guard_ = 0; }

void Aimer::note_confirmed_observation()
{
  confirmed_frames_since_guard_ =
    std::min(confirmed_frames_since_guard_ + 1, fire_rearm_confirmed_frames_);
}

bool Aimer::can_hold_last_leaf_plan(double observed_time_abs) const
{
  return has_last_plan_sample_ && std::isfinite(observed_time_abs) &&
         observed_time_abs >= last_plan_observed_time_abs_ &&
         observed_time_abs - last_plan_observed_time_abs_ <= hold_last_leaf_sec_;
}

io::Command Aimer::hold_last_leaf_command(double observed_time_abs) const
{
  io::Command command{false, false, 0, 0};
  if (!can_hold_last_leaf_plan(observed_time_abs)) return command;
  command.control = true;
  command.yaw = last_yaw_;
  command.pitch = last_pitch_;
  return command;
}

auto_aim::Plan Aimer::hold_last_leaf_plan(double observed_time_abs) const
{
  auto_aim::Plan plan{false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  if (!can_hold_last_leaf_plan(observed_time_abs)) return plan;
  plan.control = true;
  plan.yaw = static_cast<float>(last_yaw_);
  plan.pitch = static_cast<float>(last_pitch_);
  return plan;
}

void Aimer::limit_and_record_plan(
  auto_aim::Plan & plan, double observed_time_abs, const io::GimbalState & gs,
  double analytical_yaw_vel, double analytical_pitch_vel) // 传入解析速度前馈
{
  if (!plan.control) return;

  const bool has_fresh_last = can_hold_last_leaf_plan(observed_time_abs);
  const double reference_yaw = has_fresh_last ? last_yaw_ : static_cast<double>(gs.yaw);
  const double reference_pitch = has_fresh_last ? last_pitch_ : static_cast<double>(gs.pitch);
  const double raw_yaw = plan.yaw;
  const double raw_pitch = plan.pitch;

  const double raw_dt =
    has_fresh_last ? observed_time_abs - last_plan_observed_time_abs_ : max_slew_dt_sec_;
  const double dt = std::clamp(raw_dt, 1e-3, max_slew_dt_sec_);

  // 1. 连续角度展开
  const double raw_yaw_anchor = has_fresh_last ? last_raw_yaw_ : raw_yaw;
  const double unwrapped_raw_yaw =
    raw_yaw_anchor + tools::limit_rad(raw_yaw - raw_yaw_anchor);

  // 位置残差 e = 目标位置 - 当前参考位置
  double yaw_delta = unwrapped_raw_yaw - reference_yaw;
  const double pitch_delta = raw_pitch - reference_pitch;

  // 2. 角度增量限幅
  const double limited_yaw_delta = std::clamp(yaw_delta, -max_yaw_rate_ * dt, max_yaw_rate_ * dt);
  const double limited_pitch_delta =
    std::clamp(pitch_delta, -max_pitch_rate_ * dt, max_pitch_rate_ * dt);

  // ... debug 赋值保持不变 ...

  plan.target_yaw = static_cast<float>(raw_yaw);
  plan.target_pitch = static_cast<float>(raw_pitch);
  plan.yaw = static_cast<float>(raw_yaw);
  plan.pitch = static_cast<float>(raw_pitch);

  // ==================== 【核心修改部分】 ====================
  // 定义指数衰减比例系数 K_p
  constexpr double K_p_yaw = 8.0; 
  constexpr double K_p_pitch = 12.0;

  if (has_fresh_last) {
    // 解析前馈 + 位置误差反馈（对应消灭误差方程：e_dot + K_p * e = 0）
    // 优点：前馈极度平滑推着走，反馈快速吸附对准，无延迟、无高频噪声抖动
    plan.yaw_vel = static_cast<float>(analytical_yaw_vel + K_p_yaw * limited_yaw_delta);

    // 注意：pitch 在电控与视觉方向上的正负映射，根据你实际的坐标系确定
    plan.pitch_vel = static_cast<float>(analytical_pitch_vel + K_p_pitch * limited_pitch_delta); 
  } else {
    // 重新锁定时（无连续上一帧），直接使用纯反馈响应
    plan.yaw_vel = static_cast<float>(K_p_yaw * limited_yaw_delta);
    plan.pitch_vel = static_cast<float>(K_p_pitch * limited_pitch_delta);
  }

  // 抹平加速度
  plan.yaw_acc = 0.0f;
  plan.pitch_acc = 0.0f;
  // =========================================================

  last_yaw_ = plan.yaw;
  last_pitch_ = plan.pitch;
  last_raw_yaw_ = unwrapped_raw_yaw;
  last_yaw_delta_ = yaw_delta;
  last_pitch_delta_ = pitch_delta;
  last_plan_observed_time_abs_ = observed_time_abs;
  has_last_plan_sample_ = true;
}

void Aimer::limit_and_record_command(io::Command & command, double observed_time_abs)
{
  if (!command.control) return;
  if (can_hold_last_leaf_plan(observed_time_abs)) {
    const double dt =
      std::clamp(observed_time_abs - last_plan_observed_time_abs_, 1e-3, max_slew_dt_sec_);
    command.yaw = last_yaw_ + std::clamp(
                                tools::limit_rad(command.yaw - last_yaw_), -max_yaw_rate_ * dt,
                                max_yaw_rate_ * dt);
    command.pitch =
      last_pitch_ +
      std::clamp(command.pitch - last_pitch_, -max_pitch_rate_ * dt, max_pitch_rate_ * dt);
  }
  last_yaw_ = command.yaw;
  last_pitch_ = command.pitch;
  last_plan_observed_time_abs_ = observed_time_abs;
  has_last_plan_sample_ = true;
}

bool Aimer::can_aim_cached_r_center(const PowerRune & rune, double observed_time_abs) const
{
  const auto & pose = pose_filter_.snapshot();
  const double pose_age = observed_time_abs - pose.observed_time_abs;
  return std::isfinite(rune.r_center.x) && std::isfinite(rune.r_center.y) && pose.valid &&
         std::isfinite(pose_age) && pose_age >= 0.0 &&
         pose_age <= angle_estimator_.config().id_reset_timeout_sec;
}

// Track the transition from multiple visible blades to one visible blade in big-buff mode.
void Aimer::update_big_single_leaf_fire_guard(const PowerRune & rune, BuffMode mode)
{
  if (!big_single_leaf_fire_guard_enabled_) {
    big_single_leaf_fire_guard_active_ = false;
    previous_big_detected_leaf_count_ = -1;
    return;
  }
  if (mode != BuffMode::BIG) {
    big_single_leaf_fire_guard_active_ = false;
    previous_big_detected_leaf_count_ = -1;
    return;
  }

  const int detected_leaf_count = static_cast<int>(std::count_if(
    rune.fanblades.begin(), rune.fanblades.end(),
    [](const FanBlade & blade) { return blade.type != _unlight; }));

  if (detected_leaf_count >= 2) {
    big_single_leaf_fire_guard_active_ = false;
  } else if (detected_leaf_count == 1 && previous_big_detected_leaf_count_ >= 2) {
    big_single_leaf_fire_guard_active_ = true;
    big_single_leaf_start_t_ = std::chrono::steady_clock::now();
  }

  previous_big_detected_leaf_count_ = detected_leaf_count;
}

// 自动开火间隔
bool Aimer::should_fire()
{
  const auto now = std::chrono::steady_clock::now();
  if (tools::delta_time(now, last_fire_t_) <= fire_gap_time_) return false;
  last_fire_t_ = now;
  return true;
}

// 误差门控 + 切扇叶立即开火 + 同扇叶fly_time间隔
bool Aimer::should_fire(int leaf_id, double yaw_err_rad, double pitch_err_rad, double fly_time)
{
  if (std::abs(yaw_err_rad) > max_fire_yaw_error_ ||
      std::abs(pitch_err_rad) > max_fire_pitch_error_) {
    return false;
  }

  const auto now = std::chrono::steady_clock::now();
  const auto commit_fire = [&]() {
    if (big_single_leaf_fire_guard_active_) {
      const double elapsed = tools::delta_time(now, big_single_leaf_start_t_);
      std::cout << "single leaf elapsed time:" << elapsed << std::endl;
      if (fly_time >= kBigSingleLeafFireWindowSec - elapsed) return false;
    }
    last_fire_t_ = now;
    last_fire_key_ = {selection_epoch_, leaf_id, true};
    return true;
  };

  if (mpc_fire_gap_time_override_.has_value()) {
    if (tools::delta_time(now, last_fire_t_) <= *mpc_fire_gap_time_override_) return false;
    return commit_fire();
  }

  const bool blade_switched = last_fire_key_.valid &&
                              selection_epoch_ == last_fire_key_.epoch &&
                              leaf_id != last_fire_key_.leaf_id;
  if (blade_switched) {
    return commit_fire();
  }

  std::cout << "fly time:" << fly_time << std::endl;

  if (tools::delta_time(now, last_fire_t_) > 0.2 * fly_time) return commit_fire();
  return false;
}

BuffBallisticResult Aimer::solve_r_center(
  const PowerRune & rune, const BuffBallisticTiming & timing, double bullet_speed) const
{
  const auto & pose = pose_filter_.snapshot();
  const Eigen::Vector3d & r_center_world = pose.valid ? pose.filtered_xyz_world : rune.xyz_in_world;
  auto result = ballistic_.solve_once(r_center_world, bullet_speed);
  result.aim_source = BuffAimSource::R_CENTER;
  result.used_filtered_pose = pose.valid;
  result.observed_time_abs = timing.observed_time_abs;
  result.now_time_abs = timing.now_time_abs;
  result.observation_age_ms = timing.observation_age_ms();
  result.runtime_solve_delay_ms = timing.runtime_solve_delay_ms;
  result.tuned_bias_time_ms = timing.tuned_bias_time_ms;
  result.delay_time_ms = timing.delay_time_ms();
  return result;
}

// 构建云台转到yaw和pitch距离的代价函数进行选择，若上一帧已有目标扇叶，则依据运动连续性继续选择该id作为击打扇叶
void Aimer::select_attack_leaf(
  const PowerRune & rune, const BuffBallisticTiming & timing, const Eigen::Vector3d & center_world,
  const Eigen::Matrix3d & rotation_world, double bullet_speed, const io::GimbalState & gs,
  BuffBallisticResult & out_best)
{
  out_best = {};
  last_attack_selection_debug_ = {};
  auto & selection_debug = last_attack_selection_debug_;
  selection_debug.observed_leaf_id = angle_estimator_.current_leaf_id();
  selection_debug.previous_attack_leaf_id = attack_leaf_id_;
  selection_debug.switch_frames = sel_switch_frames_;
  const auto candidates = collect_attack_candidates(rune);
  selection_debug.candidate_count = static_cast<int>(candidates.size());
  if (candidates.empty()) {
    selection_debug.reason = AttackSelectionReason::NO_CANDIDATES;
    return;
  }

  const int observed_id = selection_debug.observed_leaf_id;
  // 在循环外调一次，所有候选 blade 共用同一个法向量，避免窗口在一帧内被多次更新
  const Eigen::Vector3d n = get_averaged_normal(rotation_world);
  const auto evaluate = [&](int leaf_id) {
    const double attack_angle_now = angle_estimator_.predict(timing.observed_time_abs, leaf_id);
    // const double attack_angle_now = tools::limit_rad(rune.physical_angle + leaf_id * 2.0 * CV_PI / 5.0);
    Eigen::Matrix3d rotation_world_for_leaf =
      tools::rotation_matrix(Eigen::Vector3d(
        std::atan2(n.y(), n.x()),
        -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y())),
        attack_angle_now + CV_PI / 2.0));
    return ballistic_.solve_with_iteration(
      center_world, rotation_world_for_leaf, attack_angle_now, timing, bullet_speed,
      [&, leaf_id](double predict_time_abs) {
        return angle_estimator_.predict(predict_time_abs, leaf_id);
      });
  };
  const auto cost = [&](const BuffBallisticResult & result) {
    const double dyaw = tools::limit_rad(result.yaw - static_cast<double>(gs.yaw));
    const double dpitch = result.pitch - static_cast<double>(-gs.pitch);
    return sel_w_yaw_ * std::abs(dyaw) + sel_w_pitch_ * std::abs(dpitch);
  };
  const auto finish = [&](BuffBallisticResult result, int leaf_id, double selection_cost) {
    result.aim_source = BuffAimSource::OBSERVED_LEAF;
    result.attack_leaf_id = leaf_id;
    result.observed_leaf_id = observed_id;
    result.selection_cost = selection_cost;
    out_best = result;
    attack_leaf_id_ = leaf_id;
  };

  int best_id = -1;
  BuffBallisticResult best_result;
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const int leaf_id = candidates[candidate_index];
    auto result = evaluate(leaf_id);
    auto & candidate_debug = selection_debug.candidates[candidate_index];
    candidate_debug.leaf_id = leaf_id;
    candidate_debug.valid = result.valid;
    candidate_debug.yaw = result.valid ? result.yaw : std::numeric_limits<double>::quiet_NaN();
    candidate_debug.pitch = result.valid ? result.pitch : std::numeric_limits<double>::quiet_NaN();
    if (!result.valid) continue;
    const double candidate_cost = cost(result);
    candidate_debug.cost = candidate_cost;
    if (candidate_cost < best_cost) {
      best_cost = candidate_cost;
      best_id = leaf_id;
      best_result = result;
    }
  }

  selection_debug.best_leaf_id = best_id;
  selection_debug.best_cost = best_id >= 0 ? best_cost : std::numeric_limits<double>::quiet_NaN();
  if (best_id < 0) {
    selection_debug.reason = AttackSelectionReason::NO_VALID_BALLISTIC;
    return;
  }

  const bool old_in_candidates =
    attack_leaf_id_ >= 0 &&
    std::find(candidates.begin(), candidates.end(), attack_leaf_id_) != candidates.end();
  BuffBallisticResult old_result;
  double old_cost = std::numeric_limits<double>::infinity();
  // 检查上一帧击打扇叶是否还可用（id有效 && 仍在当前帧候选集 && 弹道解算的结果有效）
  const bool old_valid = old_in_candidates && [&]() {
    old_result = attack_leaf_id_ == best_id ? best_result : evaluate(attack_leaf_id_);
    if (!old_result.valid) return false;
    old_cost = attack_leaf_id_ == best_id ? best_cost : cost(old_result);
    return true;
  }();
  selection_debug.previous_valid = old_valid;
  selection_debug.previous_cost = old_valid ? old_cost : std::numeric_limits<double>::quiet_NaN();
  const auto finish_debug = [&]() {
    selection_debug.selected_leaf_id = out_best.attack_leaf_id;
    selection_debug.pending_leaf_id = pending_attack_leaf_id_;
    selection_debug.switch_streak = switch_streak_;
  };

  if (!old_valid) {
    if (attack_leaf_id_ < 0) {
      pending_attack_leaf_id_ = -1;
      switch_streak_ = 0;
      finish(best_result, best_id, best_cost);
      selection_debug.reason = AttackSelectionReason::INITIAL_SELECTION;
      finish_debug();
      return;
    }

    // 旧叶暂时不可见时也要连续确认，避免单帧漏检直接跳向新叶。
    if (best_id == pending_attack_leaf_id_) {
      ++switch_streak_;
    } else {
      pending_attack_leaf_id_ = best_id;
      switch_streak_ = 1;
    }
    if (switch_streak_ < sel_switch_frames_) {
      selection_debug.reason = AttackSelectionReason::HOLD_MISSING_PREVIOUS;
      finish_debug();
      return;
    }

    pending_attack_leaf_id_ = -1;
    switch_streak_ = 0;
    finish(best_result, best_id, best_cost);
    selection_debug.reason = AttackSelectionReason::SWITCH_MISSING_PREVIOUS;
  } else if (best_id == attack_leaf_id_) {
    // 最优片仍是上一帧击打片，清空迟滞状态，保持连续跟踪。
    pending_attack_leaf_id_ = -1;
    switch_streak_ = 0;
    finish(old_result, attack_leaf_id_, old_cost);
    selection_debug.reason = AttackSelectionReason::KEEP_CURRENT;
  } else {
    const bool better_enough = best_cost + sel_switch_margin_ < old_cost;
    selection_debug.better_enough = better_enough;
    // 如果新的击打扇叶的cost比和上一帧击打扇叶同id的cost要少 sel_switch_margin_，才认为值得切换，且需要多帧确认
    if (best_id == pending_attack_leaf_id_ && better_enough) {
      ++switch_streak_;
    } else {
      pending_attack_leaf_id_ = best_id;
      switch_streak_ = better_enough ? 1 : 0;
    }

    if (switch_streak_ >= sel_switch_frames_) {
      // 主动切换必须是同一候选连续多帧更优，避免两片亮扇叶之间抖动。
      pending_attack_leaf_id_ = -1;
      switch_streak_ = 0;
      finish(best_result, best_id, best_cost);
      selection_debug.reason = AttackSelectionReason::SWITCH_BETTER;
    } else {
      finish(old_result, attack_leaf_id_, old_cost);
      selection_debug.reason = AttackSelectionReason::HOLD_SWITCH_HYSTERESIS;
    }
  }
  finish_debug();

  tools::logger()->debug(
    "[Buff TargetSelection] observed={} attack={} best={} best_cost={:.4f} old_cost={:.4f} "
    "pending={} streak={}/{}",
    observed_id, out_best.attack_leaf_id, best_id, best_cost, old_cost, pending_attack_leaf_id_,
    switch_streak_, sel_switch_frames_);
}

void Aimer::calculate_analytical_velocities(
  const BuffBallisticTiming & timing, 
  const Eigen::Vector3d & center_world, 
  const Eigen::Matrix3d & rotation_world, 
  double bullet_speed, 
  int attack_leaf_id,
  double & out_yaw_vel, 
  double & out_pitch_vel) const
{
  out_yaw_vel = 0.0;
  out_pitch_vel = 0.0;

  // 必须保证前一步的弹道解算已经成功
  if (!last_ballistic_result_.valid) return;

  constexpr double delta_t = 0.002; // 2 ms 微小步长

  // 1. 当前时刻击打点预测绝对时间
  const double predict_time_now = last_ballistic_result_.predict_time_abs;
  // 2. 微小步长后的绝对时间
  const double predict_time_next = predict_time_now + delta_t;

  // 3. 利用 AngleEstimate 直接计算未来的角度解析值（平滑无噪）
  const double angle_next = angle_estimator_.predict(predict_time_next, attack_leaf_id);

  // 4. 构建未来时刻的姿态旋转矩阵
  const Eigen::Vector3d n = get_averaged_normal(rotation_world);
  const Eigen::Matrix3d rotation_world_next =
    tools::rotation_matrix(Eigen::Vector3d(
      std::atan2(n.y(), n.x()),
      -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y())),
      angle_next + CV_PI / 2.0));

  // 5. 求解未来时刻的弹道结果
  auto result_next = ballistic_.solve_with_iteration(
    center_world, rotation_world_next, angle_next, timing, bullet_speed,
    [&, attack_leaf_id](double p_time) {
      return angle_estimator_.predict(p_time, attack_leaf_id);
    });

  if (result_next.valid) {
    // 6. 通过解析差分提取理想速度 (rad/s)
    const double dyaw = tools::limit_rad(result_next.yaw - last_ballistic_result_.yaw);
    const double dpitch = result_next.pitch - last_ballistic_result_.pitch;

    out_yaw_vel = dyaw / delta_t;
    out_pitch_vel = dpitch / delta_t;
  }
}

Eigen::Vector3d Aimer::get_averaged_normal(const Eigen::Matrix3d & rotation_world) const
{
  if (!normal_avg_enabled_ || !rotation_world.allFinite()) {
    return rotation_world.col(0);
  }

  const Eigen::Vector3d n = rotation_world.col(0);

  // 异常值过滤：窗口有足够样本时，拒绝偏离均值超过 3σ 的帧
  if (static_cast<int>(normal_history_.size()) >= kNormalMinSamplesForOutlier) {
    const double count = static_cast<double>(normal_history_.size());
    const Eigen::Vector3d mean = normal_sum_ / count;
    const Eigen::Vector3d sq_mean = normal_sum_sq_ / count;
    const Eigen::Vector3d variance(
      std::max(0.0, sq_mean.x() - mean.x() * mean.x()),
      std::max(0.0, sq_mean.y() - mean.y() * mean.y()),
      std::max(0.0, sq_mean.z() - mean.z() * mean.z()));
    const double dist = (n - mean).norm();
    const double threshold =
      kNormalOutlierSigma * std::sqrt(variance.x() + variance.y() + variance.z());
    if (dist > threshold && threshold > 1e-8) {
      return mean;  // 异常帧，返回当前均值但不参与更新
    }
  }

  // 变化检测：与窗口最后一个样本比较，变化小于阈值则跳过入窗
  // threshold=0 时关闭 deadband，每帧都入窗（纯滑动平均）
  if (normal_avg_change_threshold_ > 0.0 && !normal_history_.empty()) {
    const double dist_to_last = (n - normal_history_.back()).norm();
    if (dist_to_last < normal_avg_change_threshold_) {
      return normal_sum_ / static_cast<double>(normal_history_.size());
    }
  }

  // 加入滑动窗口
  normal_history_.push_back(n);
  normal_sum_ += n;
  normal_sum_sq_ += Eigen::Vector3d(n.x() * n.x(), n.y() * n.y(), n.z() * n.z());

  // 超出窗口大小时移除最旧样本
  while (static_cast<int>(normal_history_.size()) > normal_avg_window_size_) {
    const Eigen::Vector3d old = normal_history_.front();
    normal_history_.pop_front();
    normal_sum_ -= old;
    normal_sum_sq_ -= Eigen::Vector3d(old.x() * old.x(), old.y() * old.y(), old.z() * old.z());
  }

  if (normal_history_.empty()) return n;
  return normal_sum_ / static_cast<double>(normal_history_.size());
}

}  // namespace auto_buff
