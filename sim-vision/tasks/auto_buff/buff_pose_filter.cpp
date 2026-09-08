#include "buff_pose_filter.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{
namespace
{
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

double wrap_angle(double angle) { return tools::limit_rad(angle); }

double rotation_distance(const Eigen::Matrix3d & from, const Eigen::Matrix3d & to)
{
  const Eigen::Matrix3d relative = from.transpose() * to;
  const double cosine = std::clamp((relative.trace() - 1.0) * 0.5, -1.0, 1.0);
  return std::acos(cosine);
}

bool finite_scalar(double value) { return std::isfinite(value); }

bool finite_vector(const Eigen::Vector3d & value)
{
  return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

Eigen::VectorXd add_state(const Eigen::VectorXd & a, const Eigen::VectorXd & b)
{
  Eigen::VectorXd out = a + b;
  out[BuffPoseFilter::kRYaw]    = wrap_angle(out[BuffPoseFilter::kRYaw]);
  out[BuffPoseFilter::kRPitch]  = wrap_angle(out[BuffPoseFilter::kRPitch]);
  out[BuffPoseFilter::kBuffYaw] = wrap_angle(out[BuffPoseFilter::kBuffYaw]);
  out[BuffPoseFilter::kBuffPitch] = wrap_angle(out[BuffPoseFilter::kBuffPitch]);
  return out;
}

Eigen::VectorXd subtract_linear_obs(const Eigen::VectorXd & a, const Eigen::VectorXd & b)
{
  Eigen::VectorXd out = a - b;
  out[0] = wrap_angle(out[0]);  // R_yaw
  out[1] = wrap_angle(out[1]);  // R_pitch
  out[3] = wrap_angle(out[3]);  // buff_yaw
  out[4] = wrap_angle(out[4]);  // buff_pitch
  return out;
}

Eigen::VectorXd subtract_blade_obs(const Eigen::VectorXd & a, const Eigen::VectorXd & b)
{
  Eigen::VectorXd out = a - b;
  out[0] = wrap_angle(out[0]);
  out[1] = wrap_angle(out[1]);
  return out;
}
}  // namespace

PoseFilterConfig PoseFilterConfig::from_yaml(const std::string & config_path)
{
  PoseFilterConfig config;
  const auto yaml = YAML::LoadFile(config_path);
  const auto pose_filter = yaml["pose_filter"];

  read_optional(yaml, pose_filter, "pose_filter_enabled", "enabled", config.enabled);
  read_optional(yaml, pose_filter, "pose_filter_max_dt_sec", "max_dt_sec", config.max_dt_sec);

  read_optional(yaml, pose_filter, "pose_filter_p0_r_yaw", "p0_r_yaw", config.p0_r_yaw);
  read_optional(yaml, pose_filter, "pose_filter_p0_r_v_yaw", "p0_r_v_yaw", config.p0_r_v_yaw);
  read_optional(yaml, pose_filter, "pose_filter_p0_r_pitch", "p0_r_pitch", config.p0_r_pitch);
  read_optional(yaml, pose_filter, "pose_filter_p0_r_dist", "p0_r_dist", config.p0_r_dist);
  read_optional(yaml, pose_filter, "pose_filter_p0_buff_yaw", "p0_buff_yaw", config.p0_buff_yaw);

  read_optional(
    yaml, pose_filter, "pose_filter_q_r_yaw_accel", "q_r_yaw_accel", config.q_r_yaw_accel);
  read_optional(yaml, pose_filter, "pose_filter_q_r_pitch", "q_r_pitch", config.q_r_pitch);
  read_optional(yaml, pose_filter, "pose_filter_q_r_dist", "q_r_dist", config.q_r_dist);
  read_optional(yaml, pose_filter, "pose_filter_q_buff_yaw", "q_buff_yaw", config.q_buff_yaw);

  read_optional(yaml, pose_filter, "pose_filter_r_r_yaw", "r_r_yaw", config.r_r_yaw);
  read_optional(yaml, pose_filter, "pose_filter_r_r_pitch", "r_r_pitch", config.r_r_pitch);
  read_optional(yaml, pose_filter, "pose_filter_r_r_dist", "r_r_dist", config.r_r_dist);
  read_optional(yaml, pose_filter, "pose_filter_r_buff_yaw", "r_buff_yaw", config.r_buff_yaw);
  read_optional(
    yaml, pose_filter, "pose_filter_r_buff_pitch", "r_buff_pitch", config.r_buff_pitch);
  read_optional(
    yaml, pose_filter, "pose_filter_p0_buff_pitch", "p0_buff_pitch", config.p0_buff_pitch);
  read_optional(
    yaml, pose_filter, "pose_filter_q_buff_pitch", "q_buff_pitch", config.q_buff_pitch);

  read_optional(yaml, pose_filter, "pose_filter_r_blade_yaw", "r_blade_yaw", config.r_blade_yaw);
  read_optional(
    yaml, pose_filter, "pose_filter_r_blade_pitch", "r_blade_pitch", config.r_blade_pitch);
  read_optional(yaml, pose_filter, "pose_filter_r_blade_dist", "r_blade_dist", config.r_blade_dist);

  read_optional(
    yaml, pose_filter, "pose_filter_use_blade_obs", "use_blade_obs", config.use_blade_obs);

  // ── 自适应过程噪声 ──
  auto read_adaptive_channel = [&](
    const std::string & prefix, PoseFilterConfig::AdaptiveQChannel & ch) {
    read_optional(yaml, pose_filter, prefix + "_enabled", prefix + "_enabled", ch.enabled);
    read_optional(yaml, pose_filter, prefix + "_high_factor", prefix + "_high_factor", ch.high_factor);
    read_optional(yaml, pose_filter, prefix + "_decay", prefix + "_decay", ch.decay);
    read_optional(
      yaml, pose_filter, prefix + "_residual_threshold", prefix + "_residual_threshold",
      ch.residual_threshold);
    read_optional(yaml, pose_filter, prefix + "_min_hits", prefix + "_min_hits", ch.min_hits);
    ch.decay = std::clamp(ch.decay, 0.0, 1.0);
    ch.high_factor = std::max(1.0, ch.high_factor);
    ch.min_hits = std::max(1, ch.min_hits);
  };
  read_adaptive_channel("adaptive_q_r_yaw", config.adaptive_q_r_yaw);
  read_adaptive_channel("adaptive_q_r_pitch", config.adaptive_q_r_pitch);
  read_adaptive_channel("adaptive_q_r_dist", config.adaptive_q_r_dist);
  read_adaptive_channel("adaptive_q_buff_yaw", config.adaptive_q_buff_yaw);
  read_adaptive_channel("adaptive_q_buff_pitch", config.adaptive_q_buff_pitch);

  // ── Blade 软门控 ──
  read_optional(
    yaml, pose_filter, "pose_filter_blade_soft_gate", "blade_soft_gate", config.blade_soft_gate);

  // ── NIS 发散检测 ──
  read_optional(
    yaml, pose_filter, "pose_filter_nis_detection", "nis_detection", config.nis_detection);
  read_optional(
    yaml, pose_filter, "pose_filter_nis_window_size", "nis_window_size", config.nis_window_size);
  read_optional(
    yaml, pose_filter, "pose_filter_nis_fail_ratio", "nis_fail_ratio", config.nis_fail_ratio);
  read_optional(
    yaml, pose_filter, "pose_filter_nis_threshold", "nis_threshold", config.nis_threshold);
  read_optional(
    yaml, pose_filter, "pose_filter_nis_joint_threshold", "nis_joint_threshold",
    config.nis_joint_threshold);
  config.nis_window_size = std::max(5, config.nis_window_size);
  config.nis_fail_ratio = std::clamp(config.nis_fail_ratio, 0.0, 1.0);

  return config;
}

BuffPoseFilter::BuffPoseFilter(const PoseFilterConfig & config, double target_radius)
: config_(config),
  target_radius_(target_radius),
  current_q_r_yaw_accel_(config.q_r_yaw_accel),
  current_q_r_pitch_(config.q_r_pitch),
  current_q_r_dist_(config.q_r_dist),
  current_q_buff_yaw_(config.q_buff_yaw),
  current_q_buff_pitch_(config.q_buff_pitch)
{
  H1_(0, kRYaw)     = 1.0;
  H1_(1, kRPitch)   = 1.0;
  H1_(2, kRDist)    = 1.0;
  H1_(3, kBuffYaw)  = 1.0;
  H1_(4, kBuffPitch) = 1.0;

  R1_(0, 0) = config_.r_r_yaw;
  R1_(1, 1) = config_.r_r_pitch;
  R1_(2, 2) = config_.r_r_dist;
  R1_(3, 3) = config_.r_buff_yaw;
  R1_(4, 4) = config_.r_buff_pitch;

  R2_(0, 0) = config_.r_blade_yaw;
  R2_(1, 1) = config_.r_blade_pitch;
  R2_(2, 2) = config_.r_blade_dist;
}

bool BuffPoseFilter::finite_pose_input(const PowerRune & rune) const
{
  return rune.pnp_valid && finite_vector(rune.ypd_in_world) &&
         finite_vector(rune.blade_ypd_in_world) && finite_vector(rune.ypr_in_world);
}

void BuffPoseFilter::reset()
{
  initialized_ = false;
  last_time_ = 0.0;
  snapshot_ = {};

  // 重置自适应状态
  current_q_r_yaw_accel_ = config_.q_r_yaw_accel;
  current_q_r_pitch_ = config_.q_r_pitch;
  current_q_r_dist_ = config_.q_r_dist;
  current_q_buff_yaw_ = config_.q_buff_yaw;
  current_q_buff_pitch_ = config_.q_buff_pitch;
  adaptive_hit_r_yaw_ = 0;
  adaptive_hit_r_pitch_ = 0;
  adaptive_hit_r_dist_ = 0;
  adaptive_hit_buff_pitch_ = 0;
  adaptive_boost_r_yaw_ = false;
  adaptive_boost_r_pitch_ = false;
  adaptive_boost_r_dist_ = false;
  adaptive_boost_buff_yaw_ = false;
  adaptive_boost_buff_pitch_ = false;
  recent_nis_failures_.clear();
}

// 用pnp结果进行初始化
void BuffPoseFilter::initialize(const PowerRune & rune, double observed_time_abs)
{
  Eigen::VectorXd x0(kStateSize);
  x0 << rune.ypd_in_world[0], 0.0, rune.ypd_in_world[1], rune.ypd_in_world[2],
        rune.ypr_in_world[0], rune.ypr_in_world[1];

  Eigen::MatrixXd P0 = Eigen::MatrixXd::Zero(kStateSize, kStateSize);
  P0(kRYaw,     kRYaw)     = config_.p0_r_yaw;
  P0(kRvyaw,    kRvyaw)    = config_.p0_r_v_yaw;
  P0(kRPitch,   kRPitch)   = config_.p0_r_pitch;
  P0(kRDist,    kRDist)    = config_.p0_r_dist;
  P0(kBuffYaw,  kBuffYaw)  = config_.p0_buff_yaw;
  P0(kBuffPitch,kBuffPitch) = config_.p0_buff_pitch;

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, add_state);
  initialized_ = true;
  last_time_ = observed_time_abs;
  rebuild_snapshot(rune.ypr_in_world[2], observed_time_abs);
}

// R标中心到扇叶中心的偏移量
Eigen::Vector3d BuffPoseFilter::rotated_leaf_offset(
  double buff_yaw, double buff_pitch, double raw_roll) const
{
  const Eigen::Matrix3d rotation =
    tools::rotation_matrix(Eigen::Vector3d(buff_yaw, buff_pitch, raw_roll));
  return rotation * Eigen::Vector3d(0.0, 0.0, target_radius_);
}

// R标中心到扇叶中心的偏移量对符yaw的偏导数
Eigen::Vector3d BuffPoseFilter::rotated_leaf_offset_dyaw(
  double buff_yaw, double buff_pitch, double raw_roll) const
{
  const Eigen::Matrix3d rotation =
    tools::rotation_matrix(Eigen::Vector3d(buff_yaw, buff_pitch, raw_roll));
  const Eigen::Vector3d local(0.0, 0.0, target_radius_);
  return Eigen::Vector3d::UnitZ().cross(rotation * local);
}

Eigen::Vector3d BuffPoseFilter::rotated_leaf_offset_dpitch(
  double buff_yaw, double buff_pitch, double raw_roll) const
{
  // 数值微分：∂(R*(0,0,r))/∂pitch，精度足够EKF使用
  constexpr double eps = 1e-6;
  return (rotated_leaf_offset(buff_yaw, buff_pitch + eps, raw_roll) -
          rotated_leaf_offset(buff_yaw, buff_pitch - eps, raw_roll)) /
         (2.0 * eps);
}

// z2的观测模型，给定当前状态，预测"目标扇叶中心在球坐标系下应该在哪儿
Eigen::Vector3d BuffPoseFilter::predict_blade_ypd(
  const Eigen::VectorXd & state, double raw_roll) const
{
  const Eigen::Vector3d filtered_ypd(state[kRYaw], state[kRPitch], state[kRDist]);
  const Eigen::Vector3d center_world = tools::ypd2xyz(filtered_ypd);
  const Eigen::Vector3d leaf_world =
    center_world + rotated_leaf_offset(state[kBuffYaw], state[kBuffPitch], raw_roll);
  return tools::xyz2ypd(leaf_world);
}

Eigen::Matrix<double, BuffPoseFilter::kBladeObsSize, BuffPoseFilter::kStateSize> BuffPoseFilter::blade_observation_jacobian(
  const Eigen::VectorXd & state, double raw_roll) const
{
  Eigen::Matrix<double, BuffPoseFilter::kBladeObsSize, BuffPoseFilter::kStateSize> H =
    Eigen::Matrix<double, BuffPoseFilter::kBladeObsSize, BuffPoseFilter::kStateSize>::Zero();

  const Eigen::Vector3d filtered_ypd(state[kRYaw], state[kRPitch], state[kRDist]);
  const Eigen::Vector3d center_world = tools::ypd2xyz(filtered_ypd);
  const Eigen::MatrixXd center_jacobian = tools::ypd2xyz_jacobian(filtered_ypd);
  const Eigen::Vector3d leaf_offset =
    rotated_leaf_offset(state[kBuffYaw], state[kBuffPitch], raw_roll);
  const Eigen::Vector3d leaf_offset_dyaw =
    rotated_leaf_offset_dyaw(state[kBuffYaw], state[kBuffPitch], raw_roll);
  const Eigen::Vector3d leaf_offset_dpitch =
    rotated_leaf_offset_dpitch(state[kBuffYaw], state[kBuffPitch], raw_roll);
  const Eigen::Vector3d leaf_world = center_world + leaf_offset;
  const Eigen::MatrixXd ypd_jacobian = tools::xyz2ypd_jacobian(leaf_world);

  H.block<kBladeObsSize, 1>(0, kRYaw)      = ypd_jacobian * center_jacobian.col(0);
  H.block<kBladeObsSize, 1>(0, kRPitch)    = ypd_jacobian * center_jacobian.col(1);
  H.block<kBladeObsSize, 1>(0, kRDist)     = ypd_jacobian * center_jacobian.col(2);
  H.block<kBladeObsSize, 1>(0, kBuffYaw)   = ypd_jacobian * leaf_offset_dyaw;
  H.block<kBladeObsSize, 1>(0, kBuffPitch) = ypd_jacobian * leaf_offset_dpitch;
  return H;
}

// 从 EKF 的当前状态向量重建输出快照
void BuffPoseFilter::rebuild_snapshot(double raw_roll, double observed_time_abs)
{
  snapshot_ = {};
  if (!initialized_) return;

  const Eigen::Vector3d filtered_ypd(ekf_.x[kRYaw], ekf_.x[kRPitch], ekf_.x[kRDist]);
  const Eigen::Vector3d filtered_xyz = tools::ypd2xyz(filtered_ypd);
  const Eigen::Matrix3d filtered_rotation =
    tools::rotation_matrix(
      Eigen::Vector3d(ekf_.x[kBuffYaw], ekf_.x[kBuffPitch], raw_roll));
  const Eigen::Vector3d filtered_target_center =
    filtered_xyz + filtered_rotation * Eigen::Vector3d(0.0, 0.0, target_radius_);

  if (
    !finite_vector(filtered_ypd) || !finite_vector(filtered_xyz) ||
    !filtered_rotation.allFinite() || !finite_vector(filtered_target_center)) {
    snapshot_ = {};
    return;
  }

  snapshot_.valid = true;
  snapshot_.filtered_ypd_world = filtered_ypd;
  snapshot_.filtered_xyz_world = filtered_xyz;
  snapshot_.filtered_buff_yaw = ekf_.x[kBuffYaw];
  snapshot_.filtered_rotation_world = filtered_rotation;
  snapshot_.filtered_target_center_world = filtered_target_center;
  snapshot_.observed_time_abs = observed_time_abs;
}

void BuffPoseFilter::record_output_delta(const PowerRune & rune)
{
  if (!snapshot_.valid || !rune.rotation_world.allFinite()) return;
  last_debug_.raw_minus_filtered_r_xyz = rune.xyz_in_world - snapshot_.filtered_xyz_world;
  last_debug_.raw_minus_filtered_target_xyz =
    rune.target_center_world - snapshot_.filtered_target_center_world;
  last_debug_.raw_to_filtered_rotation_distance =
    rotation_distance(rune.rotation_world, snapshot_.filtered_rotation_world);
}

bool BuffPoseFilter::update(const PowerRune & rune, double observed_time_abs)
{
  last_debug_ = {};
  last_debug_.enabled = config_.enabled;
  last_debug_.input_valid = finite_pose_input(rune) && finite_scalar(observed_time_abs);
  if (!config_.enabled) {
    last_debug_.update_kind = PoseFilterUpdateKind::DISABLED;
    reset();
    return false;
  }
  if (!last_debug_.input_valid) {
    last_debug_.update_kind = PoseFilterUpdateKind::INVALID_INPUT;
    reset();
    return false;
  }

  if (!initialized_) {
    initialize(rune, observed_time_abs);
    tools::logger()->info(
      "[pose_filter] INIT R_dist={:.3f}m R_yaw={:.3f}rad buff_yaw={:.3f}rad buff_pitch={:.3f}rad",
      ekf_.x[kRDist], ekf_.x[kRYaw], ekf_.x[kBuffYaw], ekf_.x[kBuffPitch]);
    last_debug_.update_kind = PoseFilterUpdateKind::INITIALIZED;
    last_debug_.state_after = ekf_.x;
    record_output_delta(rune);
    return snapshot_.valid;
  }

  const double dt = observed_time_abs - last_time_;
  last_debug_.dt = dt;
  if (!std::isfinite(dt) || dt <= 0.0 || dt > config_.max_dt_sec) {
    const double old_r_dist = ekf_.x[kRDist];
    const double old_r_yaw = ekf_.x[kRYaw];
    const double old_buff_yaw = ekf_.x[kBuffYaw];
    const double old_buff_pitch = ekf_.x[kBuffPitch];
    initialize(rune, observed_time_abs);
    tools::logger()->info(
      "[pose_filter] REINIT_TIME dt={:.4f}s max={:.3f}s | "
      "R_dist {:.3f}->{:.3f}m buff_yaw {:.3f}->{:.3f}rad r_yaw {:.3f}->{:.3f}rad buff_pitch {:.3f}->{:.3f}rad",
      dt, config_.max_dt_sec,
      old_r_dist, ekf_.x[kRDist],
      old_buff_yaw, ekf_.x[kBuffYaw],
      old_r_yaw, ekf_.x[kRYaw],
      old_buff_pitch, ekf_.x[kBuffPitch]);
    last_debug_.update_kind = PoseFilterUpdateKind::REINITIALIZED_TIME;
    last_debug_.state_after = ekf_.x;
    record_output_delta(rune);
    return snapshot_.valid;
  }

  // 状态转移方程
  // R_yaw(k+1)    = R_yaw(k) + dt * R_v_yaw(k)
  // R_v_yaw(k+1)  = R_v_yaw(k)
  // R_pitch       = R_pitch
  // R_dist        = R_dist
  // buff_yaw      = buff_yaw
  // buff_pitch    = buff_pitch

  F_.setIdentity();
  F_(kRYaw, kRvyaw) = dt;

  const double dt2 = dt * dt;
  const double dt3 = dt2 * dt;
  const double dt4 = dt2 * dt2;
  Q_.setZero();
  Q_(kRYaw,    kRYaw)    = dt4 * current_q_r_yaw_accel_ / 4.0;
  Q_(kRYaw,    kRvyaw)   = dt3 * current_q_r_yaw_accel_ / 2.0;
  Q_(kRvyaw,   kRYaw)    = dt3 * current_q_r_yaw_accel_ / 2.0;
  Q_(kRvyaw,   kRvyaw)   = dt2 * current_q_r_yaw_accel_;
  Q_(kRPitch,    kRPitch)    = current_q_r_pitch_;
  Q_(kRDist,     kRDist)     = current_q_r_dist_;
  Q_(kBuffYaw,   kBuffYaw)   = current_q_buff_yaw_;
  Q_(kBuffPitch, kBuffPitch) = current_q_buff_pitch_;

  last_debug_.state_before = ekf_.x;
  ekf_.predict(F_, Q_, [&](const Eigen::VectorXd & x) {
    Eigen::VectorXd prior = F_ * x;
    prior[kRYaw]     = wrap_angle(prior[kRYaw]);
    prior[kRPitch]   = wrap_angle(prior[kRPitch]);
    prior[kBuffYaw]  = wrap_angle(prior[kBuffYaw]);
    prior[kBuffPitch] = wrap_angle(prior[kBuffPitch]);
    return prior;
  });
  last_debug_.state_predicted = ekf_.x;
  decay_adaptive_process_noise();

  // 合并 z1（线性：R标球坐标 + buff_yaw + buff_pitch）和 z2（非线性：扇叶球坐标）为单次联合更新。
  // 在同一先验 P 下处理两组测量，避免 z1 先更新压缩 P 后 z2 增益被低估。
  // H2 在 predict 后的先验状态上线性化（而非 z1 更新后的后验状态）。
  const double raw_roll = rune.ypr_in_world[2];

  Eigen::Matrix<double, kLinearObsSize, 1> z1;
  z1 << rune.ypd_in_world[0], rune.ypd_in_world[1], rune.ypd_in_world[2],
        rune.ypr_in_world[0], rune.ypr_in_world[1];
  const Eigen::Vector3d z2 = rune.blade_ypd_in_world;

  // 调试新息：在更新前用先验状态计算
  last_debug_.linear_innovation = subtract_linear_obs(z1, H1_ * ekf_.x);
  last_debug_.blade_innovation  = subtract_blade_obs(
    z2, predict_blade_ypd(ekf_.x, raw_roll));

  if (config_.use_blade_obs) {
    // ── 联合更新：z1（R标+符朝向）+ z2（blade）──
    // 联合观测向量 [z1(5); z2(3)]
    Eigen::Matrix<double, kJointObsSize, 1> z_joint;
    z_joint.head<kLinearObsSize>() = z1;
    z_joint.tail<kBladeObsSize>()  = z2;

    // 联合 Jacobian [H1; H2]，H2 在先验状态线性化
    const auto H2 = blade_observation_jacobian(ekf_.x, raw_roll);
    Eigen::Matrix<double, kJointObsSize, kStateSize> H_joint;
    H_joint.topRows<kLinearObsSize>()   = H1_;
    H_joint.bottomRows<kBladeObsSize>() = H2;

    // blade 观测门控：软门控（连续衰减）或硬截断（旧行为）
    Eigen::Matrix<double, kBladeObsSize, kBladeObsSize> R2_adaptive = R2_;
    if (config_.blade_soft_gate) {
      auto soft_gate = [](double innov, double base_r) -> double {
        constexpr double kBladeGate = 0.15;
        const double ratio = std::abs(innov) / kBladeGate;
        if (ratio <= 1.0) return base_r;
        return base_r * ratio * ratio;  // 平方连续放大
      };
      R2_adaptive(0, 0) = soft_gate(last_debug_.blade_innovation[0], R2_(0, 0));
      R2_adaptive(1, 1) = soft_gate(last_debug_.blade_innovation[1], R2_(1, 1));
      R2_adaptive(2, 2) = soft_gate(last_debug_.blade_innovation[2], R2_(2, 2));
    } else {
      const double blade_dist_innov = std::abs(last_debug_.blade_innovation[2]);
      constexpr double kBladeDistGate = 0.2;
      if (blade_dist_innov > kBladeDistGate) {
        R2_adaptive(2, 2) = 1e4;  // 增益 ≈ 0，blade 距离测量不影响状态
        tools::logger()->debug(
          "[pose_filter] blade dist GATE: |innov|={:.3f} > {:.2f}",
          blade_dist_innov, kBladeDistGate);
      }
    }

    Eigen::Matrix<double, kJointObsSize, kJointObsSize> R_joint =
      Eigen::Matrix<double, kJointObsSize, kJointObsSize>::Zero();
    R_joint.topLeftCorner<kLinearObsSize, kLinearObsSize>()   = R1_;
    R_joint.bottomRightCorner<kBladeObsSize, kBladeObsSize>() = R2_adaptive;

    const auto predict_joint = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      Eigen::Matrix<double, kJointObsSize, 1> h;
      h.head<kLinearObsSize>() = H1_ * x;
      h.tail<kBladeObsSize>()  = predict_blade_ypd(x, raw_roll);
      return h;
    };

    const auto subtract_joint = [](const Eigen::VectorXd & a,
                                    const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd out = a - b;
      out[0] = wrap_angle(out[0]);
      out[1] = wrap_angle(out[1]);
      out[3] = wrap_angle(out[3]);
      out[4] = wrap_angle(out[4]);
      out[5] = wrap_angle(out[5]);
      out[6] = wrap_angle(out[6]);
      return out;
    };

    ekf_.update(z_joint, H_joint, R_joint, predict_joint, subtract_joint);
  } else {
    // ── 纯线性更新：只用 z1（R标+符朝向），不使用 blade 观测 ──
    ekf_.update(z1, H1_, R1_,
      [&](const Eigen::VectorXd & x) -> Eigen::VectorXd { return H1_ * x; },
      subtract_linear_obs);
  }

  // NIS 发散检测 + 自适应过程噪声触发
  update_nis_window(
    ekf_.data.at("nis"),
    config_.use_blade_obs ? kJointObsSize : kLinearObsSize);
  update_adaptive_process_noise();

  last_time_ = observed_time_abs;
  rebuild_snapshot(raw_roll, observed_time_abs);
  last_debug_.update_kind = PoseFilterUpdateKind::UPDATED;
  last_debug_.state_after = ekf_.x;
  record_output_delta(rune);

  // Debug: 统计线性 innovation 每 30 帧输出一次，用于诊断 Q/R 参数
  {
    static int frame_count = 0;
    frame_count++;

    // 只统计 use_blade_obs=false 的情况（纯线性 KF），z1 = [R_yaw, R_pitch, R_dist, buff_yaw, buff_pitch]
    struct InnovStats {
      double sum[5] = {}, sum_sq[5] = {};
      double r_dist_raw = 0, r_dist_filt = 0;
      double buff_yaw_raw = 0, buff_yaw_filt = 0;
      double r_yaw_raw = 0, r_yaw_filt = 0;
      int count = 0;
    };
    static InnovStats stats;

    const auto & innov = last_debug_.linear_innovation;  // 5D: R_yaw, R_pitch, R_dist, buff_yaw, buff_pitch
    for (int i = 0; i < 5; ++i) {
      stats.sum[i] += innov[i];
      stats.sum_sq[i] += innov[i] * innov[i];
    }
    // raw = filtered + innovation (innovation = raw - H*x_prior, H*x_prior ≈ filtered for identity H)
    stats.r_dist_raw  += ekf_.x[kRDist] + innov[2];
    stats.r_dist_filt += ekf_.x[kRDist];
    stats.buff_yaw_raw  += ekf_.x[kBuffYaw] + innov[3];
    stats.buff_yaw_filt += ekf_.x[kBuffYaw];
    stats.r_yaw_raw  += ekf_.x[kRYaw] + innov[0];
    stats.r_yaw_filt += ekf_.x[kRYaw];
    stats.count++;

    constexpr int kDiagInterval = 30;  // ~0.5s at 60fps
    if (frame_count % kDiagInterval == 0 && stats.count > 0) {
      const double n = static_cast<double>(stats.count);
      double mean[5], std[5];
      for (int i = 0; i < 5; ++i) {
        mean[i] = stats.sum[i] / n;
        std[i] = std::sqrt(std::max(0.0, stats.sum_sq[i] / n - mean[i] * mean[i]));
      }
      tools::logger()->info(
        "[pose_filter_diag] frames={} dt_avg={:.4f}s | "
        "innov_mean[yaw={:+.4f} pitch={:+.4f} dist={:+.3f} byaw={:+.4f} bpitch={:+.4f}]rad/m | "
        "innov_std[ {:.4f}  {:.4f}  {:.3f}  {:.4f}  {:.4f}] | "
        "R_dist raw/filt={:.3f}/{:.3f}m buff_yaw raw/filt={:.3f}/{:.3f}rad r_yaw raw/filt={:.3f}/{:.3f}rad",
        stats.count, dt,
        mean[0], mean[1], mean[2], mean[3], mean[4],
        std[0], std[1], std[2], std[3], std[4],
        stats.r_dist_raw / n, stats.r_dist_filt / n,
        stats.buff_yaw_raw / n, stats.buff_yaw_filt / n,
        stats.r_yaw_raw / n, stats.r_yaw_filt / n);
      stats = {};
    }
  }

  return snapshot_.valid;
}

// ── 自适应过程噪声：每帧指数衰减 ──
void BuffPoseFilter::decay_adaptive_process_noise()
{
  auto decay_q = [](double & current, double nominal, double d, bool /*boosting*/) {
    if (current <= nominal) return;
    current = std::max(nominal, current * d + nominal * (1.0 - d));
  };

  decay_q(current_q_r_yaw_accel_, config_.q_r_yaw_accel,
          config_.adaptive_q_r_yaw.decay, adaptive_boost_r_yaw_);
  decay_q(current_q_r_pitch_, config_.q_r_pitch,
          config_.adaptive_q_r_pitch.decay, adaptive_boost_r_pitch_);
  decay_q(current_q_r_dist_, config_.q_r_dist,
          config_.adaptive_q_r_dist.decay, adaptive_boost_r_dist_);
  decay_q(current_q_buff_yaw_, config_.q_buff_yaw,
          config_.adaptive_q_buff_yaw.decay, adaptive_boost_buff_yaw_);
  decay_q(current_q_buff_pitch_, config_.q_buff_pitch,
          config_.adaptive_q_buff_pitch.decay, adaptive_boost_buff_pitch_);
}

// ── 自适应过程噪声：检测 innovation 超标 → Q boost ──
void BuffPoseFilter::update_adaptive_process_noise()
{
  const auto & innov = last_debug_.linear_innovation;

  // R_yaw: innov[0], 带 hit 计数防抖
  if (config_.adaptive_q_r_yaw.enabled) {
    const auto & cfg = config_.adaptive_q_r_yaw;
    if (std::abs(innov[0]) > cfg.residual_threshold) {
      adaptive_hit_r_yaw_ = std::min(adaptive_hit_r_yaw_ + 1, 100);
    } else {
      adaptive_hit_r_yaw_ = 0;
      adaptive_boost_r_yaw_ = false;
    }
    if (adaptive_hit_r_yaw_ >= cfg.min_hits) {
      if (!adaptive_boost_r_yaw_) {
        tools::logger()->debug(
          "[pose_filter] adaptive Q r_yaw BOOST: innov={:.4f} > {:.4f} "
          "hits={} q {:.4f}→{:.4f}",
          innov[0], cfg.residual_threshold, adaptive_hit_r_yaw_,
          current_q_r_yaw_accel_, config_.q_r_yaw_accel * cfg.high_factor);
      }
      adaptive_boost_r_yaw_ = true;
      current_q_r_yaw_accel_ = std::max(
        current_q_r_yaw_accel_, config_.q_r_yaw_accel * cfg.high_factor);
    }
  }

  // R_pitch: innov[1]
  if (config_.adaptive_q_r_pitch.enabled) {
    const auto & cfg = config_.adaptive_q_r_pitch;
    if (std::abs(innov[1]) > cfg.residual_threshold) {
      adaptive_hit_r_pitch_ = std::min(adaptive_hit_r_pitch_ + 1, 100);
    } else {
      adaptive_hit_r_pitch_ = 0;
      adaptive_boost_r_pitch_ = false;
    }
    if (adaptive_hit_r_pitch_ >= cfg.min_hits) {
      if (!adaptive_boost_r_pitch_) {
        tools::logger()->debug(
          "[pose_filter] adaptive Q r_pitch BOOST: innov={:.4f} > {:.4f} "
          "hits={} q {:.4f}→{:.4f}",
          innov[1], cfg.residual_threshold, adaptive_hit_r_pitch_,
          current_q_r_pitch_, config_.q_r_pitch * cfg.high_factor);
      }
      adaptive_boost_r_pitch_ = true;
      current_q_r_pitch_ = std::max(
        current_q_r_pitch_, config_.q_r_pitch * cfg.high_factor);
    }
  }

  // R_dist: innov[2]
  if (config_.adaptive_q_r_dist.enabled) {
    const auto & cfg = config_.adaptive_q_r_dist;
    if (std::abs(innov[2]) > cfg.residual_threshold) {
      adaptive_hit_r_dist_ = std::min(adaptive_hit_r_dist_ + 1, 100);
    } else {
      adaptive_hit_r_dist_ = 0;
      adaptive_boost_r_dist_ = false;
    }
    if (adaptive_hit_r_dist_ >= cfg.min_hits) {
      if (!adaptive_boost_r_dist_) {
        tools::logger()->debug(
          "[pose_filter] adaptive Q r_dist BOOST: innov={:.3f} > {:.3f} "
          "hits={} q {:.4f}→{:.4f}",
          innov[2], cfg.residual_threshold, adaptive_hit_r_dist_,
          current_q_r_dist_, config_.q_r_dist * cfg.high_factor);
      }
      adaptive_boost_r_dist_ = true;
      current_q_r_dist_ = std::max(
        current_q_r_dist_, config_.q_r_dist * cfg.high_factor);
    }
  }

  // Buff_yaw: innov[3], min_hits=1 — 单帧触发
  if (config_.adaptive_q_buff_yaw.enabled) {
    const auto & cfg = config_.adaptive_q_buff_yaw;
    if (std::abs(innov[3]) > cfg.residual_threshold) {
      if (!adaptive_boost_buff_yaw_) {
        tools::logger()->debug(
          "[pose_filter] adaptive Q buff_yaw BOOST: innov={:.4f} > {:.4f} "
          "q {:.4f}→{:.4f}",
          innov[3], cfg.residual_threshold,
          current_q_buff_yaw_, config_.q_buff_yaw * cfg.high_factor);
      }
      adaptive_boost_buff_yaw_ = true;
      current_q_buff_yaw_ = std::max(
        current_q_buff_yaw_, config_.q_buff_yaw * cfg.high_factor);
    } else {
      adaptive_boost_buff_yaw_ = false;  // 单帧恢复正常即可
    }
  }

  // Buff_pitch: innov[4], min_hits=2 — blade 耦合导致周期性尖峰
  if (config_.adaptive_q_buff_pitch.enabled) {
    const auto & cfg = config_.adaptive_q_buff_pitch;
    if (std::abs(innov[4]) > cfg.residual_threshold) {
      adaptive_hit_buff_pitch_ = std::min(adaptive_hit_buff_pitch_ + 1, 100);
    } else {
      adaptive_hit_buff_pitch_ = 0;
      adaptive_boost_buff_pitch_ = false;
    }
    if (adaptive_hit_buff_pitch_ >= cfg.min_hits) {
      if (!adaptive_boost_buff_pitch_) {
        tools::logger()->debug(
          "[pose_filter] adaptive Q buff_pitch BOOST: innov={:.4f} > {:.4f} "
          "hits={} q {:.6f}→{:.6f}",
          innov[4], cfg.residual_threshold, adaptive_hit_buff_pitch_,
          current_q_buff_pitch_, config_.q_buff_pitch * cfg.high_factor);
      }
      adaptive_boost_buff_pitch_ = true;
      current_q_buff_pitch_ = std::max(
        current_q_buff_pitch_, config_.q_buff_pitch * cfg.high_factor);
    }
  }
}

// ── NIS 滑动窗口：检测整体发散 ──
void BuffPoseFilter::update_nis_window(double nis, int obs_dim)
{
  if (!config_.nis_detection) return;

  const double threshold = (obs_dim == kJointObsSize)
    ? config_.nis_joint_threshold
    : config_.nis_threshold;
  const int fail = (nis > threshold) ? 1 : 0;
  recent_nis_failures_.push_back(fail);
  while (static_cast<int>(recent_nis_failures_.size()) > config_.nis_window_size) {
    recent_nis_failures_.pop_front();
  }

  const int total = static_cast<int>(recent_nis_failures_.size());
  if (total < config_.nis_window_size) return;  // 窗口未满不检查

  int fail_count = 0;
  for (int v : recent_nis_failures_) fail_count += v;
  const double ratio = static_cast<double>(fail_count) / static_cast<double>(total);
  if (ratio > config_.nis_fail_ratio) {
    tools::logger()->warn(
      "[pose_filter] NIS DIVERGENCE: fail_ratio={:.2f} > {:.2f} "
      "({}/{}), threshold={:.2f} obs_dim={}",
      ratio, config_.nis_fail_ratio, fail_count, total, threshold, obs_dim);
  }
}

}  // namespace auto_buff
