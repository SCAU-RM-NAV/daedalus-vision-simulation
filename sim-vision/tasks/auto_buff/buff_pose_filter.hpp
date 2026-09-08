#ifndef AUTO_BUFF__BUFF_POSE_FILTER_HPP
#define AUTO_BUFF__BUFF_POSE_FILTER_HPP

#include <Eigen/Dense>
#include <deque>
#include <limits>
#include <string>

#include "buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_buff
{

struct PoseFilterConfig
{
  bool enabled = true;
  double max_dt_sec = 0.25;

  double p0_r_yaw = 10.0;
  double p0_r_v_yaw = 10.0;
  double p0_r_pitch = 10.0;
  double p0_r_dist = 10.0;
  double p0_buff_yaw = 10.0;
  double p0_buff_pitch = 0.1;   // 符仰角初始协方差（安装后不变，初始值很小）

  double q_r_yaw_accel = 0.001;
  double q_r_pitch = 0.001;
  double q_r_dist = 0.01;
  double q_buff_yaw = 0.01;
  double q_buff_pitch = 0.0001; // 符仰角过程噪声（几乎不变）

  double r_r_yaw = 0.01;
  double r_r_pitch = 0.01;
  double r_r_dist = 0.5;
  double r_buff_yaw = 0.1;
  double r_buff_pitch = 0.01;   // 符仰角观测噪声

  double r_blade_yaw = 0.01;
  double r_blade_pitch = 0.01;
  double r_blade_dist = 0.5;

  bool use_blade_obs = true;  // false → 只用 R 标+符朝向观测，不引入 blade 耦合

  // ── 自适应过程噪声（每通道独立） ──
  struct AdaptiveQChannel
  {
    bool enabled = false;
    double high_factor = 10.0;      // Q 提升倍数
    double decay = 0.85;            // 指数衰减率
    double residual_threshold = 0.0; // innovation 触发阈值
    int min_hits = 3;               // 连续触发帧数（防抖）
  };

  AdaptiveQChannel adaptive_q_r_yaw    = {false, 10.0, 0.85, 0.02, 3};
  AdaptiveQChannel adaptive_q_r_pitch  = {false, 10.0, 0.85, 0.02, 3};
  AdaptiveQChannel adaptive_q_r_dist   = {false, 10.0, 0.85, 0.3,  3};
  AdaptiveQChannel adaptive_q_buff_yaw   = {false, 50.0, 0.85, 0.05, 1}; // 单帧触发
  AdaptiveQChannel adaptive_q_buff_pitch = {false, 30.0, 0.85, 0.10, 2}; // blade 耦合导致周期性尖峰

  // ── Blade 软门控 ──
  bool blade_soft_gate = true;  // true=连续衰减, false=硬截断(旧行为)

  // ── NIS 发散检测 ──
  bool nis_detection = false;
  int nis_window_size = 50;
  double nis_fail_ratio = 0.5;       // 窗口内 NIS 超标比例阈值
  double nis_threshold = 11.07;      // χ²(5) 95% 置信度（纯线性模式）
  double nis_joint_threshold = 15.51; // χ²(8) 95% 置信度（联合模式）

  static PoseFilterConfig from_yaml(const std::string & config_path);
};

struct BuffPoseSnapshot // 滤波后的结果（给弹道解算用）
{
  bool valid = false;
  Eigen::Vector3d filtered_ypd_world = Eigen::Vector3d::Zero(); // R标在世界球坐标
  Eigen::Vector3d filtered_xyz_world = Eigen::Vector3d::Zero(); // R标在世界直角坐标
  double filtered_buff_yaw = 0.0; // 符yaw
  Eigen::Matrix3d filtered_rotation_world = Eigen::Matrix3d::Identity(); // buff坐标系到世界坐标系的旋转
  Eigen::Vector3d filtered_target_center_world = Eigen::Vector3d::Zero(); // 扇叶中心世界坐标
  double observed_time_abs = 0.0;
};

enum class PoseFilterUpdateKind
{
  NONE,
  DISABLED,
  INVALID_INPUT,
  INITIALIZED,
  REINITIALIZED_TIME,
  UPDATED,
};

struct BuffPoseFilterDebug
{
  PoseFilterUpdateKind update_kind = PoseFilterUpdateKind::NONE;
  bool enabled = false;
  bool input_valid = false;
  double dt = std::numeric_limits<double>::quiet_NaN();
  Eigen::Matrix<double, 6, 1> state_before =
    Eigen::Matrix<double, 6, 1>::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix<double, 6, 1> state_predicted =
    Eigen::Matrix<double, 6, 1>::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix<double, 6, 1> state_after =
    Eigen::Matrix<double, 6, 1>::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix<double, 5, 1> linear_innovation =
    Eigen::Matrix<double, 5, 1>::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d blade_innovation =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d raw_minus_filtered_r_xyz =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d raw_minus_filtered_target_xyz =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  double raw_to_filtered_rotation_distance = std::numeric_limits<double>::quiet_NaN();
};

class BuffPoseFilter
{
public:
  static constexpr int kStateSize = 6; // x = [R_yaw, R_v_yaw, R_pitch, R_dist, buff_yaw, buff_pitch]
  static constexpr int kLinearObsSize = 5; // z1 = [R_yaw, R_pitch, R_dist, buff_yaw, buff_pitch]
  static constexpr int kBladeObsSize = 3;  // z2 = [blade_yaw, blade_pitch, blade_dist]
  static constexpr int kJointObsSize = kLinearObsSize + kBladeObsSize; // z = [z1; z2]

  enum StateIndex
  {
    kRYaw = 0,
    kRvyaw = 1,
    kRPitch = 2,
    kRDist = 3,
    kBuffYaw = 4,
    kBuffPitch = 5 
  };

  explicit BuffPoseFilter(const PoseFilterConfig & config = {}, double target_radius = 0.700);

  bool update(const PowerRune & rune, double observed_time_abs);
  void reset();

  bool valid() const { return snapshot_.valid; }
  const BuffPoseSnapshot & snapshot() const { return snapshot_; }
  const BuffPoseFilterDebug & debug_snapshot() const { return last_debug_; }
  const PoseFilterConfig & config() const { return config_; }

private:
  void initialize(const PowerRune & rune, double observed_time_abs);
  bool finite_pose_input(const PowerRune & rune) const;
  void rebuild_snapshot(double raw_roll, double observed_time_abs);
  void record_output_delta(const PowerRune & rune);
  Eigen::Vector3d predict_blade_ypd(
    const Eigen::VectorXd & state, double raw_roll) const;
  Eigen::Matrix<double, kBladeObsSize, kStateSize> blade_observation_jacobian(
    const Eigen::VectorXd & state, double raw_roll) const;
  Eigen::Vector3d rotated_leaf_offset(double buff_yaw, double buff_pitch, double raw_roll) const;
  Eigen::Vector3d rotated_leaf_offset_dyaw(
    double buff_yaw, double buff_pitch, double raw_roll) const;
  Eigen::Vector3d rotated_leaf_offset_dpitch(
    double buff_yaw, double buff_pitch, double raw_roll) const;

  // 自适应过程噪声
  void update_adaptive_process_noise();
  void decay_adaptive_process_noise();
  void update_nis_window(double nis, int obs_dim);

  PoseFilterConfig config_;
  double target_radius_ = 0.700;
  bool initialized_ = false;
  double last_time_ = 0.0;
  tools::ExtendedKalmanFilter ekf_;
  BuffPoseSnapshot snapshot_;
  BuffPoseFilterDebug last_debug_;

  // 动态 Q 值（运行时随自适应变化）
  double current_q_r_yaw_accel_;
  double current_q_r_pitch_;
  double current_q_r_dist_;
  double current_q_buff_yaw_;
  double current_q_buff_pitch_;

  // 每通道独立触发状态
  int adaptive_hit_r_yaw_ = 0;
  int adaptive_hit_r_pitch_ = 0;
  int adaptive_hit_r_dist_ = 0;
  int adaptive_hit_buff_pitch_ = 0;
  bool adaptive_boost_r_yaw_ = false;
  bool adaptive_boost_r_pitch_ = false;
  bool adaptive_boost_r_dist_ = false;
  bool adaptive_boost_buff_yaw_ = false;
  bool adaptive_boost_buff_pitch_ = false;

  // NIS 滑动窗口
  std::deque<int> recent_nis_failures_;

  Eigen::Matrix<double, kStateSize, kStateSize> F_ =
    Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
  Eigen::Matrix<double, kStateSize, kStateSize> Q_ =
    Eigen::Matrix<double, kStateSize, kStateSize>::Zero();
  Eigen::Matrix<double, kLinearObsSize, kStateSize> H1_ =
    Eigen::Matrix<double, kLinearObsSize, kStateSize>::Zero();
  Eigen::Matrix<double, kLinearObsSize, kLinearObsSize> R1_ =
    Eigen::Matrix<double, kLinearObsSize, kLinearObsSize>::Zero();
  Eigen::Matrix<double, kBladeObsSize, kBladeObsSize> R2_ =
    Eigen::Matrix<double, kBladeObsSize, kBladeObsSize>::Zero();
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__BUFF_POSE_FILTER_HPP
