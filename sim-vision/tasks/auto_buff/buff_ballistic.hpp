#ifndef AUTO_BUFF__BUFF_BALLISTIC_HPP
#define AUTO_BUFF__BUFF_BALLISTIC_HPP

#include <Eigen/Dense>
#include <functional>
#include <limits>
#include <string>

#include "tools/trajectory.hpp"

namespace auto_buff
{

enum class BuffAimSource
{
  NONE,
  OBSERVED_LEAF,
  ANONYMOUS_LEAF,
  R_CENTER
};

struct BallisticParams
{
  double default_bullet_speed = 24.0;
  double energy_yaw_bias = 0.0;
  double energy_pitch_bias = 0.0;
  int fixed_iterations = 3;
  double target_radius = 0.700;
  bool attack_leaf_center = true;
  tools::TrajectoryConfig trajectory_config;

  static BallisticParams from_yaml(const std::string & config_path);
};

struct BuffBallisticResult
{
  bool valid = false;
  bool used_filtered_pose = false;
  BuffAimSource aim_source = BuffAimSource::NONE;
  double yaw = 0.0;
  double pitch = 0.0;  // 这里定义是抬头为正，而在后面要发给电控时会乘以负号，因为电控定义是抬头为负
  double fly_time = 0.0;
  double predict_time_abs = 0.0;
  double predicted_angle = std::numeric_limits<double>::quiet_NaN();
  double observed_time_abs = 0.0;
  double now_time_abs = 0.0;
  double observation_age_ms = 0.0;
  double runtime_solve_delay_ms = 0.0;
  double tuned_bias_time_ms = 0.0;
  double delay_time_ms = 0.0;
  int iterations = 0;
  int attack_leaf_id = -1;    // 实际击打扇叶id，仅用于调试/可视化/录制
  int observed_leaf_id = -1;  // PnP建模观测扇叶id，仅用于调试/可视化/录制
  double selection_cost = std::numeric_limits<double>::infinity();  // 击打片选择代价
  Eigen::Vector3d target_world = Eigen::Vector3d::Zero();
};

struct BuffBallisticTiming
{
  double observed_time_abs = 0.0;
  double now_time_abs = 0.0;
  double runtime_solve_delay_ms = 0.0;
  double tuned_bias_time_ms = 0.0;

  double observation_age_ms() const;
  double delay_time_ms() const;
};

class BuffBallistic
{
public:
  explicit BuffBallistic(const BallisticParams & params = {});

  BuffBallisticResult solve_once(
    const Eigen::Vector3d & target_world, double bullet_speed, double imu_pitch = 0.0) const;

  BuffBallisticResult solve_with_iteration(
    const Eigen::Vector3d & rune_center_world, const Eigen::Matrix3d & rotation_world,
    double target_angle_at_detect, const BuffBallisticTiming & timing, double bullet_speed,
    const std::function<double(double)> & predict_angle_abs) const;

private:
  BallisticParams params_;
};

}  // namespace auto_buff

#endif  // AUTO_BUFF__BUFF_BALLISTIC_HPP
