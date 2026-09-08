#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

#include <string>

namespace tools
{
enum class TrajectoryModel
{
  Standard,
  Hero
};

struct TrajectoryConfig
{
  TrajectoryModel model = TrajectoryModel::Standard;
  double min_bullet_speed = 10.0;
  double max_bullet_speed = 25.0;
  double default_bullet_speed = 22.0;
  bool use_electrical_control_bullet_speed = true;
  double hero_air_resistance_k = 0.000556;
  double hero_s_bias = 0.2;
  double hero_imu_pitch = 0.0;
  int hero_rk_iter = 60;
  bool debug = false;
  double debug_interval = 0.5;
};

struct Trajectory
{
  bool unsolvable;
  double fly_time;
  double pitch;  // 抬头为正

  // 不考虑空气阻力
  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m
  Trajectory(const double v0, const double d, const double h);
  Trajectory(
    const double v0, const double d, const double h, const TrajectoryConfig & config,
    bool is_energy_buff = false);
};

TrajectoryModel trajectory_model_from_string(const std::string & model);
double normalize_bullet_speed(double bullet_speed, const TrajectoryConfig & config);

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
