#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <list>
#include <optional>

#include "tasks/auto_aim/fire_gate.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed, double imu_pitch = 0.0);
  Plan plan(std::optional<Target> target, double bullet_speed, double imu_pitch = 0.0);
  const FireGateDebug & fire_gate_debug() const;

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double comming_angle_, leaving_angle_;
  double outpost_comming_angle_, outpost_leaving_angle_;
  double low_speed_delay_time_, high_speed_delay_time_, outpost_delay_time_, decision_speed_;
  double armor_lock_max_w_;
  double armor_lock_angle_;
  bool aim_center_enable_ = true;
  bool aim_center_near_spin_enable_ = true;
  bool aim_center_spin_enable_ = true;
  double aim_center_near_spin_w_ = 5.5;
  double aim_center_near_distance_ = 2.5;
  double aim_center_spin_w_ = 5.0;
  double aim_center_shoot_yaw_ = 6.0 / 57.3;
  double aim_center_fire_lead_time_ = 0.03;
  tools::TrajectoryConfig trajectory_config_;
  int aim_center_enter_step_ = 50;
  int aim_center_exit_step_ = 30;
  int aim_center_hold_threshold_ = 100;
  int aim_center_hold_max_ = 400;
  int lock_id_ = -1;
  int current_aim_id_ = -1;
  bool aim_center_ = false;
  bool final_aim_center_ = false;
  bool center_lock_fire_ready_ = true;
  double center_lock_aim_yaw_ = 0.0;
  int aim_center_last_time_ = 0;
  int center_lock_armor_id_ = -1;
  int center_lock_pitch_armor_id_ = -1;
  FireGate fire_gate_;
  std::optional<Target> previous_fire_gate_target_;
  uint64_t previous_fire_gate_epoch_ = 0;
  uint64_t previous_fire_gate_observation_seq_ = 0;
  bool previous_fire_gate_aim_center_ = false;
  double last_plan_fly_time_ = 0.0;
  bool has_logged_fire_gate_state_ = false;
  FireGateState last_logged_fire_gate_state_ = FireGateState::disabled;

  struct CenterLockArmorSelection
  {
    int pitch_id = -1;
    int fire_id = -1;
    double pitch_yaw_error = 0.0;
    double fire_yaw_error = 0.0;
    double fire_fly_time = 0.0;
    double fire_predict_time = 0.0;
    double fire_dist = 0.0;
  };

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  Plan plan_impl(Target target, double bullet_speed, double imu_pitch);
  std::optional<double> update_fire_gate_prediction(
    const Target & observation_target, double bullet_speed, double imu_pitch,
    std::chrono::steady_clock::time_point now, double delay_time, bool plan_valid);
  std::optional<Eigen::Vector2d> fire_gate_aim(
    const Target & target, int armor_id, double bullet_speed,
    const tools::TrajectoryConfig & trajectory_config) const;
  void log_fire_gate_state();
  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  bool aimed_armor_in_fire_window(const Target & target) const;
  Eigen::Matrix<double, 2, 1> aim(
    const Target & target, double bullet_speed, const tools::TrajectoryConfig & trajectory_config);
  void reset_aim_center();
  void update_aim_center(const Target & target);
  int select_center_lock_armor_id(const Target & target, double fire_yaw_window) const;
  int select_center_lock_pitch_armor_id(const Target & target) const;
  CenterLockArmorSelection select_center_lock_armors_by_fly_time(
    const Target & target, double bullet_speed, const tools::TrajectoryConfig & trajectory_config,
    double fire_yaw_window) const;
  double center_lock_radius(const Target & target) const;
  double center_lock_flight_dist(const Target & target) const;
  Eigen::Vector4d center_aim_xyza(const Target & target) const;
  void update_aim_lock(const Target & target);
  Eigen::Vector4d select_aim_xyza(const Target & target) const;
  Trajectory get_trajectory(
    Target & target, double yaw0, double bullet_speed,
    const tools::TrajectoryConfig & trajectory_config);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
