#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <deque>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tracking_info.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

enum class ZObservationOnlyMode
{
  disabled,
  enabled,
  automatic,
};

struct EkfNoiseConfig
{
  double normal_v1 = 100.0;
  double normal_v2 = 400.0;
  double outpost_v1 = 10.0;
  double outpost_v2 = 0.1;
  bool adaptive_v1 = false;
  double adaptive_v1_high = 500.0;
  double adaptive_v1_decay = 0.85;
  double adaptive_v1_residual_yaw = 0.015;
  double adaptive_v1_residual_pitch = 0.015;
  double adaptive_v1_residual_distance = 0.25;
  bool adaptive_v2 = false;
  double adaptive_v2_high = 1500.0;
  double adaptive_v2_decay = 0.85;
  double adaptive_v2_residual_angle = 0.2;
  double adaptive_v2_nis = 0.711;
  ZObservationOnlyMode z_observation_only_mode = ZObservationOnlyMode::disabled;
  int z_observation_auto_window_frames = 150;
  double z_observation_auto_range_threshold = 0.5;
};

class Target
{
public:
  ArmorName name = ArmorName::not_armor;
  ArmorType armor_type = ArmorType::small;
  ArmorPriority priority = ArmorPriority::fifth;
  bool jumped = false;
  int last_id = 0;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, const EkfNoiseConfig & ekf_noise_config = {});
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor, bool record_z_observation = true);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;
  const Eigen::Vector3d & last_armor_pnp_xyz_in_camera() const;
  double last_observed_z() const;
  bool z_observation_only() const;
  void reset_z_observation_history();
  const TrackingInfo & tracking_info() const;
  void set_tracking_info(const TrackingInfo & tracking_info);

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;
  std::array<bool, 3> outpost_height_initialized_{false, false, false};

  tools::ExtendedKalmanFilter ekf_;
  Eigen::Vector3d last_armor_pnp_xyz_in_camera_ = Eigen::Vector3d::Constant(
    std::numeric_limits<double>::quiet_NaN());
  double last_observed_z_ = std::numeric_limits<double>::quiet_NaN();
  double observed_base_z_ = std::numeric_limits<double>::quiet_NaN();
  std::deque<double> z_observation_history_;
  bool z_observation_only_active_ = false;
  std::chrono::steady_clock::time_point t_;
  TrackingInfo tracking_info_;
  EkfNoiseConfig ekf_noise_config_;
  double current_v1_ = 100.0;
  double current_v2_ = 400.0;
  bool adaptive_v1_boosting_ = false;
  bool adaptive_v2_boosting_ = false;
  int adaptive_v1_hit_count_ = 0;

  void update_ypda(
    const Armor & armor, int id, bool record_z_observation);  // yaw pitch distance angle
  void apply_z_observation(const Armor & armor, int id, bool record_z_observation);
  void update_z_observation_only_state(double observed_z);
  bool is_outpost() const;
  void init_outpost_height(const Armor & armor, int id);
  void update_adaptive_process_noise();
  void decay_adaptive_process_noise();
  void update_ekf_debug_data(double v1, double v2, double v1_boost, double v2_boost);

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
