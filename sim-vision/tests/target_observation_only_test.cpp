#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tools/math_tools.hpp"

namespace
{
auto_aim::Armor make_armor(double z)
{
  std::vector<cv::Point2f> points{{0, 0}, {20, 0}, {20, 10}, {0, 10}};
  auto_aim::Armor armor(7, 1.0F, cv::Rect(0, 0, 20, 10), points);
  armor.name = auto_aim::ArmorName::one;
  armor.type = auto_aim::ArmorType::small;
  armor.xyz_in_world = Eigen::Vector3d(5.0, 0.0, z);
  armor.xyz_in_gimbal = armor.xyz_in_world;
  armor.pnp_xyz_in_camera = armor.xyz_in_world;
  armor.ypr_in_world = Eigen::Vector3d::Zero();
  armor.ypr_in_gimbal = armor.ypr_in_world;
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  return armor;
}

auto_aim::Armor make_outpost_armor(int id, double z)
{
  constexpr double radius = 0.2765;
  constexpr double center_x = 5.0 + radius;
  const auto angle = id * 2.0 * CV_PI / 3.0;
  auto armor = make_armor(z);
  armor.name = auto_aim::ArmorName::outpost;
  armor.xyz_in_world = Eigen::Vector3d(
    center_x - radius * std::cos(angle), -radius * std::sin(angle), z);
  armor.xyz_in_gimbal = armor.xyz_in_world;
  armor.pnp_xyz_in_camera = armor.xyz_in_world;
  armor.ypr_in_world = Eigen::Vector3d(angle, 0.0, 0.0);
  armor.ypr_in_gimbal = armor.ypr_in_world;
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  return armor;
}
}  // namespace

int main()
{
  auto armor = make_armor(0.10);
  auto_aim::EkfNoiseConfig config;
  config.z_observation_only_mode = auto_aim::ZObservationOnlyMode::enabled;
  config.adaptive_v1 = false;
  config.adaptive_v2 = false;

  Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
  auto_aim::Target target(
    armor, std::chrono::steady_clock::now(), 0.2, 4, P0_dig, config);

  target.predict(0.05);
  armor = make_armor(0.35);
  target.update(armor);

  if (std::abs(target.last_observed_z() - 0.35) >= 1e-12) {
    std::cerr << "latest observed Z was not retained\n";
    return 1;
  }
  if (std::abs(target.armor_xyza_list().front().z() - 0.35) >= 1e-9) {
    std::cerr << "observed Z was not applied to the matched plate\n";
    return 1;
  }
  if (std::abs(target.ekf_x()[5]) >= 1e-12) {
    std::cerr << "vertical velocity was not cleared\n";
    return 1;
  }

  // Planner-style future and rewind predictions must not move Z from the latest observation.
  target.predict(0.30);
  target.predict(-0.50);
  target.predict(0.80);

  if (std::abs(target.armor_xyza_list().front().z() - 0.35) >= 1e-9) {
    std::cerr << "planner-style predictions changed observation-only Z\n";
    return 1;
  }
  if (std::abs(target.ekf_x()[5]) >= 1e-12) {
    std::cerr << "planner-style predictions restored vertical velocity\n";
    return 1;
  }

  auto_aim::EkfNoiseConfig auto_config;
  auto_config.z_observation_only_mode = auto_aim::ZObservationOnlyMode::automatic;
  auto_config.z_observation_auto_window_frames = 3;
  auto_config.z_observation_auto_range_threshold = 0.5;

  armor = make_armor(0.10);
  auto_aim::Target auto_target(
    armor, std::chrono::steady_clock::now(), 0.2, 4, P0_dig, auto_config);
  if (auto_target.z_observation_only()) {
    std::cerr << "auto mode enabled before the Z range exceeded its threshold\n";
    return 1;
  }

  armor = make_armor(0.35);
  auto_target.update(armor);
  if (auto_target.z_observation_only()) {
    std::cerr << "auto mode enabled below the Z range threshold\n";
    return 1;
  }

  armor = make_armor(0.70);
  auto_target.update(armor);
  if (!auto_target.z_observation_only()) {
    std::cerr << "auto mode did not enable after the Z range exceeded its threshold\n";
    return 1;
  }

  auto_target.predict(0.30);
  if (std::abs(auto_target.armor_xyza_list().front().z() - 0.70) >= 1e-9) {
    std::cerr << "auto mode did not hold the latest observed Z\n";
    return 1;
  }

  // The 0.10 m sample leaves the three-frame window, so the remaining range is only 0.35 m.
  armor = make_armor(0.70);
  auto_target.update(armor);
  if (auto_target.z_observation_only()) {
    std::cerr << "auto mode did not use a bounded recent Z window\n";
    return 1;
  }

  auto_target.reset_z_observation_history();
  if (auto_target.z_observation_only() || std::abs(auto_target.last_observed_z()) >= 1e-12) {
    std::cerr << "auto mode was not reset after target loss\n";
    return 1;
  }

  armor = make_armor(0.70);
  auto_target.update(armor);
  if (auto_target.z_observation_only()) {
    std::cerr << "auto mode reused Z observations from before target loss\n";
    return 1;
  }

  auto_aim::EkfNoiseConfig disabled_config;
  disabled_config.z_observation_only_mode = auto_aim::ZObservationOnlyMode::disabled;
  armor = make_armor(0.10);
  auto_aim::Target disabled_target(
    armor, std::chrono::steady_clock::now(), 0.2, 4, P0_dig, disabled_config);
  armor = make_armor(0.70);
  disabled_target.update(armor);
  if (disabled_target.z_observation_only()) {
    std::cerr << "false mode incorrectly enabled observation-only Z\n";
    return 1;
  }

  auto outpost_armor = make_armor(0.20);
  outpost_armor.name = auto_aim::ArmorName::outpost;
  Eigen::VectorXd outpost_P0{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 1, 1}};
  auto_aim::Target outpost(
    outpost_armor, std::chrono::steady_clock::now(), 0.2765, 3, outpost_P0, config);
  if (outpost.z_observation_only()) {
    std::cerr << "outpost incorrectly enabled observation-only Z\n";
    return 1;
  }

  outpost.update(make_outpost_armor(1, 0.266));
  outpost.update(make_outpost_armor(2, 0.369));
  const auto outpost_armors = outpost.armor_xyza_list();
  const auto [minimum_z, maximum_z] = std::minmax_element(
    outpost_armors.begin(), outpost_armors.end(),
    [](const Eigen::Vector4d & a, const Eigen::Vector4d & b) { return a.z() < b.z(); });

  auto_aim::Planner planner("configs/sentry.yaml");
  for (int i = 0; i < 30; ++i) {
    const auto plan = planner.plan(outpost, 24.0);
    if (!plan.control || !std::isfinite(planner.debug_xyza.z()) ||
        planner.debug_xyza.z() < minimum_z->z() - 0.05 ||
        planner.debug_xyza.z() > maximum_z->z() + 0.05) {
      std::cerr << "outpost planner aim Z left the modeled armor-height range\n";
      return 1;
    }
  }
  return 0;
}
