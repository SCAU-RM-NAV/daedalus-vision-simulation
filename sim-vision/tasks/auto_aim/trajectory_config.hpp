#ifndef AUTO_AIM__TRAJECTORY_CONFIG_HPP
#define AUTO_AIM__TRAJECTORY_CONFIG_HPP

#include <algorithm>
#include <cctype>
#include <string>

#include <yaml-cpp/yaml.h>

#include "tools/trajectory.hpp"

namespace auto_aim
{
namespace trajectory_config_detail
{
inline std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

inline bool contains_hero(const std::string & value) { return lower(value).find("hero") != std::string::npos; }

template <typename T>
inline bool read_optional(const YAML::Node & root, const YAML::Node & group, const std::string & flat_key,
  const std::string & group_key, T & value)
{
  if (root[flat_key]) {
    value = root[flat_key].as<T>();
    return true;
  }

  if (group && group[group_key]) {
    value = group[group_key].as<T>();
    return true;
  }

  return false;
}

inline std::string read_model(const YAML::Node & root, const YAML::Node & trajectory, const std::string & config_path)
{
  std::string model;
  if (read_optional(root, trajectory, "trajectory_model", "model", model)) return model;
  if (root["ballistic_type"]) return root["ballistic_type"].as<std::string>();
  if (root["robot_type"]) return root["robot_type"].as<std::string>();
  if (root["robot"] && root["robot"].IsScalar()) return root["robot"].as<std::string>();

  if (root["bullet_type"]) {
    auto bullet_type = root["bullet_type"].as<std::string>();
    if (bullet_type == "1" || contains_hero(bullet_type) || lower(bullet_type).find("42") != std::string::npos) {
      return "hero";
    }
  }

  if (trajectory && trajectory["bullet_type"]) {
    auto bullet_type = trajectory["bullet_type"].as<std::string>();
    if (bullet_type == "1" || contains_hero(bullet_type) || lower(bullet_type).find("42") != std::string::npos) {
      return "hero";
    }
  }

  return contains_hero(config_path) ? "hero" : "standard";
}
}  // namespace trajectory_config_detail

inline tools::TrajectoryConfig load_trajectory_config(
  const YAML::Node & yaml, const std::string & config_path, tools::TrajectoryConfig config = {})
{
  auto trajectory = yaml["trajectory"];

  auto model = trajectory_config_detail::read_model(yaml, trajectory, config_path);
  config.model = tools::trajectory_model_from_string(model);
  if (config.model == tools::TrajectoryModel::Hero) {
    config.min_bullet_speed = 9.5;
    config.max_bullet_speed = 13.5;
    config.default_bullet_speed = 11.5;
    trajectory_config_detail::read_optional(
      yaml, trajectory, "use_electrical_control_bullet_speed",
      "use_electrical_control_bullet_speed", config.use_electrical_control_bullet_speed);
  }

  trajectory_config_detail::read_optional(
    yaml, trajectory, "min_bullet_speed", "min_bullet_speed", config.min_bullet_speed);
  trajectory_config_detail::read_optional(
    yaml, trajectory, "max_bullet_speed", "max_bullet_speed", config.max_bullet_speed);
  if (
    !trajectory_config_detail::read_optional(
      yaml, trajectory, "default_bullet_speed", "default_bullet_speed",
      config.default_bullet_speed)) {
    trajectory_config_detail::read_optional(
      yaml, trajectory, "current_v", "current_v", config.default_bullet_speed);
  }

  trajectory_config_detail::read_optional(
    yaml, trajectory, "hero_air_resistance_k", "k", config.hero_air_resistance_k);
  trajectory_config_detail::read_optional(yaml, trajectory, "hero_s_bias", "s_bias", config.hero_s_bias);
  trajectory_config_detail::read_optional(
    yaml, trajectory, "hero_imu_pitch", "imu_pitch", config.hero_imu_pitch);
  if (
    !trajectory_config_detail::read_optional(
      yaml, trajectory, "hero_rk_iter", "rk_iter", config.hero_rk_iter)) {
    trajectory_config_detail::read_optional(yaml, trajectory, "R_K_iter", "R_K_iter", config.hero_rk_iter);
  }
  trajectory_config_detail::read_optional(yaml, trajectory, "ballistic_debug", "debug", config.debug);
  trajectory_config_detail::read_optional(
    yaml, trajectory, "ballistic_debug_interval", "debug_interval", config.debug_interval);

  return config;
}

}  // namespace auto_aim

#endif  // AUTO_AIM__TRAJECTORY_CONFIG_HPP
