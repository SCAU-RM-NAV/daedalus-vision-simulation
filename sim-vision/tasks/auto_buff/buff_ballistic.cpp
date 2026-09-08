#include "buff_ballistic.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>

#include "tasks/auto_aim/trajectory_config.hpp"
#include "tools/logger.hpp"

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

bool finite_vector(const Eigen::Vector3d & value)
{
  return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

double finite_or_default(double value, double fallback)
{
  return std::isfinite(value) && value > 1e-6 ? value : fallback;
}

Eigen::Vector3d leaf_center_at_angle(
  const Eigen::Matrix3d & rotation_world, double delta_angle, double radius)
{
  const Eigen::AngleAxisd delta_rotation(delta_angle, Eigen::Vector3d::UnitX());
  return rotation_world * (delta_rotation * Eigen::Vector3d(0.0, 0.0, radius));
}
}  // namespace

BallisticParams BallisticParams::from_yaml(const std::string & config_path)
{
  BallisticParams params;
  tools::TrajectoryConfig default_trajectory_config;
  default_trajectory_config.min_bullet_speed = 10.0;
  default_trajectory_config.max_bullet_speed = 1e9;
  default_trajectory_config.default_bullet_speed = params.default_bullet_speed;
  params.trajectory_config = default_trajectory_config;

  auto yaml = YAML::LoadFile(config_path);
  auto ballistic = yaml["ballistic"];

  params.trajectory_config =
    auto_aim::load_trajectory_config(yaml, config_path, default_trajectory_config);
  params.default_bullet_speed = params.trajectory_config.default_bullet_speed;

  read_optional(
    yaml, ballistic, "default_bullet_speed", "default_bullet_speed", params.default_bullet_speed);
  read_optional(yaml, ballistic, "energy_yaw_bias", "energy_yaw_bias", params.energy_yaw_bias);
  read_optional(
    yaml, ballistic, "energy_pitch_bias", "energy_pitch_bias", params.energy_pitch_bias);
  read_optional(yaml, ballistic, "buff_yaw_bias", "yaw_bias", params.energy_yaw_bias);
  read_optional(yaml, ballistic, "buff_pitch_bias", "pitch_bias", params.energy_pitch_bias);
  read_optional(yaml, ballistic, "fixed_iterations", "fixed_iterations", params.fixed_iterations);
  read_optional(yaml, ballistic, "target_radius", "target_radius", params.target_radius);
  read_optional(
    yaml, ballistic, "attack_leaf_center", "attack_leaf_center", params.attack_leaf_center);

  if (ballistic) {
    if (ballistic["model"]) {
      params.trajectory_config.model =
        tools::trajectory_model_from_string(ballistic["model"].as<std::string>());
    } else if (ballistic["trajectory_model"]) {
      params.trajectory_config.model =
        tools::trajectory_model_from_string(ballistic["trajectory_model"].as<std::string>());
    }
    read_optional(
      yaml, ballistic, "min_bullet_speed", "min_bullet_speed",
      params.trajectory_config.min_bullet_speed);
    read_optional(
      yaml, ballistic, "max_bullet_speed", "max_bullet_speed",
      params.trajectory_config.max_bullet_speed);
    read_optional(
      yaml, ballistic, "default_bullet_speed", "default_bullet_speed",
      params.trajectory_config.default_bullet_speed);
    read_optional(
      yaml, ballistic, "hero_air_resistance_k", "k",
      params.trajectory_config.hero_air_resistance_k);
    read_optional(yaml, ballistic, "hero_s_bias", "s_bias", params.trajectory_config.hero_s_bias);
    read_optional(
      yaml, ballistic, "hero_imu_pitch", "imu_pitch", params.trajectory_config.hero_imu_pitch);
    read_optional(
      yaml, ballistic, "hero_rk_iter", "rk_iter", params.trajectory_config.hero_rk_iter);
    read_optional(yaml, ballistic, "ballistic_debug", "debug", params.trajectory_config.debug);
    read_optional(
      yaml, ballistic, "ballistic_debug_interval", "debug_interval",
      params.trajectory_config.debug_interval);
  }

  const bool has_group_yaw_bias =
    ballistic && (ballistic["energy_yaw_bias"] || ballistic["yaw_bias"]);
  const bool has_group_pitch_bias =
    ballistic && (ballistic["energy_pitch_bias"] || ballistic["pitch_bias"]);
  if (!has_group_yaw_bias) {
    if (yaml["yaw_offset"]) params.energy_yaw_bias = yaml["yaw_offset"].as<double>() / 57.3;
  }
  if (!has_group_pitch_bias) {
    if (yaml["pitch_offset"]) params.energy_pitch_bias = yaml["pitch_offset"].as<double>() / 57.3;
  }

  params.fixed_iterations = std::max(1, params.fixed_iterations);
  params.target_radius = std::max(0.0, params.target_radius);
  params.trajectory_config.default_bullet_speed = params.default_bullet_speed;
  return params;
}

BuffBallistic::BuffBallistic(const BallisticParams & params) : params_(params) {}

double BuffBallisticTiming::observation_age_ms() const
{
  return std::max(0.0, now_time_abs - observed_time_abs) * 1000.0;
}

double BuffBallisticTiming::delay_time_ms() const
{
  return std::max(0.0, runtime_solve_delay_ms) + std::max(0.0, tuned_bias_time_ms);
}

// 单点弹道解算
BuffBallisticResult BuffBallistic::solve_once(
  const Eigen::Vector3d & target_world, double bullet_speed, double imu_pitch) const
{
  BuffBallisticResult result;
  result.target_world = target_world;
  if (!finite_vector(target_world)) return result;

  const double used_bullet_speed = finite_or_default(bullet_speed, params_.default_bullet_speed);
  const double horizontal_distance = std::hypot(target_world.x(), target_world.y());
  if (!std::isfinite(horizontal_distance) || horizontal_distance <= 1e-6) return result;

  auto trajectory_config = params_.trajectory_config;
  trajectory_config.hero_imu_pitch = imu_pitch;
  tools::Trajectory trajectory(
    used_bullet_speed, horizontal_distance, target_world.z(), trajectory_config, true);
  if (
    trajectory.unsolvable || !std::isfinite(trajectory.fly_time) ||
    !std::isfinite(trajectory.pitch)) {
    return result;
  }

  result.valid = true;
  result.yaw = std::atan2(target_world.y(), target_world.x()) + params_.energy_yaw_bias;
  result.pitch = trajectory.pitch + params_.energy_pitch_bias;
  result.fly_time = trajectory.fly_time;
  return result;
}

// 主函数：预测+弹道迭代（不动点迭代）
BuffBallisticResult BuffBallistic::solve_with_iteration(
  const Eigen::Vector3d & rune_center_world, const Eigen::Matrix3d & rotation_world,
  double target_angle_at_detect, const BuffBallisticTiming & timing, double bullet_speed,
  const std::function<double(double)> & predict_angle_abs) const
{
  tools::logger()->info("speed:{}",bullet_speed);

  BuffBallisticResult result;
  result.observed_time_abs = timing.observed_time_abs;
  result.now_time_abs = timing.now_time_abs;
  result.observation_age_ms = timing.observation_age_ms();
  result.runtime_solve_delay_ms = std::max(0.0, timing.runtime_solve_delay_ms);
  result.tuned_bias_time_ms = std::max(0.0, timing.tuned_bias_time_ms);
  result.delay_time_ms = timing.delay_time_ms();
  if (!finite_vector(rune_center_world) || !rotation_world.allFinite() || !predict_angle_abs) {
    return result;
  }

  const double delay_time = result.delay_time_ms * 1e-3;
  double fly_time = 0.0;
  double prev_fly_time = 0.0;
  for (int iter = 0; iter < params_.fixed_iterations; ++iter) {
    prev_fly_time = fly_time;
    const double predict_time_abs = timing.now_time_abs + delay_time + fly_time;
    const double predicted_angle = predict_angle_abs(predict_time_abs);
    if (!std::isfinite(predicted_angle)) return {};

    Eigen::Vector3d target_world = rune_center_world;
    if (params_.attack_leaf_center) {
      const double delta_angle = predicted_angle - target_angle_at_detect;
      target_world += leaf_center_at_angle(rotation_world, delta_angle, params_.target_radius);
    }

    auto once = solve_once(target_world, bullet_speed);
    once.predict_time_abs = predict_time_abs;
    once.predicted_angle = predicted_angle;
    once.observed_time_abs = result.observed_time_abs;
    once.now_time_abs = result.now_time_abs;
    once.observation_age_ms = result.observation_age_ms;
    once.runtime_solve_delay_ms = result.runtime_solve_delay_ms;
    once.tuned_bias_time_ms = result.tuned_bias_time_ms;
    once.delay_time_ms = result.delay_time_ms;
    once.iterations = iter + 1;
    if (!once.valid) return once;

    result = once;
    fly_time = once.fly_time;

    if (std::abs(fly_time - prev_fly_time) < 1e-6) break; // 收敛
  }

  if (params_.trajectory_config.debug) {
    tools::logger()->debug(
      "[BuffBallistic] target=({:.3f},{:.3f},{:.3f}) yaw={:.4f} pitch={:.4f} fly={:.4f}s "
      "predict_t={:.4f}s observed_t={:.4f}s now_t={:.4f}s observation_age={:.3f}ms "
      "runtime_solve={:.3f}ms tuned_bias={:.3f}ms total_delay={:.3f}ms iterations={}",
      result.target_world.x(), result.target_world.y(), result.target_world.z(), result.yaw,
      result.pitch, result.fly_time, result.predict_time_abs, result.observed_time_abs,
      result.now_time_abs, result.observation_age_ms, result.runtime_solve_delay_ms,
      result.tuned_bias_time_ms, result.delay_time_ms, result.iterations);
  }
  return result;
}

}  // namespace auto_buff
