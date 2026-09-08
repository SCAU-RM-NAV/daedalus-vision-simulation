#include "trajectory.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace tools
{
namespace
{
constexpr double STANDARD_G = 9.7833;
constexpr double HERO_G = 9.78;
constexpr double EPS = 1e-6;

std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

void solve_auto_aim(Trajectory & trajectory, const double v0, const double d, const double h)
{
  if (v0 <= EPS || d <= EPS) {
    trajectory.unsolvable = true;
    return;
  }

  auto a = STANDARD_G * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    trajectory.unsolvable = true;
    return;
  }

  trajectory.unsolvable = false;
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  trajectory.pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  trajectory.fly_time = (t_1 < t_2) ? t_1 : t_2;
}

void solve_energy_buff(Trajectory & trajectory, const double v0, const double d, const double h)
{
  if (v0 <= EPS || d <= EPS) {
    trajectory.unsolvable = true;
    return;
  }

  auto a = STANDARD_G * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  if (delta < 0) {
    trajectory.unsolvable = true;
    return;
  }

  trajectory.unsolvable = false;
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  trajectory.pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  trajectory.fly_time = (t_1 < t_2) ? t_1 : t_2;
}

void solve_hero(Trajectory & trajectory, const double v0, const double d, const double h,
  const TrajectoryConfig & config)
{
  auto dist_vertical = h + config.hero_s_bias * std::sin(config.hero_imu_pitch);
  auto dist_horizontal = d - config.hero_s_bias * std::cos(config.hero_imu_pitch);
  auto vertical_tmp = dist_vertical;
  auto rk_iter = std::max(1, config.hero_rk_iter);

  if (v0 <= EPS || dist_horizontal <= EPS) {
    trajectory.unsolvable = true;
    return;
  }

  auto direct_pitch = std::atan(dist_vertical / dist_horizontal);
  trajectory.fly_time = dist_horizontal / (v0 * std::cos(direct_pitch));

  auto pitch_new = direct_pitch;
  for (int i = 0; i < 10; i++) {
    auto y = 0.0;
    auto p = std::tan(pitch_new);
    auto u = v0 / std::sqrt(1 + p * p);
    auto delta_x = dist_horizontal / rk_iter;

    for (int j = 0; j < rk_iter; j++) {
      auto k1_u = -config.hero_air_resistance_k * u * std::sqrt(1 + p * p);
      auto k1_p = -HERO_G / (u * u);
      auto k1_u_sum = u + k1_u * (delta_x / 2);
      auto k1_p_sum = p + k1_p * (delta_x / 2);

      auto k2_u = -config.hero_air_resistance_k * k1_u_sum * std::sqrt(1 + k1_p_sum * k1_p_sum);
      auto k2_p = -HERO_G / (k1_u_sum * k1_u_sum);
      auto k2_u_sum = u + k2_u * (delta_x / 2);
      auto k2_p_sum = p + k2_p * (delta_x / 2);

      auto k3_u = -config.hero_air_resistance_k * k2_u_sum * std::sqrt(1 + k2_p_sum * k2_p_sum);
      auto k3_p = -HERO_G / (k2_u_sum * k2_u_sum);
      auto k3_u_sum = u + k3_u * (delta_x / 2);
      auto k3_p_sum = p + k3_p * (delta_x / 2);

      auto k4_u = -config.hero_air_resistance_k * k3_u_sum * std::sqrt(1 + k3_p_sum * k3_p_sum);
      auto k4_p = -HERO_G / (k3_u_sum * k3_u_sum);

      u += (delta_x / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u);
      p += (delta_x / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p);

      if (!std::isfinite(u) || std::abs(u) <= EPS || !std::isfinite(p)) {
        trajectory.unsolvable = true;
        return;
      }

      y += p * delta_x;
    }

    auto error = dist_vertical - y;
    if (std::abs(error) <= 0.0005) {
      break;
    }

    vertical_tmp += error;
    pitch_new = std::atan(vertical_tmp / dist_horizontal);
    if (!std::isfinite(pitch_new)) {
      trajectory.unsolvable = true;
      return;
    }
  }

  trajectory.unsolvable = false;
  trajectory.pitch = pitch_new;
}
}  // namespace

Trajectory::Trajectory(const double v0, const double d, const double h)
{
  solve_auto_aim(*this, v0, d, h);
}

Trajectory::Trajectory(
  const double v0, const double d, const double h, const TrajectoryConfig & config,
  bool is_energy_buff)
{
  if (config.model == TrajectoryModel::Hero) {
    solve_hero(*this, v0, d, h, config);
  } else if (is_energy_buff) {
    solve_energy_buff(*this, v0, d, h);
  } else {
    solve_auto_aim(*this, v0, d, h);
  }
}

TrajectoryModel trajectory_model_from_string(const std::string & model)
{
  auto value = lower(model);
  if (value == "hero" || value == "1" || value == "42" || value == "bullet_42") {
    return TrajectoryModel::Hero;
  }
  return TrajectoryModel::Standard;
}

double normalize_bullet_speed(double bullet_speed, const TrajectoryConfig & config)
{
  if (!config.use_electrical_control_bullet_speed) {
    return config.default_bullet_speed;
  }

  if (
    !std::isfinite(bullet_speed) || bullet_speed < config.min_bullet_speed ||
    bullet_speed > config.max_bullet_speed) {
    return config.default_bullet_speed;
  }

  return bullet_speed;
}

}  // namespace tools
