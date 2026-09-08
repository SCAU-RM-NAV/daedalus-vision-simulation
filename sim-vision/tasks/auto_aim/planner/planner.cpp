#include "planner.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"
#include "tasks/auto_aim/trajectory_config.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
FireGateConfig load_fire_gate_config(const std::string & config_path)
{
  FireGateConfig result;
  const auto yaml = tools::load(config_path);
  const auto config = yaml["fire_gate"];
  if (!config) return result;

  if (config["mode"]) {
    const auto mode = config["mode"].as<std::string>();
    if (mode == "off")
      result.mode = FireGateMode::off;
    else if (mode == "shadow")
      result.mode = FireGateMode::shadow;
    else if (mode == "enforce")
      result.mode = FireGateMode::enforce;
    else
      throw std::runtime_error("fire_gate.mode must be off, shadow or enforce");
  }
  if (config["warmup_time"]) result.warmup_time = config["warmup_time"].as<double>();
  if (config["stable_observations"])
    result.stable_observations = config["stable_observations"].as<int>();
  if (config["max_prediction_change_deg"])
    result.max_prediction_change_deg = config["max_prediction_change_deg"].as<double>();

  tools::logger()->info(
    "[FireGate] mode={}, warmup={:.3f}s, stable_observations={}, max_change={:.3f}deg",
    FireGate::mode_name(result.mode), result.warmup_time, result.stable_observations,
    result.max_prediction_change_deg);
  return result;
}

int nearest_armor_id(const std::vector<Eigen::Vector4d> & armor_xyza_list)
{
  auto it = std::min_element(
    armor_xyza_list.begin(), armor_xyza_list.end(),
    [](const Eigen::Vector4d & a, const Eigen::Vector4d & b) {
      return a.head<2>().norm() < b.head<2>().norm();
    });
  return static_cast<int>(std::distance(armor_xyza_list.begin(), it));
}

}  // namespace

Planner::Planner(const std::string & config_path) : fire_gate_(load_fire_gate_config(config_path))
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  comming_angle_ = tools::read<double>(yaml, "comming_angle") / 57.3;
  leaving_angle_ = tools::read<double>(yaml, "leaving_angle") / 57.3;
  outpost_comming_angle_ = tools::read<double>(yaml, "outpost_comming_angle") / 57.3;
  outpost_leaving_angle_ = tools::read<double>(yaml, "outpost_leaving_angle") / 57.3;
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  outpost_delay_time_ = tools::read<double>(yaml, "outpost_delay_time");
  trajectory_config_ = load_trajectory_config(yaml, config_path);
  armor_lock_max_w_ = yaml["armor_lock_max_w"] ? yaml["armor_lock_max_w"].as<double>() : 2.0;
  armor_lock_angle_ =
    (yaml["armor_lock_angle"] ? yaml["armor_lock_angle"].as<double>() : 60.0) / 57.3;
  aim_center_enable_ =
    yaml["aim_center_enable"] ? yaml["aim_center_enable"].as<bool>() : aim_center_enable_;
  aim_center_near_spin_enable_ = yaml["aim_center_near_spin_enable"]
                                   ? yaml["aim_center_near_spin_enable"].as<bool>()
                                   : aim_center_near_spin_enable_;
  aim_center_spin_enable_ =
    yaml["aim_center_spin_enable"] ? yaml["aim_center_spin_enable"].as<bool>()
                                   : aim_center_spin_enable_;
  aim_center_near_spin_w_ = yaml["aim_center_near_spin_w"]
                              ? yaml["aim_center_near_spin_w"].as<double>()
                              : aim_center_near_spin_w_;
  aim_center_near_distance_ = yaml["aim_center_near_distance"]
                                ? yaml["aim_center_near_distance"].as<double>()
                                : aim_center_near_distance_;
  aim_center_spin_w_ =
    yaml["aim_center_spin_w"] ? yaml["aim_center_spin_w"].as<double>() : aim_center_spin_w_;
  aim_center_shoot_yaw_ =
    (yaml["aim_center_shoot_yaw"] ? yaml["aim_center_shoot_yaw"].as<double>() : 6.0) / 57.3;
  aim_center_fire_lead_time_ = yaml["aim_center_fire_lead_time"]
                                 ? yaml["aim_center_fire_lead_time"].as<double>()
                                 : aim_center_fire_lead_time_;
  aim_center_enter_step_ = yaml["aim_center_enter_step"]
                             ? yaml["aim_center_enter_step"].as<int>()
                             : aim_center_enter_step_;
  aim_center_exit_step_ =
    yaml["aim_center_exit_step"] ? yaml["aim_center_exit_step"].as<int>() : aim_center_exit_step_;
  aim_center_hold_threshold_ = yaml["aim_center_hold_threshold"]
                                 ? yaml["aim_center_hold_threshold"].as<int>()
                                 : aim_center_hold_threshold_;
  aim_center_hold_max_ =
    yaml["aim_center_hold_max"] ? yaml["aim_center_hold_max"].as<int>() : aim_center_hold_max_;

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan_impl(Target target, double bullet_speed, double imu_pitch)
{
  // 0. Check bullet speed
  auto raw_bullet_speed = bullet_speed;
  auto trajectory_config = trajectory_config_;
  trajectory_config.hero_imu_pitch = imu_pitch;
  bullet_speed = tools::normalize_bullet_speed(bullet_speed, trajectory_config);

  update_aim_center(target);
  update_aim_lock(target);

  // 1. Predict fly_time
  auto center_lock_base_target = target;
  auto xyza = select_aim_xyza(target);
  Eigen::Vector3d xyz = xyza.head<3>();
  auto min_dist = xyza.head<2>().norm();
  if (final_aim_center_) {
    min_dist = center_lock_flight_dist(target);
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z(), trajectory_config);
  last_plan_fly_time_ =
    !bullet_traj.unsolvable && std::isfinite(bullet_traj.fly_time) ? bullet_traj.fly_time : 0.0;
  target.predict(bullet_traj.fly_time);
  if (final_aim_center_) {
    auto center_lock_selection = select_center_lock_armors_by_fly_time(
      center_lock_base_target, bullet_speed, trajectory_config, aim_center_shoot_yaw_);
    center_lock_pitch_armor_id_ = center_lock_selection.pitch_id;
    center_lock_armor_id_ = center_lock_selection.fire_id;
    center_lock_fire_ready_ = center_lock_armor_id_ >= 0;

    if (center_lock_fire_ready_) {
      static auto last_center_fire_log_time = std::chrono::steady_clock::time_point{};
      auto now = std::chrono::steady_clock::now();
      if (
        last_center_fire_log_time.time_since_epoch().count() == 0 ||
        tools::delta_time(now, last_center_fire_log_time) >= 0.5) {
        last_center_fire_log_time = now;
        tools::logger()->info(
          "[Planner] aim center fire ready: pitch_id={} fire_id={} yaw_error={:.2f}deg "
          "fly={:.4f}s predict={:.4f}s lead={:.4f}s dist={:.3f}m",
          center_lock_pitch_armor_id_, center_lock_armor_id_,
          center_lock_selection.fire_yaw_error * 57.3, center_lock_selection.fire_fly_time,
          center_lock_selection.fire_predict_time, aim_center_fire_lead_time_,
          center_lock_selection.fire_dist);
      }
    }
  }
  const bool armor_in_fire_window = aimed_armor_in_fire_window(target);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed, trajectory_config)(0);
    traj = get_trajectory(target, yaw0, bullet_speed, trajectory_config);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  auto fire_ready_by_error =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  plan.fire = fire_ready_by_error && center_lock_fire_ready_ && armor_in_fire_window;
  if (trajectory_config.debug) {
    static auto last_debug_time = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();
    if (
      last_debug_time.time_since_epoch().count() == 0 ||
      tools::delta_time(now, last_debug_time) >= trajectory_config.debug_interval) {
      last_debug_time = now;
      auto model = trajectory_config.model == tools::TrajectoryModel::Hero ? "hero" : "standard";
      tools::logger()->info(
        "[Ballistic][Planner] model={} raw_v={:.2f} used_v={:.2f} imu_pitch={:.4f}rad "
        "xyza=({:.3f},{:.3f},{:.3f},{:.3f}) d={:.3f} traj_pitch0={:.4f}rad/{:.2f}deg "
        "fly={:.4f}s target_pitch={:.4f}rad/{:.2f}deg plan_pitch={:.4f}rad/{:.2f}deg "
        "target_yaw={:.4f}rad plan_yaw={:.4f}rad fire_error_ready={} center_ready={} "
        "armor_window_ready={} "
        "center_pitch_id={} center_fire_id={} s_bias={:.3f} k={:.6f}",
        model, raw_bullet_speed, bullet_speed, imu_pitch, xyza.x(), xyza.y(), xyza.z(), xyza.w(),
        min_dist, bullet_traj.pitch, bullet_traj.pitch * 57.3, bullet_traj.fly_time,
        plan.target_pitch, plan.target_pitch * 57.3, plan.pitch, plan.pitch * 57.3,
        plan.target_yaw, plan.yaw, fire_ready_by_error, center_lock_fire_ready_,
        armor_in_fire_window,
        center_lock_pitch_armor_id_, center_lock_armor_id_, trajectory_config.hero_s_bias,
        trajectory_config.hero_air_resistance_k);
    }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
  }
  return plan;
}

Plan Planner::plan(Target target, double bullet_speed, double imu_pitch)
{
  return plan(std::optional<Target>{std::move(target)}, bullet_speed, imu_pitch);
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed, double imu_pitch)
{
  const auto now = std::chrono::steady_clock::now();
  if (!target.has_value()) {
    reset_aim_center();
    previous_fire_gate_target_.reset();
    previous_fire_gate_epoch_ = 0;
    previous_fire_gate_observation_seq_ = 0;
    fire_gate_.update_no_target(now);
    fire_gate_.set_fire_result(false, false);
    log_fire_gate_state();
    return {false};
  }

  const auto observation_target = *target;
  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  if (target->name == ArmorName::outpost) {
    delay_time += outpost_delay_time_;
  }

  auto future = now + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  last_plan_fly_time_ = 0.0;
  auto result = plan_impl(*target, bullet_speed, imu_pitch);
  auto prediction_change = update_fire_gate_prediction(
    observation_target, bullet_speed, imu_pitch, now, delay_time, result.control);

  FireGateSample gate_sample;
  gate_sample.tracking = observation_target.tracking_info();
  gate_sample.aim_center = final_aim_center_;
  gate_sample.prediction_change_deg = prediction_change;
  gate_sample.now = now;
  fire_gate_.update(gate_sample);

  const bool raw_fire = result.fire;
  result.fire = raw_fire && fire_gate_.allow_fire();
  fire_gate_.set_fire_result(raw_fire, result.fire);
  log_fire_gate_state();
  return result;
}

const FireGateDebug & Planner::fire_gate_debug() const { return fire_gate_.debug(); }

void Planner::log_fire_gate_state()
{
  const auto & gate = fire_gate_.debug();
  if (gate.mode == FireGateMode::off) return;
  if (has_logged_fire_gate_state_ && gate.state == last_logged_fire_gate_state_) return;

  tools::logger()->info(
    "[FireGate] {} -> {}, stable={}, prediction_change={:.3f}deg, warmup_left={:.0f}ms, "
    "observation_age={:.0f}/{:.0f}ms",
    has_logged_fire_gate_state_ ? FireGate::state_name(last_logged_fire_gate_state_) : "START",
    FireGate::state_name(gate.state), gate.stable_observations, gate.prediction_change_deg,
    gate.warmup_remaining_ms, gate.observation_age_ms, gate.max_observation_age_ms);
  has_logged_fire_gate_state_ = true;
  last_logged_fire_gate_state_ = gate.state;
}

std::optional<double> Planner::update_fire_gate_prediction(
  const Target & observation_target, double bullet_speed, double imu_pitch,
  std::chrono::steady_clock::time_point now, double delay_time, bool plan_valid)
{
  const auto & tracking = observation_target.tracking_info();
  const bool new_observation =
    tracking.observation_seq != 0 &&
    (tracking.track_epoch != previous_fire_gate_epoch_ ||
     tracking.observation_seq != previous_fire_gate_observation_seq_);
  if (!new_observation) return std::nullopt;

  std::optional<double> prediction_change;
  const bool comparable =
    previous_fire_gate_target_.has_value() && tracking.track_epoch == previous_fire_gate_epoch_ &&
    final_aim_center_ == previous_fire_gate_aim_center_ && plan_valid && last_plan_fly_time_ > 0.0;

  if (comparable) {
    auto current = observation_target;
    auto previous = *previous_fire_gate_target_;
    const auto hit_time = now + std::chrono::microseconds(
                                  static_cast<int>((delay_time + last_plan_fly_time_) * 1e6));
    current.predict(hit_time);
    previous.predict(hit_time);

    auto trajectory_config = trajectory_config_;
    trajectory_config.hero_imu_pitch = imu_pitch;
    const auto normalized_bullet_speed =
      tools::normalize_bullet_speed(bullet_speed, trajectory_config);

    // 小陀螺装甲板交接时，两份快照必须比较同一块物理装甲板，避免最近装甲板编号
    // 在边界两侧变化而被误判为模型发散。
    const auto armor_list = current.armor_xyza_list();
    int armor_id = current_aim_id_;
    if (armor_id < 0 || armor_id >= static_cast<int>(armor_list.size())) {
      armor_id = armor_list.empty() ? -1 : nearest_armor_id(armor_list);
    }

    const auto current_yaw_pitch =
      fire_gate_aim(current, armor_id, normalized_bullet_speed, trajectory_config);
    const auto previous_yaw_pitch =
      fire_gate_aim(previous, armor_id, normalized_bullet_speed, trajectory_config);
    if (current_yaw_pitch && previous_yaw_pitch) {
      const auto delta_yaw =
        tools::limit_rad((*current_yaw_pitch)[0] - (*previous_yaw_pitch)[0]);
      const auto delta_pitch = (*current_yaw_pitch)[1] - (*previous_yaw_pitch)[1];
      prediction_change = std::hypot(delta_yaw, delta_pitch) * 57.3;
    }
  }

  if (tracking.live_observation && !tracking.temp_lost) {
    previous_fire_gate_target_ = observation_target;
    previous_fire_gate_epoch_ = tracking.track_epoch;
    previous_fire_gate_observation_seq_ = tracking.observation_seq;
    previous_fire_gate_aim_center_ = final_aim_center_;
  }

  return prediction_change;
}

std::optional<Eigen::Vector2d> Planner::fire_gate_aim(
  const Target & target, int armor_id, double bullet_speed,
  const tools::TrajectoryConfig & trajectory_config) const
{
  Eigen::Vector4d xyza;
  double flight_distance;
  if (final_aim_center_) {
    xyza = center_aim_xyza(target);
    flight_distance = center_lock_flight_dist(target);
  } else {
    const auto armor_list = target.armor_xyza_list();
    if (armor_id < 0 || armor_id >= static_cast<int>(armor_list.size())) return std::nullopt;
    xyza = armor_list[armor_id];
    flight_distance = xyza.head<2>().norm();
  }

  const auto bullet_traj =
    tools::Trajectory(bullet_speed, std::max(1e-3, flight_distance), xyza.z(), trajectory_config);
  if (bullet_traj.unsolvable || !std::isfinite(bullet_traj.pitch)) return std::nullopt;

  const auto yaw = tools::limit_rad(std::atan2(xyza.y(), xyza.x()) + yaw_offset_);
  const auto pitch = -bullet_traj.pitch - pitch_offset_;
  return Eigen::Vector2d{yaw, pitch};
}

bool Planner::aimed_armor_in_fire_window(const Target & target) const
{
  // 锁中心模式已有独立的装甲板穿线开火窗口，不叠加普通跟随装甲板的角度门控。
  if (final_aim_center_) return true;

  const auto ekf_x = target.ekf_x();
  const auto angular_velocity = ekf_x[7];

  // comming/leaving角度只约束小陀螺；低速目标保持原有开火逻辑。
  if (target.name != ArmorName::outpost && std::abs(angular_velocity) < armor_lock_max_w_)
    return true;

  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return false;

  // 必须检查Planner当前实际跟随的装甲板，不能用其他满足窗口的装甲板放行开火。
  const auto aimed_armor = select_aim_xyza(target);
  const auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  const auto delta_angle = tools::limit_rad(aimed_armor[3] - center_yaw);

  auto comming_angle = comming_angle_;
  auto leaving_angle = leaving_angle_;
  if (target.name == ArmorName::outpost) {
    comming_angle = outpost_comming_angle_;
    leaving_angle = outpost_leaving_angle_;
  }

  if (std::abs(delta_angle) > comming_angle) return false;
  if (angular_velocity > 0) return delta_angle < leaving_angle;
  if (angular_velocity < 0) return delta_angle > -leaving_angle;
  return false;
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(
  const Target & target, double bullet_speed, const tools::TrajectoryConfig & trajectory_config)
{
  auto xyza = select_aim_xyza(target);
  Eigen::Vector3d xyz = xyza.head<3>();
  auto yaw = xyza[3];
  auto min_dist = xyza.head<2>().norm();
  if (final_aim_center_) {
    min_dist = center_lock_flight_dist(target);
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z(), trajectory_config);
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

void Planner::reset_aim_center()
{
  aim_center_ = false;
  final_aim_center_ = false;
  center_lock_fire_ready_ = true;
  center_lock_aim_yaw_ = 0.0;
  aim_center_last_time_ = 0;
  center_lock_armor_id_ = -1;
  center_lock_pitch_armor_id_ = -1;
}

void Planner::update_aim_center(const Target & target)
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (!aim_center_enable_ || armor_xyza_list.empty()) {
    reset_aim_center();
    return;
  }

  const bool was_final_aim_center = final_aim_center_;
  auto ekf_x = target.ekf_x();
  const double center_dist_xy = std::hypot(ekf_x[0], ekf_x[2]);
  center_lock_aim_yaw_ = std::atan2(ekf_x[2], ekf_x[0]);
  const double abs_w = std::abs(ekf_x[7]);

  const bool hero_outpost_trigger =
    trajectory_config_.model == tools::TrajectoryModel::Hero && target.name == ArmorName::outpost;
  const bool near_spin_trigger =
    aim_center_near_spin_enable_ &&
    abs_w > aim_center_near_spin_w_ &&
    center_dist_xy < aim_center_near_distance_;
  const bool spin_trigger = aim_center_spin_enable_ && abs_w > aim_center_spin_w_;
  const bool enter_aim_center = hero_outpost_trigger || near_spin_trigger || spin_trigger;

  if (enter_aim_center) {
    aim_center_ = true;
    aim_center_last_time_ = hero_outpost_trigger
                              ? aim_center_hold_max_
                              : std::min(
                                  aim_center_hold_max_,
                                  aim_center_last_time_ + aim_center_enter_step_);
  } else {
    aim_center_ = false;
    aim_center_last_time_ = std::max(0, aim_center_last_time_ - aim_center_exit_step_);
  }

  final_aim_center_ =
    hero_outpost_trigger ||
    (enter_aim_center && aim_center_last_time_ >= aim_center_hold_threshold_);
  if (final_aim_center_) {
    center_lock_pitch_armor_id_ = select_center_lock_pitch_armor_id(target);
    center_lock_armor_id_ = select_center_lock_armor_id(target, aim_center_shoot_yaw_);
    center_lock_fire_ready_ = center_lock_armor_id_ >= 0;
  } else {
    center_lock_armor_id_ = -1;
    center_lock_pitch_armor_id_ = -1;
    center_lock_fire_ready_ = true;
  }

  if (!was_final_aim_center && final_aim_center_) {
    tools::logger()->info(
      "[Planner] enter aim center: abs_w={:.2f}rad/s dist={:.2f}m hero_outpost={} "
      "near_spin={} spin={} hold={} pitch_id={} fire_id={} fire_ready={}",
      abs_w, center_dist_xy, hero_outpost_trigger, near_spin_trigger, spin_trigger,
      aim_center_last_time_, center_lock_pitch_armor_id_, center_lock_armor_id_,
      center_lock_fire_ready_);
  }
}

int Planner::select_center_lock_armor_id(const Target & target, double fire_yaw_window) const
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return -1;

  auto ekf_x = target.ekf_x();
  const double center_dist_xy = std::hypot(ekf_x[0], ekf_x[2]);

  int best_id = -1;
  double min_yaw_error = std::numeric_limits<double>::max();

  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); i++) {
    const double armor_dist = armor_xyza_list[i].head<2>().norm();
    if (armor_dist > center_dist_xy) continue;

    const double future_armor_yaw = std::atan2(armor_xyza_list[i].y(), armor_xyza_list[i].x());
    const double yaw_error = std::abs(tools::limit_rad(future_armor_yaw - center_lock_aim_yaw_));
    if (yaw_error < fire_yaw_window && yaw_error < min_yaw_error) {
      best_id = i;
      min_yaw_error = yaw_error;
    }
  }

  return best_id;
}

int Planner::select_center_lock_pitch_armor_id(const Target & target) const
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return -1;

  auto ekf_x = target.ekf_x();
  const double center_dist_xy = std::hypot(ekf_x[0], ekf_x[2]);

  int best_id = -1;
  double min_yaw_error = std::numeric_limits<double>::max();

  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); i++) {
    const double armor_dist = armor_xyza_list[i].head<2>().norm();
    if (armor_dist > center_dist_xy) continue;

    const double future_armor_yaw = std::atan2(armor_xyza_list[i].y(), armor_xyza_list[i].x());
    const double yaw_error = std::abs(tools::limit_rad(future_armor_yaw - center_lock_aim_yaw_));
    if (yaw_error < min_yaw_error) {
      best_id = i;
      min_yaw_error = yaw_error;
    }
  }

  return best_id;
}

Planner::CenterLockArmorSelection Planner::select_center_lock_armors_by_fly_time(
  const Target & target, double bullet_speed, const tools::TrajectoryConfig & trajectory_config,
  double fire_yaw_window) const
{
  auto armor_xyza_list = target.armor_xyza_list();
  CenterLockArmorSelection selection;
  selection.pitch_yaw_error = std::numeric_limits<double>::max();
  selection.fire_yaw_error = std::numeric_limits<double>::max();
  if (armor_xyza_list.empty()) return selection;

  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); i++) {
    Target target_predict = target;
    Eigen::Vector4d armor_xyza = armor_xyza_list[i];
    double fly_time = 0.0;
    bool solved = true;

    // 用该候选装甲板自己的距离反推飞行时间，再预测装甲板位置；迭代两次减小近距离误差。
    for (int iter = 0; iter < 2; iter++) {
      const auto predict_armor_xyza_list = target_predict.armor_xyza_list();
      if (i >= static_cast<int>(predict_armor_xyza_list.size())) {
        solved = false;
        break;
      }

      armor_xyza = predict_armor_xyza_list[i];
      const double armor_dist = std::max(1e-3, armor_xyza.head<2>().norm());
      auto bullet_traj =
        tools::Trajectory(bullet_speed, armor_dist, armor_xyza.z(), trajectory_config);
      if (bullet_traj.unsolvable || !std::isfinite(bullet_traj.fly_time)) {
        solved = false;
        break;
      }

      fly_time = bullet_traj.fly_time;
      target_predict = target;
      target_predict.predict(fly_time);
    }

    if (!solved) continue;

    const auto pitch_armor_xyza_list = target_predict.armor_xyza_list();
    if (i >= static_cast<int>(pitch_armor_xyza_list.size())) continue;

    armor_xyza = pitch_armor_xyza_list[i];
    auto pitch_ekf_x = target_predict.ekf_x();
    const double pitch_center_dist_xy = std::hypot(pitch_ekf_x[0], pitch_ekf_x[2]);
    const double pitch_armor_dist = armor_xyza.head<2>().norm();
    if (pitch_armor_dist > pitch_center_dist_xy) continue;

    const double pitch_armor_yaw = std::atan2(armor_xyza.y(), armor_xyza.x());
    const double pitch_yaw_error =
      std::abs(tools::limit_rad(pitch_armor_yaw - center_lock_aim_yaw_));

    if (pitch_yaw_error < selection.pitch_yaw_error) {
      selection.pitch_id = i;
      selection.pitch_yaw_error = pitch_yaw_error;
    }

    Target fire_target_predict = target;
    const double fire_predict_time = std::max(0.0, fly_time + aim_center_fire_lead_time_);
    fire_target_predict.predict(fire_predict_time);
    const auto fire_armor_xyza_list = fire_target_predict.armor_xyza_list();
    if (i >= static_cast<int>(fire_armor_xyza_list.size())) continue;

    armor_xyza = fire_armor_xyza_list[i];
    auto fire_ekf_x = fire_target_predict.ekf_x();
    const double fire_center_dist_xy = std::hypot(fire_ekf_x[0], fire_ekf_x[2]);
    const double fire_armor_dist = armor_xyza.head<2>().norm();
    if (fire_armor_dist > fire_center_dist_xy) continue;

    const double fire_armor_yaw = std::atan2(armor_xyza.y(), armor_xyza.x());
    const double fire_yaw_error =
      std::abs(tools::limit_rad(fire_armor_yaw - center_lock_aim_yaw_));

    if (fire_yaw_error < fire_yaw_window && fire_yaw_error < selection.fire_yaw_error) {
      selection.fire_id = i;
      selection.fire_yaw_error = fire_yaw_error;
      selection.fire_fly_time = fly_time;
      selection.fire_predict_time = fire_predict_time;
      selection.fire_dist = fire_armor_dist;
    }
  }

  return selection;
}

double Planner::center_lock_radius(const Target & target) const
{
  auto ekf_x = target.ekf_x();
  const double r1 = std::abs(ekf_x[8]);
  const double r2 = target.name == ArmorName::outpost ? r1 : std::abs(ekf_x[8] + ekf_x[9]);
  return 0.5 * (r1 + r2);
}

double Planner::center_lock_flight_dist(const Target & target) const
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (
    center_lock_pitch_armor_id_ >= 0 &&
    center_lock_pitch_armor_id_ < static_cast<int>(armor_xyza_list.size())) {
    return std::max(1e-3, armor_xyza_list[center_lock_pitch_armor_id_].head<2>().norm());
  }

  auto ekf_x = target.ekf_x();
  const double center_dist_xy = std::hypot(ekf_x[0], ekf_x[2]);
  return std::max(1e-3, center_dist_xy - center_lock_radius(target));
}

Eigen::Vector4d Planner::center_aim_xyza(const Target & target) const
{
  auto ekf_x = target.ekf_x();
  auto armor_xyza_list = target.armor_xyza_list();

  double z = ekf_x[4];
  double yaw = std::atan2(ekf_x[2], ekf_x[0]);

  if (
    target.name == ArmorName::outpost && center_lock_pitch_armor_id_ >= 0 &&
    center_lock_pitch_armor_id_ < static_cast<int>(armor_xyza_list.size())) {
    z = armor_xyza_list[center_lock_pitch_armor_id_][2];
    yaw = armor_xyza_list[center_lock_pitch_armor_id_][3];
  }

  return {ekf_x[0], ekf_x[2], z, yaw};
}

void Planner::update_aim_lock(const Target & target)
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) {
    lock_id_ = -1;
    current_aim_id_ = -1;
    return;
  }

  auto use_nearest = [&]() {
    lock_id_ = -1;
    current_aim_id_ = nearest_armor_id(armor_xyza_list);
  };

  if (!target.jumped) {
    lock_id_ = -1;
    current_aim_id_ = 0;
    return;
  }

  if (target.name == ArmorName::outpost || std::abs(target.ekf_x()[7]) >= armor_lock_max_w_) {
    use_nearest();
    return;
  }

  auto ekf_x = target.ekf_x();
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);
  std::vector<int> candidate_ids;
  std::vector<double> delta_angles;

  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angles.push_back(delta_angle);
    if (std::abs(delta_angle) <= armor_lock_angle_) candidate_ids.push_back(i);
  }

  if (candidate_ids.empty()) {
    use_nearest();
    return;
  }

  if (candidate_ids.size() == 1) {
    lock_id_ = -1;
    current_aim_id_ = candidate_ids.front();
    return;
  }

  auto locked =
    std::find(candidate_ids.begin(), candidate_ids.end(), lock_id_) != candidate_ids.end();
  if (!locked) {
    lock_id_ = *std::min_element(
      candidate_ids.begin(), candidate_ids.end(),
      [&](int a, int b) { return std::abs(delta_angles[a]) < std::abs(delta_angles[b]); });
  }

  current_aim_id_ = lock_id_;
}

Eigen::Vector4d Planner::select_aim_xyza(const Target & target) const
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return Eigen::Vector4d::Zero();
  if (final_aim_center_) return center_aim_xyza(target);
  if (!target.jumped) return armor_xyza_list[0];

  auto use_current_aim =
    target.name != ArmorName::outpost && std::abs(target.ekf_x()[7]) < armor_lock_max_w_ &&
    current_aim_id_ >= 0 &&
    current_aim_id_ < static_cast<int>(armor_xyza_list.size());
  if (use_current_aim) return armor_xyza_list[current_aim_id_];

  return armor_xyza_list[nearest_armor_id(armor_xyza_list)];
}

Trajectory Planner::get_trajectory(
  Target & target, double yaw0, double bullet_speed,
  const tools::TrajectoryConfig & trajectory_config)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed, trajectory_config);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed, trajectory_config);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed, trajectory_config);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim
