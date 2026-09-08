#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "ignore_num.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  ignored_armors_ = read_ignore_num(yaml, "Tracker");
  priority_armors_ = read_priority_num(yaml, "Tracker");
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  max_extinguish_track_count_ =
    yaml["max_extinguish_track_count"] ? yaml["max_extinguish_track_count"].as<int>() : 8;
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;

  auto read_double = [&](const std::string & key, double default_value) {
    return yaml[key] ? yaml[key].as<double>() : default_value;
  };
  auto read_bool = [&](const std::string & key, bool default_value) {
    return yaml[key] ? yaml[key].as<bool>() : default_value;
  };
  auto read_int = [&](const std::string & key, int default_value) {
    return yaml[key] ? yaml[key].as<int>() : default_value;
  };

  max_tracking_dt_s_ = read_double("tracker_max_dt_s", max_tracking_dt_s_);
  if (!std::isfinite(max_tracking_dt_s_) || max_tracking_dt_s_ <= 0.0) {
    throw std::runtime_error("tracker_max_dt_s must be a finite value greater than zero");
  }

  ekf_noise_config_.normal_v1 = read_double("ekf_normal_v1", ekf_noise_config_.normal_v1);
  ekf_noise_config_.normal_v2 = read_double("ekf_normal_v2", ekf_noise_config_.normal_v2);
  ekf_noise_config_.outpost_v1 = read_double("ekf_outpost_v1", ekf_noise_config_.outpost_v1);
  ekf_noise_config_.outpost_v2 = read_double("ekf_outpost_v2", ekf_noise_config_.outpost_v2);
  ekf_noise_config_.adaptive_v1 = read_bool("ekf_adaptive_v1", ekf_noise_config_.adaptive_v1);
  ekf_noise_config_.adaptive_v1_high =
    read_double("ekf_adaptive_v1_high", ekf_noise_config_.adaptive_v1_high);
  ekf_noise_config_.adaptive_v1_decay =
    read_double("ekf_adaptive_v1_decay", ekf_noise_config_.adaptive_v1_decay);
  ekf_noise_config_.adaptive_v1_residual_yaw = read_double(
    "ekf_adaptive_v1_residual_yaw", ekf_noise_config_.adaptive_v1_residual_yaw);
  ekf_noise_config_.adaptive_v1_residual_pitch = read_double(
    "ekf_adaptive_v1_residual_pitch", ekf_noise_config_.adaptive_v1_residual_pitch);
  ekf_noise_config_.adaptive_v1_residual_distance = read_double(
    "ekf_adaptive_v1_residual_distance", ekf_noise_config_.adaptive_v1_residual_distance);
  ekf_noise_config_.adaptive_v2 = read_bool("ekf_adaptive_v2", ekf_noise_config_.adaptive_v2);
  ekf_noise_config_.adaptive_v2_high =
    read_double("ekf_adaptive_v2_high", ekf_noise_config_.adaptive_v2_high);
  ekf_noise_config_.adaptive_v2_decay =
    read_double("ekf_adaptive_v2_decay", ekf_noise_config_.adaptive_v2_decay);
  ekf_noise_config_.adaptive_v2_residual_angle = read_double(
    "ekf_adaptive_v2_residual_angle", ekf_noise_config_.adaptive_v2_residual_angle);
  ekf_noise_config_.adaptive_v2_nis =
    read_double("ekf_adaptive_v2_nis", ekf_noise_config_.adaptive_v2_nis);
  if (yaml["z_observation_only"]) {
    auto z_observation_mode = yaml["z_observation_only"].Scalar();
    std::transform(
      z_observation_mode.begin(), z_observation_mode.end(), z_observation_mode.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (z_observation_mode == "true") {
      ekf_noise_config_.z_observation_only_mode = ZObservationOnlyMode::enabled;
    } else if (z_observation_mode == "false") {
      ekf_noise_config_.z_observation_only_mode = ZObservationOnlyMode::disabled;
    } else if (z_observation_mode == "auto") {
      ekf_noise_config_.z_observation_only_mode = ZObservationOnlyMode::automatic;
    } else {
      throw std::runtime_error(
        "z_observation_only must be true, false, or auto; got: " + z_observation_mode);
    }
  }
  ekf_noise_config_.z_observation_auto_window_frames = read_int(
    "z_observation_auto_window_frames", ekf_noise_config_.z_observation_auto_window_frames);
  ekf_noise_config_.z_observation_auto_range_threshold = read_double(
    "z_observation_auto_range_threshold",
    ekf_noise_config_.z_observation_auto_range_threshold);
  if (ekf_noise_config_.z_observation_only_mode == ZObservationOnlyMode::enabled) {
    tools::logger()->info(
      "[Tracker] Z observation-only mode enabled for normal targets; outpost keeps Z/vz EKF "
      "prediction.");
  } else if (ekf_noise_config_.z_observation_only_mode == ZObservationOnlyMode::automatic) {
    tools::logger()->info(
      "[Tracker] Z observation-only auto mode: enable when the latest {} observed frames span "
      "more than {:.3f} m; reset on target loss.",
      std::max(1, ekf_noise_config_.z_observation_auto_window_frames),
      std::max(0.0, ekf_noise_config_.z_observation_auto_range_threshold));
  }
}

std::string Tracker::state() const { return current_slot_.state; }

double Tracker::max_tracking_dt_s() const { return max_tracking_dt_s_; }

void Tracker::set_enemy_color(Color enemy_color)
{
  if (enemy_color_ == enemy_color) return;

  enemy_color_ = enemy_color;
  reset_slot(current_slot_);
  reset_slot(candidate_slot_);
  pre_state_ = "lost";

  tools::logger()->info(
    "[Tracker] Enemy color set to {}", COLORS.at(static_cast<std::size_t>(enemy_color_)));
}

void Tracker::set_enemy_color_from_camp(uint8_t camp)
{
  // camp is self camp: red shoots blue, blue shoots red.
  switch (camp) {
    case 0:
      set_enemy_color(Color::blue);
      break;
    case 1:
      set_enemy_color(Color::red);
      break;
    default: {
      static uint8_t last_invalid_camp = 0xff;
      if (camp != last_invalid_camp) {
        tools::logger()->warn("[Tracker] Invalid camp: {}", int(camp));
        last_invalid_camp = camp;
      }
      break;
    }
  }
}

void Tracker::apply_priority_config(std::list<Armor> & armors) const
{
  set_configured_priority(armors, priority_armors_);
}

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  auto & current = current_slot_;
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (current.state != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    reset_slot(current);
    reset_slot(candidate_slot_);
  }
  // 过滤掉非敌方装甲板
  if (use_enemy_color) {
    const bool allow_extinguish_current_target = current.state != "lost";
    armors.remove_if([&](const auto_aim::Armor & a) {
      if (a.color == enemy_color_) return false;
      return !(
        allow_extinguish_current_target && a.color == Color::extinguish &&
        a.name == current.target.name && a.type == current.target.armor_type);
    });
  }
  remove_ignored_armors(armors, ignored_armors_);
  apply_priority_config(armors);

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  // The normal tracking entry point is used by the current executables.  Keep the tracked target
  // until a strictly higher-priority, non-extinguished armor is visible, then rebuild the track
  // from that armor.  Previously this switch existed only in the unused omniperception overload,
  // so priority_num affected initial target selection but not an active track.
  const auto best_target = std::find_if(
    armors.cbegin(), armors.cend(),
    [](const Armor & armor) { return armor.color != Color::extinguish; });
  const bool switch_to_higher_priority =
    current.state != "lost" && best_target != armors.cend() &&
    best_target->priority < current.target.priority;

  if (switch_to_higher_priority) {
    const auto old_name = current.target.name;
    reset_slot(current);
    reset_slot(candidate_slot_);
    process_slot(current, armors, t);
    tools::logger()->info(
      "[Tracker] Higher-priority target selected: {} -> {}", ARMOR_NAMES[old_name],
      ARMOR_NAMES[current.target.name]);
  } else {
    process_slot(current, armors, t);
  }
  const bool current_valid = validate_slot(current, true);
  const auto current_loss_reason = current.loss_reason;
  const bool current_gone =
    current_loss_reason == LossReason::invisible ||
    current_loss_reason == LossReason::extinguished;

  if (!current_valid && !current_gone) {
    reset_slot(candidate_slot_);
    return {};
  }

  // Prepare a replacement while the current target is invisible or under gray-armor judgment.
  if (
    current_gone || current.state == "temp_lost" || current.extinguish_track_count > 0) {
    process_slot(candidate_slot_, armors, t);
    const bool candidate_valid = validate_slot(candidate_slot_, true);

    // The candidate's age is not a promotion gate; only definitive loss of the current target is.
    if (current_gone && candidate_valid) {
      const auto old_name = current.target.name;
      const auto new_name = candidate_slot_.target.name;
      const char * reason = current_loss_reason == LossReason::extinguished
                              ? "gray armor timeout"
                              : "visibility timeout";
      std::swap(current, candidate_slot_);
      reset_slot(candidate_slot_);
      tools::logger()->info(
        "[Tracker] Candidate target promoted after {}: {} -> {}", reason,
        ARMOR_NAMES[old_name], ARMOR_NAMES[new_name]);
    } else if (current_gone) {
      reset_slot(candidate_slot_);
      return {};
    }
  } else {
    reset_slot(candidate_slot_);
  }

  if (current.state == "lost") return {};

  sync_tracking_info(current);
  std::list<Target> targets = {current.target};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  auto & current = current_slot_;
  reset_slot(candidate_slot_);
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (current.state != "lost" && dt > max_tracking_dt_s_) {
    tools::logger()->warn(
      "[Tracker] Large dt: {:.3f}s exceeds reset threshold {:.3f}s", dt, max_tracking_dt_s_);
    reset_slot(current);
  }

  if (use_enemy_color) {
    const bool allow_extinguish_current_target = current.state != "lost";
    auto should_remove = [&](const auto_aim::Armor & a) {
      if (a.color == enemy_color_) return false;
      return !(
        allow_extinguish_current_target && a.color == Color::extinguish &&
        a.name == current.target.name && a.type == current.target.armor_type);
    };
    armors.remove_if(should_remove);
    temp_target.armors.remove_if(should_remove);
  }
  remove_ignored_armors(armors, ignored_armors_);
  remove_ignored_armors(temp_target.armors, ignored_armors_);
  apply_priority_config(armors);
  apply_priority_config(temp_target.armors);

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1280 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (current.state == "lost") {
    found = set_target(current, armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    current.state == "tracking" && !armors.empty() &&
    armors.front().priority < current.target.priority) {
    found = set_target(current, armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    current.state == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < current.target.priority && current.target.convergened()) {
    current.state = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (current.state == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  else if (current.state == "detecting" && pre_state_ == "switching") {
    found = set_target(current, armors, t);
  }

  else {
    found = update_target(current, armors, t);
  }

  pre_state_ = current.state;
  // 更新状态机
  state_machine(current, found);

  // 发散检测
  if (current.state != "lost" && current.target.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    reset_slot(current);
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (current.state == "lost")
    return {switch_target, {}};  // 返回switch_target和空的targets

  sync_tracking_info(current);
  std::list<Target> targets = {current.target};
  return {switch_target, targets};
}

void Tracker::reset_slot(TrackSlot & slot, LossReason loss_reason)
{
  slot.state = "lost";
  slot.detect_count = 0;
  slot.temp_lost_count = 0;
  slot.extinguish_track_count = 0;
  slot.loss_reason = loss_reason;
  slot.track_epoch = 0;
  slot.observation_seq = 0;
  slot.observed_frames = 0;
  slot.live_observation = false;
  slot.first_observed = {};
  slot.last_observed = {};
}

void Tracker::sync_tracking_info(TrackSlot & slot)
{
  TrackingInfo info;
  info.track_epoch = slot.track_epoch;
  info.observation_seq = slot.observation_seq;
  info.observed_frames = slot.observed_frames;
  info.confirmed = slot.state == "tracking";
  info.temp_lost = slot.state == "temp_lost";
  info.live_observation = slot.live_observation;
  info.first_observed = slot.first_observed;
  info.last_observed = slot.last_observed;
  slot.target.set_tracking_info(info);
}

bool Tracker::process_slot(
  TrackSlot & slot, std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  const bool found =
    slot.state == "lost" ? set_target(slot, armors, t) : update_target(slot, armors, t);
  state_machine(slot, found);
  return found;
}

bool Tracker::validate_slot(TrackSlot & slot, bool check_nis)
{
  if (slot.state == "lost") return false;

  if (slot.target.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    reset_slot(slot, LossReason::invalid);
    return false;
  }

  if (
    check_nis &&
    std::accumulate(
      slot.target.ekf().recent_nis_failures.begin(),
      slot.target.ekf().recent_nis_failures.end(), 0) >=
      (0.4 * slot.target.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    reset_slot(slot, LossReason::invalid);
    return false;
  }

  return true;
}

void Tracker::state_machine(TrackSlot & slot, bool found)
{
  if (slot.state == "lost") {
    if (!found) return;

    slot.state = "detecting";
    slot.detect_count = 1;
    slot.extinguish_track_count = 0;
  }

  else if (slot.state == "detecting") {
    if (found) {
      slot.detect_count++;
      if (slot.detect_count >= min_detect_count_) slot.state = "tracking";
    } else {
      reset_slot(slot);
    }
  }

  else if (slot.state == "tracking") {
    if (found) return;

    slot.temp_lost_count = 1;
    slot.state = "temp_lost";
  }

  else if (slot.state == "switching") {
    if (found) {
      slot.state = "detecting";
    } else {
      slot.temp_lost_count++;
      if (slot.temp_lost_count > 200) {
        reset_slot(slot);
      }
    }
  }

  else if (slot.state == "temp_lost") {
    if (found) {
      slot.state = "tracking";
    } else {
      slot.temp_lost_count++;
      const int lost_count_limit = slot.target.name == ArmorName::outpost
                                     ? outpost_max_temp_lost_count_
                                     : normal_temp_lost_count_;
      if (slot.temp_lost_count > lost_count_limit) {
        reset_slot(slot, LossReason::invisible);
      }
    }
  }
}

bool Tracker::set_target(
  TrackSlot & slot, std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  slot.loss_reason = LossReason::none;
  slot.live_observation = false;
  if (armors.empty()) return false;

  auto armor_it = armors.begin();
  while (armor_it != armors.end() && armor_it->color == Color::extinguish) {
    ++armor_it;
  }
  if (armor_it == armors.end()) return false;

  auto & armor = *armor_it;
  solver_.solve(armor);

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    slot.target = Target(armor, t, 0.2, 2, P0_dig, ekf_noise_config_);
  }

  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 1, 1}};
    slot.target = Target(armor, t, 0.2765, 3, P0_dig, ekf_noise_config_);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    slot.target = Target(armor, t, 0.3205, 3, P0_dig, ekf_noise_config_);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    slot.target = Target(armor, t, 0.2, 4, P0_dig, ekf_noise_config_);
  }

  slot.track_epoch = ++next_track_epoch_;
  slot.observation_seq = 1;
  slot.observed_frames = 1;
  slot.live_observation = true;
  slot.first_observed = t;
  slot.last_observed = t;
  slot.extinguish_track_count = 0;
  return true;
}

bool Tracker::update_target(
  TrackSlot & slot, std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  slot.loss_reason = LossReason::none;
  slot.live_observation = false;
  slot.target.predict(t);

  bool found_enemy_color = false;
  bool found_extinguish = false;
  for (const auto & armor : armors) {
    if (armor.name != slot.target.name || armor.type != slot.target.armor_type) continue;
    if (armor.color == enemy_color_) {
      found_enemy_color = true;
    } else if (armor.color == Color::extinguish) {
      found_extinguish = true;
    }
  }

  if (!found_enemy_color && !found_extinguish) {
    slot.target.reset_z_observation_history();
    return false;
  }

  const bool use_extinguish = !found_enemy_color && found_extinguish;
  if (use_extinguish) {
    slot.extinguish_track_count++;
    if (slot.extinguish_track_count > max_extinguish_track_count_) {
      tools::logger()->info(
        "[Tracker] {} stayed extinguished for {} frames, treat as dead.",
        ARMOR_NAMES[slot.target.name], slot.extinguish_track_count);
      reset_slot(slot, LossReason::extinguished);
      return false;
    }
  } else {
    slot.extinguish_track_count = 0;
  }

  bool record_z_observation = true;
  for (auto & armor : armors) {
    if (
      armor.name != slot.target.name || armor.type != slot.target.armor_type ||
      armor.color != (use_extinguish ? Color::extinguish : enemy_color_))
      continue;

    solver_.solve(armor);

    slot.target.update(armor, record_z_observation);
    record_z_observation = false;
  }

  if (!use_extinguish) {
    slot.observation_seq++;
    slot.observed_frames++;
    slot.live_observation = true;
    slot.last_observed = t;
  }

  return true;
}

}  // namespace auto_aim
