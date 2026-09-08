#include <fmt/core.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/ros2/ros2.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/omniperception/omni_status_decider.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | show command line help }"
  "{@config-path   | configs/sentry.yaml | config yaml path }";

namespace
{
struct AimInput
{
  std::optional<auto_aim::Target> target;
};

struct PlanSnapshot
{
  auto_aim::Plan plan{};
  std::optional<auto_aim::Target> target;
  Eigen::Vector4d aim_xyza = Eigen::Vector4d::Zero();
  bool aim_valid = false;
};

struct MainAutoAimDebugFrame
{
  cv::Mat img;
  std::list<auto_aim::Armor> armors;
  std::optional<auto_aim::Target> target;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  double detect_ms = 0.0;
  bool valid = false;
};

struct OmniDebugFrame
{
  cv::Mat img;
  std::list<auto_aim::Armor> armors;
  int status = 0;
  bool active = false;
};

enum class SentryMode : uint8_t
{
  IDLE = 0,
  AUTO_AIM = 1,
  SMALL_BUFF = 2,
  BIG_BUFF = 3
};

SentryMode sentry_mode(uint8_t mode)
{
  switch (mode) {
    case 1:
      return SentryMode::AUTO_AIM;
    case 2:
      return SentryMode::SMALL_BUFF;
    case 3:
      return SentryMode::BIG_BUFF;
    default:
      return SentryMode::IDLE;
  }
}

std::string mode_name(SentryMode mode)
{
  switch (mode) {
    case SentryMode::IDLE:
      return "IDLE";
    case SentryMode::AUTO_AIM:
      return "AUTO_AIM";
    case SentryMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case SentryMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

io::GimbalState to_gimbal_state(const io::NavGimbalState & state)
{
  return io::GimbalState{state.yaw,          state.yaw_vel,      state.pitch, state.pitch_vel,
                         state.bullet_speed, state.bullet_count, state.camp};
}

void draw_armors(cv::Mat & img, const std::list<auto_aim::Armor> & armors)
{
  for (const auto & armor : armors) {
    const auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, auto_aim::COLORS.at(armor.color),
      auto_aim::ARMOR_NAMES.at(armor.name), auto_aim::ARMOR_TYPES.at(armor.type));
    tools::draw_points(img, armor.points, {0, 255, 0}, 2);
    tools::draw_text(img, info, armor.center, {0, 255, 0}, 0.55, 2);
  }
}

void draw_line(
  cv::Mat & img, const std::string & text, int line, const cv::Scalar & color = {255, 255, 255})
{
  tools::draw_text(img, text, {12, 28 + line * 27}, color, 0.65, 2);
}

cv::Mat make_status_image(const std::string & text)
{
  cv::Mat img(360, 640, CV_8UC3, cv::Scalar(25, 25, 25));
  tools::draw_text(img, text, {30, 190}, {200, 200, 200}, 0.9, 2);
  return img;
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>(0);

  tools::Exiter exiter;
  tools::Plotter plotter;

  auto yaml = tools::load(config_path);
  const YAML::Node combat_camera_yaml = yaml["combat_camera"] ? yaml["combat_camera"] : yaml;
  const double sentry_mpc_fire_gap_time = std::max(0.0, yaml["fire_gap_time"].as<double>());
  const YAML::Node command_guard_yaml = yaml["command_guard"];
  const double buff_plan_hold_sec = std::max(
    0.0, command_guard_yaml && command_guard_yaml["publish_hold_sec"]
           ? command_guard_yaml["publish_hold_sec"].as<double>()
           : 0.05);
  const auto buff_plan_hold_time = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(buff_plan_hold_sec));
  if (!yaml["omni_camera"]) {
    throw std::runtime_error("Missing 'omni_camera' in sentry config.");
  }
  const YAML::Node omni_camera_yaml = yaml["omni_camera"];
  const bool omni_enabled =
    omni_camera_yaml["enabled"] ? omni_camera_yaml["enabled"].as<bool>() : true;
  const double omni_frame_rate =
    omni_camera_yaml["frame_rate"] ? omni_camera_yaml["frame_rate"].as<double>() : 0.0;
  const auto omni_frame_period =
    omni_frame_rate > 0 ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(1.0 / omni_frame_rate))
                        : std::chrono::steady_clock::duration::zero();

  const double exposure_ms = tools::read<double>(combat_camera_yaml, "exposure_ms");
  const double buff_exposure_ms = combat_camera_yaml["buff_exposure"]
                                    ? combat_camera_yaml["buff_exposure"].as<double>()
                                    : exposure_ms;
  auto exposure_center_offset = [](double value_ms) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(value_ms * 0.5));
  };

  io::ROS2 ros2;
  io::Camera main_camera(combat_camera_yaml, true);
  auto_aim::YOLO main_yolo_primary(config_path, false);
  auto_aim::YOLO main_yolo_secondary(config_path, false);
  auto_aim::YOLO * main_yolo_instances[] = {&main_yolo_primary, &main_yolo_secondary};
  auto_aim::Solver solver(config_path);
  solver.set_debug_xyz_optimize_log(true);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  std::unique_ptr<io::Camera> omni_camera;
  std::unique_ptr<auto_aim::YOLO> omni_yolo;
  std::unique_ptr<omniperception::OmniStatusDecider> omni_status_decider;
  if (omni_enabled) {
    omni_camera = std::make_unique<io::Camera>(omni_camera_yaml, false);
    omni_yolo = std::make_unique<auto_aim::YOLO>(config_path, false);
    omni_status_decider = std::make_unique<omniperception::OmniStatusDecider>(config_path);
  } else {
    tools::logger()->warn("[SentryMPCDebug] Omni camera is disabled by config.");
    ros2.publish_omni_target(0);
  }

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::Aimer buff_aimer(config_path);
  buff_aimer.set_mpc_fire_gap_time_override(sentry_mpc_fire_gap_time);
  tools::logger()->info(
    "[SentryMPCDebug] Buff MPC fire interval: {:.3f} s", sentry_mpc_fire_gap_time);

  tools::ThreadSafeQueue<AimInput, true> target_queue(1);
  target_queue.push({});
  tools::ThreadSafeQueue<OmniDebugFrame, true> omni_debug_queue(1);

  std::atomic<bool> quit = false;
  std::atomic<SentryMode> mode{SentryMode::IDLE};
  std::mutex plan_mutex;
  PlanSnapshot latest_plan;

  auto store_plan = [&](const PlanSnapshot & snapshot) {
    std::lock_guard<std::mutex> lock(plan_mutex);
    latest_plan = snapshot;
  };
  auto load_plan = [&]() {
    std::lock_guard<std::mutex> lock(plan_mutex);
    return latest_plan;
  };
  auto publish_empty_tracker = [&]() { ros2.publish_tracker_target(Eigen::Vector3d::Zero(), 0); };
  auto publish_tracker_plan =
    [&](const std::optional<auto_aim::Target> & target, float yaw, float pitch) {
      if (!target) {
        publish_empty_tracker();
        return;
      }

      const auto pnp_xyz_in_camera = target->last_armor_pnp_xyz_in_camera();
      if (!pnp_xyz_in_camera.allFinite()) {
        publish_empty_tracker();
        return;
      }

      ros2.publish_tracker_target(
        pnp_xyz_in_camera, static_cast<int>(target->name) + 1, yaw, pitch);
    };

  auto plan_thread = std::thread([&]() {
    const auto plan_send_period = 10ms;
    auto next_send = std::chrono::steady_clock::now();

    while (!quit) {
      if (mode.load() != SentryMode::AUTO_AIM) {
        next_send = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(20ms);
        continue;
      }

      next_send += plan_send_period;
      const auto input = target_queue.front();
      const auto nav_gs = ros2.subscribe_gimbal_state();

      PlanSnapshot snapshot;
      snapshot.target = input.target;
      if (nav_gs) {
        snapshot.plan = planner.plan(input.target, nav_gs->bullet_speed, nav_gs->pitch);
        snapshot.aim_xyza = planner.debug_xyza;
        snapshot.aim_valid = input.target.has_value() && snapshot.plan.control;
        ros2.publish_auto_aim_plan(
          snapshot.plan.control, snapshot.plan.fire, snapshot.plan.yaw, snapshot.plan.yaw_vel,
          snapshot.plan.yaw_acc, snapshot.plan.pitch, snapshot.plan.pitch_vel,
          snapshot.plan.pitch_acc);
        publish_tracker_plan(input.target, snapshot.plan.yaw, snapshot.plan.pitch);
      } else {
        ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
        publish_empty_tracker();
      }
      store_plan(snapshot);

      const auto now = std::chrono::steady_clock::now();
      if (now > next_send + plan_send_period) next_send = now;
      std::this_thread::sleep_until(next_send);
    }
  });

  std::thread omni_thread;
  if (omni_enabled) {
    omni_thread = std::thread([&]() {
      cv::Mat img;
      std::chrono::steady_clock::time_point timestamp;
      int frame_count = 0;
      bool active = false;
      auto next_frame_time = std::chrono::steady_clock::now();

      while (!quit) {
        if (mode.load() != SentryMode::AUTO_AIM) {
          if (active) {
            omni_camera->set_active(false);
            omni_status_decider->reset();
            ros2.publish_omni_target(0);
            omni_debug_queue.push({{}, {}, 0, false});
            active = false;
          }
          next_frame_time = std::chrono::steady_clock::now();
          std::this_thread::sleep_for(20ms);
          continue;
        }

        if (!active) {
          omni_camera->set_active(true);
          omni_status_decider->reset();
          active = true;
          next_frame_time = std::chrono::steady_clock::now();
        }

        if (omni_frame_period > std::chrono::steady_clock::duration::zero()) {
          auto now = std::chrono::steady_clock::now();
          if (now < next_frame_time) {
            std::this_thread::sleep_until(next_frame_time);
            if (quit) break;
            now = std::chrono::steady_clock::now();
          }
          next_frame_time =
            now > next_frame_time + omni_frame_period ? now : next_frame_time + omni_frame_period;
        }

        omni_camera->read(img, timestamp);
        auto armors = omni_yolo->detect(img, frame_count++);
        const auto nav_gs = ros2.subscribe_gimbal_state();
        const int status = omni_status_decider->update(armors, nav_gs ? nav_gs->camp : 0xff);
        ros2.publish_omni_target(status);
        omni_debug_queue.push({img.clone(), std::move(armors), status, true});
      }

      if (active) omni_camera->set_active(false);
    });
  }

  cv::namedWindow("sentry_mpc/main", cv::WINDOW_NORMAL);
  cv::namedWindow("sentry_mpc/omni", cv::WINDOW_NORMAL);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point buff_time_origin;
  bool buff_time_origin_ready = false;
  SentryMode last_mode = SentryMode::IDLE;
  double current_exposure_ms = -1.0;
  double display_fps = 0.0;
  int display_frames = 0;
  auto fps_stamp = std::chrono::steady_clock::now();
  MainAutoAimDebugFrame main_auto_aim_debug;
  OmniDebugFrame omni_debug;

  int main_yolo_frame_count = 0;
  bool has_last_auto_aim_result_t = false;
  auto last_auto_aim_result_t = std::chrono::steady_clock::time_point{};
  auto auto_aim_mode_enter_t = std::chrono::steady_clock::now();
  int buff_frame_count = 0;
  bool has_last_buff_result_t = false;
  auto last_buff_result_t = std::chrono::steady_clock::time_point{};
  auto_aim::Plan last_valid_buff_plan{};
  bool has_last_valid_buff_plan = false;
  auto last_valid_buff_plan_t = std::chrono::steady_clock::time_point{};
  constexpr size_t max_async_frame_qs = 128;
  std::map<int, Eigen::Quaterniond> async_frame_qs;
  std::map<int, Eigen::Quaterniond> buff_frame_qs;

  const bool use_async_yolo =
    main_yolo_primary.supports_async() && main_yolo_secondary.supports_async();
  tools::logger()->info(
    "Sentry debug main YOLO pipeline mode: {}",
    use_async_yolo ? "OpenVINO async dual-instance" : "sync fallback");

  auto discard_pending_yolo_results = [&]() {
    if (!use_async_yolo) return;

    auto_aim::YOLOAsyncResult dropped_result;
    for (auto * yolo : main_yolo_instances) {
      while (yolo->try_fetch(dropped_result)) {
      }
    }
    async_frame_qs.clear();
  };

  auto discard_pending_buff_results = [&]() {
    buff_detector.discard_pending_results();
    buff_frame_qs.clear();
    buff_aimer.notify_observation_missed();
  };

  auto clear_buff_plan_cache = [&]() {
    last_valid_buff_plan = {};
    has_last_valid_buff_plan = false;
    last_valid_buff_plan_t = {};
  };

  auto cached_buff_plan = [&]() {
    auto_aim::Plan plan{};
    const auto now = std::chrono::steady_clock::now();
    if (!has_last_valid_buff_plan || now - last_valid_buff_plan_t > buff_plan_hold_time) {
      has_last_valid_buff_plan = false;
      return plan;
    }

    plan = last_valid_buff_plan;
    plan.fire = false;
    return plan;
  };

  auto set_exposure = [&](SentryMode current_mode) {
    const bool buff_mode =
      current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF;
    const double desired = buff_mode ? buff_exposure_ms : exposure_ms;
    if (desired != current_exposure_ms) {
      main_camera.set_exposure_ms(desired);
      current_exposure_ms = desired;
    }
    return desired;
  };

  while (!exiter.exit()) {
    const auto nav_gs = ros2.subscribe_gimbal_state();
    const auto current_mode = nav_gs ? sentry_mode(nav_gs->mode) : SentryMode::IDLE;
    mode = current_mode;

    if (current_mode != last_mode) {
      tools::logger()->info("[SentryMPCDebug] Switch to {}", mode_name(current_mode));
      last_mode = current_mode;
      target_queue.push({});
      store_plan({});
      main_auto_aim_debug = {};
      discard_pending_yolo_results();
      if (current_mode == SentryMode::AUTO_AIM) {
        auto_aim_mode_enter_t = std::chrono::steady_clock::now();
        has_last_auto_aim_result_t = false;
        discard_pending_buff_results();
        clear_buff_plan_cache();
      } else if (current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF) {
        discard_pending_buff_results();
        has_last_buff_result_t = false;
        clear_buff_plan_cache();
        publish_empty_tracker();
      } else {
        discard_pending_buff_results();
        clear_buff_plan_cache();
        publish_empty_tracker();
      }
      ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
    }

    const double active_exposure_ms = set_exposure(current_mode);
    main_camera.read(img, timestamp);
    if (!buff_time_origin_ready) {
      buff_time_origin = timestamp;
      buff_time_origin_ready = true;
    }
    auto img_time = timestamp - exposure_center_offset(active_exposure_ms);
    cv::Mat display = img.clone();
    std::list<auto_aim::Armor> detections;
    std::optional<auto_aim::Target> current_target;
    double detect_ms = 0.0;

    if (!nav_gs) {
      discard_pending_yolo_results();
      discard_pending_buff_results();
      clear_buff_plan_cache();
      target_queue.push({});
      store_plan({});
      publish_empty_tracker();
      ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
    } else if (current_mode == SentryMode::AUTO_AIM) {
      const auto img_q = ros2.subscribe_gimbal_q(img_time);
      if (img_q) {
        if (use_async_yolo) {
          const auto frame_count = main_yolo_frame_count++;
          auto * primary_yolo = main_yolo_instances[frame_count % 2];
          auto * fallback_yolo = main_yolo_instances[(frame_count + 1) % 2];
          if (
            primary_yolo->submit(img, frame_count, img_time) ||
            fallback_yolo->submit(img, frame_count, img_time)) {
            async_frame_qs[frame_count] = *img_q;
            while (async_frame_qs.size() > max_async_frame_qs) {
              async_frame_qs.erase(async_frame_qs.begin());
            }
          }

          std::vector<auto_aim::YOLOAsyncResult> ready_results;
          for (auto * yolo : main_yolo_instances) {
            auto_aim::YOLOAsyncResult yolo_result;
            while (yolo->try_fetch(yolo_result)) {
              ready_results.emplace_back(std::move(yolo_result));
            }
          }
          std::sort(
            ready_results.begin(), ready_results.end(),
            [](const auto & lhs, const auto & rhs) { return lhs.frame_count < rhs.frame_count; });

          for (auto & yolo_result : ready_results) {
            if (yolo_result.stamp < auto_aim_mode_enter_t) continue;
            if (has_last_auto_aim_result_t && yolo_result.stamp <= last_auto_aim_result_t) {
              continue;
            }

            auto frame_q_it = async_frame_qs.find(yolo_result.frame_count);
            if (frame_q_it == async_frame_qs.end()) continue;

            has_last_auto_aim_result_t = true;
            last_auto_aim_result_t = yolo_result.stamp;
            const auto result_q = frame_q_it->second;
            async_frame_qs.erase(async_frame_qs.begin(), std::next(frame_q_it));

            display = yolo_result.img.clone();
            detections = yolo_result.armors;
            detect_ms = yolo_result.detect_dt;

            tracker.set_enemy_color_from_camp(nav_gs->camp);
            solver.set_R_gimbal2world(result_q);
            auto targets = tracker.track(yolo_result.armors, yolo_result.stamp);
            if (targets.empty()) {
              current_target.reset();
              publish_empty_tracker();
            } else {
              current_target = targets.front();
            }
            target_queue.push({current_target});

            main_auto_aim_debug.img = yolo_result.img.clone();
            main_auto_aim_debug.armors = detections;
            main_auto_aim_debug.target = current_target;
            main_auto_aim_debug.q = result_q;
            main_auto_aim_debug.detect_ms = detect_ms;
            main_auto_aim_debug.valid = true;
          }
        } else {
          const auto detect_start = std::chrono::steady_clock::now();
          detections = main_yolo_primary.detect(img, main_yolo_frame_count++);
          const auto detect_end = std::chrono::steady_clock::now();
          detect_ms = tools::delta_time(detect_end, detect_start) * 1e3;

          auto tracking_armors = detections;
          tracker.set_enemy_color_from_camp(nav_gs->camp);
          solver.set_R_gimbal2world(*img_q);
          auto targets = tracker.track(tracking_armors, img_time);
          if (!targets.empty()) {
            current_target = targets.front();
          } else {
            publish_empty_tracker();
          }
          target_queue.push({current_target});

          main_auto_aim_debug.img = img.clone();
          main_auto_aim_debug.armors = detections;
          main_auto_aim_debug.target = current_target;
          main_auto_aim_debug.q = *img_q;
          main_auto_aim_debug.detect_ms = detect_ms;
          main_auto_aim_debug.valid = true;
        }
      } else {
        target_queue.push({});
        publish_empty_tracker();
      }
    } else if (current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF) {
      discard_pending_yolo_results();
      target_queue.push({});
      publish_empty_tracker();
      const auto img_q = ros2.subscribe_gimbal_q(img_time);
      auto_aim::Plan buff_plan{};
      bool has_fresh_buff_plan = false;
      if (img_q) {
        const auto frame_count = buff_frame_count++;
        if (buff_detector.submit(img, frame_count, img_time)) {
          buff_frame_qs[frame_count] = *img_q;
          while (buff_frame_qs.size() > max_async_frame_qs) {
            buff_frame_qs.erase(buff_frame_qs.begin());
          }
        }

        std::optional<auto_buff::PowerRune> power_runes;
        cv::Mat result_img;
        int result_frame_count = -1;
        double detect_dt_ms = 0.0;
        std::chrono::steady_clock::time_point result_t;
        while (buff_detector.fetch(
          power_runes, result_img, result_frame_count, detect_dt_ms, false, nullptr, &result_t)) {
          if (has_last_buff_result_t && result_t <= last_buff_result_t) continue;

          const auto frame_q_it = buff_frame_qs.find(result_frame_count);
          if (frame_q_it == buff_frame_qs.end()) {
            if (has_fresh_buff_plan) buff_plan.fire = false;
            buff_aimer.notify_observation_missed();
            continue;
          }

          const auto result_q = frame_q_it->second;
          buff_frame_qs.erase(buff_frame_qs.begin(), std::next(frame_q_it));
          has_last_buff_result_t = true;
          last_buff_result_t = result_t;
          detect_ms = detect_dt_ms;
          display = result_img.clone();

          buff_solver.set_R_gimbal2world(result_q);
          buff_solver.solve(power_runes,&result_t);
          if (power_runes) {
            const double observed_time_abs = tools::delta_time(result_t, buff_time_origin);
            const double now_time_abs =
              tools::delta_time(std::chrono::steady_clock::now(), buff_time_origin);
            power_runes->last_observed_time = observed_time_abs;
            const auto buff_mode = current_mode == SentryMode::SMALL_BUFF
                                     ? auto_buff::BuffMode::SMALL
                                     : auto_buff::BuffMode::BIG;
            auto plan = buff_aimer.mpc_aim(
              *power_runes, observed_time_abs, now_time_abs, to_gimbal_state(*nav_gs), buff_mode);
            if (plan.control) {
              buff_plan = plan;
              has_fresh_buff_plan = true;
              last_valid_buff_plan = plan;
              has_last_valid_buff_plan = true;
              last_valid_buff_plan_t = std::chrono::steady_clock::now();
            } else if (has_fresh_buff_plan) {
              buff_plan.fire = false;
            }

            tools::draw_point(display, power_runes->r_center, {255, 255, 0}, 6);
            for (const auto & blade : power_runes->fanblades) {
              tools::draw_points(display, blade.points, {0, 255, 0}, 2);
            }
          } else {
            if (has_fresh_buff_plan) buff_plan.fire = false;
            buff_aimer.notify_observation_missed();
          }
        }
      } else {
        discard_pending_buff_results();
        clear_buff_plan_cache();
      }
      if (!has_fresh_buff_plan) buff_plan = cached_buff_plan();
      ros2.publish_auto_aim_plan(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);
      PlanSnapshot snapshot;
      snapshot.plan = buff_plan;
      store_plan(snapshot);
    } else {
      discard_pending_yolo_results();
      discard_pending_buff_results();
      target_queue.push({});
      store_plan({});
      publish_empty_tracker();
      ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
    }

    if (current_mode == SentryMode::AUTO_AIM && main_auto_aim_debug.valid) {
      display = main_auto_aim_debug.img.clone();
      detections = main_auto_aim_debug.armors;
      current_target = main_auto_aim_debug.target;
      detect_ms = main_auto_aim_debug.detect_ms;
      solver.set_R_gimbal2world(main_auto_aim_debug.q);
    }

    draw_armors(display, detections);
    const auto plan_snapshot = load_plan();
    if (current_target) {
      for (const auto & xyza : current_target->armor_xyza_list()) {
        const auto points = solver.reproject_armor(
          xyza.head(3), xyza[3], current_target->armor_type, current_target->name);
        tools::draw_points(display, points, {0, 255, 255}, 2);
      }
      if (
        plan_snapshot.aim_valid && plan_snapshot.target &&
        plan_snapshot.target->name == current_target->name &&
        plan_snapshot.target->armor_type == current_target->armor_type) {
        const auto points = solver.reproject_armor(
          plan_snapshot.aim_xyza.head(3), plan_snapshot.aim_xyza[3], current_target->armor_type,
          current_target->name);
        tools::draw_points(display, points, {0, 0, 255}, 3);
      }
    }

    display_frames++;
    const auto now = std::chrono::steady_clock::now();
    const double fps_dt = tools::delta_time(now, fps_stamp);
    if (fps_dt >= 1.0) {
      display_fps = display_frames / fps_dt;
      display_frames = 0;
      fps_stamp = now;
    }

    draw_line(display, fmt::format("mode: {}", mode_name(current_mode)), 0);
    draw_line(
      display, nav_gs ? fmt::format("tracker: {}", tracker.state()) : "WAITING FOR NAV STATE", 1,
      nav_gs ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 255));
    draw_line(
      display,
      fmt::format(
        "detections: {}  detect: {:.1f} ms  display: {:.1f} fps", detections.size(), detect_ms,
        display_fps),
      2);
    draw_line(
      display,
      fmt::format(
        "control: {} fire: {} yaw: {:.3f} pitch: {:.3f}", plan_snapshot.plan.control,
        plan_snapshot.plan.fire, plan_snapshot.plan.yaw, plan_snapshot.plan.pitch),
      3, plan_snapshot.plan.fire ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 255, 255));
    if (nav_gs) {
      draw_line(
        display,
        fmt::format(
          "gimbal yaw: {:.3f} pitch: {:.3f} bullet: {:.1f} camp: {}", nav_gs->yaw, nav_gs->pitch,
          nav_gs->bullet_speed, nav_gs->camp),
        4);
    }
    if (current_mode == SentryMode::AUTO_AIM) {
      draw_line(display, "reprojection: yellow=EKF  red=aim", 5, {0, 255, 255});
    }

    if (!omni_debug_queue.empty()) omni_debug = omni_debug_queue.pop();
    cv::Mat omni_display;
    if (!omni_enabled) {
      omni_display = make_status_image("OMNI CAMERA DISABLED");
    } else if (!omni_debug.active || omni_debug.img.empty()) {
      omni_display = make_status_image("OMNI INACTIVE IN THIS MODE");
    } else {
      omni_display = omni_debug.img.clone();
      draw_armors(omni_display, omni_debug.armors);
      const cv::Scalar status_color =
        omni_debug.status == 2 ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);
      draw_line(
        omni_display,
        fmt::format("omni status: {}  detections: {}", omni_debug.status, omni_debug.armors.size()),
        0, status_color);
    }

    nlohmann::json plot_data;
    plot_data["mode"] = static_cast<int>(current_mode);
    plot_data["armor_num"] = detections.size();
    plot_data["detect_ms"] = detect_ms;
    plot_data["display_fps"] = display_fps;
    plot_data["omni_status"] = omni_debug.status;
    plot_data["plan_yaw"] = plan_snapshot.plan.yaw;
    plot_data["plan_yaw_vel"] = plan_snapshot.plan.yaw_vel;
    plot_data["plan_yaw_acc"] = plan_snapshot.plan.yaw_acc;
    plot_data["plan_pitch"] = plan_snapshot.plan.pitch;
    plot_data["plan_pitch_vel"] = plan_snapshot.plan.pitch_vel;
    plot_data["plan_pitch_acc"] = plan_snapshot.plan.pitch_acc;
    plot_data["control"] = plan_snapshot.plan.control ? 1 : 0;
    plot_data["fire"] = plan_snapshot.plan.fire ? 1 : 0;
    if (nav_gs) {
      plot_data["gimbal_yaw"] = nav_gs->yaw;
      plot_data["gimbal_yaw_vel"] = nav_gs->yaw_vel;
      plot_data["gimbal_pitch"] = nav_gs->pitch;
      plot_data["gimbal_pitch_vel"] = nav_gs->pitch_vel;
      plot_data["bullet_speed"] = nav_gs->bullet_speed;
    }
    if (plan_snapshot.target) {
      const auto x = plan_snapshot.target->ekf_x();
      plot_data["target_x"] = x[0];
      plot_data["target_vx"] = x[1];
      plot_data["target_y"] = x[2];
      plot_data["target_vy"] = x[3];
      plot_data["target_z"] = x[4];
      plot_data["target_vz"] = x[5];
      plot_data["target_yaw"] = x[6];
      plot_data["target_yaw_vel"] = x[7];
      plot_data["target_radius"] = x[8];
      plot_data["target_last_id"] = plan_snapshot.target->last_id;

      const auto & ekf_data = plan_snapshot.target->ekf().data;
      auto add_ekf_value = [&](const std::string & key) {
        const auto it = ekf_data.find(key);
        if (it != ekf_data.end()) plot_data[key] = it->second;
      };
      add_ekf_value("residual_yaw");
      add_ekf_value("residual_pitch");
      add_ekf_value("residual_distance");
      add_ekf_value("residual_angle");
      add_ekf_value("nis");
      add_ekf_value("process_v1");
      add_ekf_value("process_v2");
      add_ekf_value("adaptive_v1_boost");
      add_ekf_value("adaptive_v2_boost");
    }
    plotter.plot(plot_data);

    cv::resize(display, display, {}, 0.6, 0.6);
    if (omni_display.cols > 800) cv::resize(omni_display, omni_display, {}, 0.6, 0.6);
    cv::imshow("sentry_mpc/main", display);
    cv::imshow("sentry_mpc/omni", omni_display);
    const int key = cv::waitKey(1);
    if (key == 'q' || key == 27) break;
  }

  quit = true;
  if (omni_thread.joinable()) omni_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
  ros2.publish_omni_target(0);
  publish_empty_tracker();
  cv::destroyAllWindows();
  return 0;
}
