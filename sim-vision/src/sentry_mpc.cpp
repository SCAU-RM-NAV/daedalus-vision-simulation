#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <memory>
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
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明 }"
  "{@config-path   | configs/sentry.yaml | yaml配置文件路径 }";

namespace
{
struct AimInput
{
  std::optional<auto_aim::Target> target;
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

}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  tools::Exiter exiter;
  tools::Recorder recorder;

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
  const auto omni_frame_rate =
    omni_camera_yaml["frame_rate"] ? omni_camera_yaml["frame_rate"].as<double>() : 0.0;
  const auto omni_frame_period =
    omni_frame_rate > 0 ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(1.0 / omni_frame_rate))
                        : std::chrono::steady_clock::duration::zero();
  if (omni_enabled && omni_frame_rate > 0) {
    tools::logger()->info("Sentry omni perception frame rate limit: {:.2f} fps", omni_frame_rate);
  }

  const auto exposure_ms = tools::read<double>(combat_camera_yaml, "exposure_ms");
  const auto buff_exposure_ms = combat_camera_yaml["buff_exposure"]
                                  ? combat_camera_yaml["buff_exposure"].as<double>()
                                  : exposure_ms;
  auto exposure_center_offset = [](double exposure_ms) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  };
  tools::logger()->info(
    "Sentry main camera timestamp offset: auto aim -{:.3f} ms, buff -{:.3f} ms "
    "(exposure center)",
    exposure_ms * 0.5, buff_exposure_ms * 0.5);

  io::ROS2 ros2;
  io::Camera main_camera(combat_camera_yaml, true);

  auto_aim::YOLO main_yolo_primary(config_path, true);
  auto_aim::YOLO main_yolo_secondary(config_path, true);
  auto_aim::YOLO * main_yolo_instances[] = {&main_yolo_primary, &main_yolo_secondary};
  auto_aim::Solver solver(config_path);
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
    tools::logger()->warn("Sentry omni camera is disabled by config.");
    ros2.publish_omni_target(0);
  }

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::Aimer buff_aimer(config_path);
  buff_aimer.set_mpc_fire_gap_time_override(sentry_mpc_fire_gap_time);
  tools::logger()->info(
    "[SentryMPC] Buff MPC fire interval: {:.3f} s", sentry_mpc_fire_gap_time);

  tools::ThreadSafeQueue<AimInput, true> target_queue(1);
  target_queue.push({});

  std::atomic<bool> quit = false;
  std::atomic<SentryMode> mode{SentryMode::IDLE};
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

      auto input = target_queue.front();
      const auto nav_gs = ros2.subscribe_gimbal_state();
      if (!nav_gs) {
        ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
        publish_empty_tracker();
        auto now = std::chrono::steady_clock::now();
        if (now > next_send + plan_send_period) next_send = now;
        std::this_thread::sleep_until(next_send);
        continue;
      }

      auto plan = planner.plan(input.target, nav_gs->bullet_speed, nav_gs->pitch);

      ros2.publish_auto_aim_plan(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);
      publish_tracker_plan(input.target, plan.yaw, plan.pitch);

      auto now = std::chrono::steady_clock::now();
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
      bool omni_active = false;
      auto next_frame_time = std::chrono::steady_clock::now();

      while (!quit) {
        if (mode.load() != SentryMode::AUTO_AIM) {
          if (omni_active) {
            omni_camera->set_active(false);
            omni_status_decider->reset();
            ros2.publish_omni_target(0);
            omni_active = false;
          }
          next_frame_time = std::chrono::steady_clock::now();
          std::this_thread::sleep_for(20ms);
          continue;
        }

        if (!omni_active) {
          omni_camera->set_active(true);
          omni_status_decider->reset();
          omni_active = true;
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
        const int omni_status =
          omni_status_decider->update(std::move(armors), nav_gs ? nav_gs->camp : 0xff);
        ros2.publish_omni_target(omni_status);
      }

      if (omni_active) omni_camera->set_active(false);
    });
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point buff_time_origin;
  bool buff_time_origin_ready = false;

  int yolo_frame_count = 0;
  int async_submitted_count = 0;
  int async_result_count = 0;
  int async_busy_drop_count = 0;
  int async_stale_drop_count = 0;
  double async_detect_time_sum = 0.0;
  bool has_last_auto_aim_result_t = false;
  auto last_auto_aim_result_t = std::chrono::steady_clock::time_point{};
  int buff_frame_count = 0;
  bool has_last_buff_result_t = false;
  auto last_buff_result_t = std::chrono::steady_clock::time_point{};
  auto_aim::Plan last_valid_buff_plan{};
  bool has_last_valid_buff_plan = false;
  auto last_valid_buff_plan_t = std::chrono::steady_clock::time_point{};
  auto auto_aim_mode_enter_t = std::chrono::steady_clock::now();
  auto fps_report_stamp = std::chrono::steady_clock::now();
  double detect_time_sum = 0.0;
  int fps_frame_count = 0;
  constexpr size_t max_async_frame_qs = 128;
  std::map<int, Eigen::Quaterniond> async_frame_qs;
  std::map<int, Eigen::Quaterniond> buff_frame_qs;
  auto last_mode = SentryMode::IDLE;

  const bool use_async_yolo =
    main_yolo_primary.supports_async() && main_yolo_secondary.supports_async();
  tools::logger()->info(
    "Sentry main YOLO pipeline mode: {}",
    use_async_yolo ? "OpenVINO async dual-instance" : "sync fallback");

  auto reset_fps = [&]() {
    fps_report_stamp = std::chrono::steady_clock::now();
    detect_time_sum = 0.0;
    fps_frame_count = 0;
    async_submitted_count = 0;
    async_result_count = 0;
    async_busy_drop_count = 0;
    async_stale_drop_count = 0;
    async_detect_time_sum = 0.0;
  };

  auto update_fps = [&](const std::chrono::steady_clock::time_point & now, double detect_dt) {
    detect_time_sum += detect_dt;
    fps_frame_count++;

    auto report_dt = tools::delta_time(now, fps_report_stamp);
    if (report_dt >= 1.0 && detect_time_sum > 0.0) {
      tools::logger()->info(
        "sentry main detect avg: {:.2f} fps, loop: {:.2f} fps", fps_frame_count / detect_time_sum,
        fps_frame_count / report_dt);
      fps_report_stamp = now;
      detect_time_sum = 0.0;
      fps_frame_count = 0;
    }
  };

  auto update_async_fps = [&](const std::chrono::steady_clock::time_point & now) {
    auto report_dt = tools::delta_time(now, fps_report_stamp);
    if (report_dt < 1.0) return;

    if (async_result_count > 0) {
      tools::logger()->info(
        "sentry main async submit: {:.2f} fps, result: {:.2f} fps, latency: {:.2f} ms, "
        "busy_drop: {}, stale_drop: {}",
        async_submitted_count / report_dt, async_result_count / report_dt,
        async_detect_time_sum / async_result_count, async_busy_drop_count, async_stale_drop_count);
    }

    fps_report_stamp = now;
    async_submitted_count = 0;
    async_result_count = 0;
    async_busy_drop_count = 0;
    async_stale_drop_count = 0;
    async_detect_time_sum = 0.0;
  };

  auto discard_pending_yolo_results = [&]() {
    if (!use_async_yolo) return;

    auto_aim::YOLOAsyncResult dropped_result;
    for (auto * yolo : main_yolo_instances) {
      while (yolo->try_fetch(dropped_result)) {
        async_stale_drop_count++;
      }
    }
    async_frame_qs.clear();
  };

  auto discard_pending_buff_results = [&]() {
    async_stale_drop_count += static_cast<int>(buff_detector.discard_pending_results());
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

  auto publish_stop_plan = [&]() { ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0); };

  auto warn_waiting_gimbal_state = [&]() {
    static auto last_warn_time = std::chrono::steady_clock::time_point{};
    auto now = std::chrono::steady_clock::now();
    if (
      last_warn_time == std::chrono::steady_clock::time_point{} ||
      tools::delta_time(now, last_warn_time) > 1.0) {
      tools::logger()->warn("[SentryMPC] Waiting for sentry_gimbal_state from nav.");
      last_warn_time = now;
    }
  };

  auto exposure_for_mode = [&](SentryMode current_mode) {
    if (current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF) {
      return buff_exposure_ms;
    }

    return exposure_ms;
  };

  double current_main_exposure_ms = -1.0;
  auto set_main_camera_exposure = [&](SentryMode current_mode) {
    const auto desired_exposure_ms = exposure_for_mode(current_mode);
    if (current_main_exposure_ms != desired_exposure_ms) {
      main_camera.set_exposure_ms(desired_exposure_ms);
      current_main_exposure_ms = desired_exposure_ms;
      tools::logger()->info("Set sentry main camera exposure: {:.3f} ms", desired_exposure_ms);
    }
    return desired_exposure_ms;
  };

  auto process_auto_aim_detection = [&](
                                      std::list<auto_aim::Armor> armors,
                                      const std::chrono::steady_clock::time_point & img_time,
                                      const Eigen::Quaterniond & img_q) {
    const auto nav_gs = ros2.subscribe_gimbal_state();

    tracker.set_enemy_color_from_camp(nav_gs ? nav_gs->camp : 0xff);
    solver.set_R_gimbal2world(img_q);

    auto targets = tracker.track(armors, img_time);
    if (!targets.empty()) {
      target_queue.push({targets.front()});
      // std::cout << "target armor id: " << static_cast<int>(targets.front().name) + 1
      //           << ", tracker panel index: " << targets.front().last_id << std::endl;
    } else {
      target_queue.push({});
      publish_empty_tracker();
    }
  };

  while (!exiter.exit()) {
    const auto nav_gs = ros2.subscribe_gimbal_state();
    if (!nav_gs) {
      mode = SentryMode::IDLE;
      target_queue.push({});
      discard_pending_yolo_results();
      discard_pending_buff_results();
      clear_buff_plan_cache();
      publish_empty_tracker();
      publish_stop_plan();
      warn_waiting_gimbal_state();
      std::this_thread::sleep_for(20ms);
      continue;
    }

    const auto current_mode = sentry_mode(nav_gs->mode);
    mode = current_mode;
    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", mode_name(current_mode));
      last_mode = current_mode;
      reset_fps();

      if (current_mode == SentryMode::AUTO_AIM) {
        auto_aim_mode_enter_t = std::chrono::steady_clock::now();
        has_last_auto_aim_result_t = false;
        discard_pending_buff_results();
        clear_buff_plan_cache();
      } else if (current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF) {
        target_queue.push({});
        discard_pending_yolo_results();
        discard_pending_buff_results();
        has_last_buff_result_t = false;
        clear_buff_plan_cache();
        publish_empty_tracker();
        publish_stop_plan();
      } else {
        target_queue.push({});
        discard_pending_yolo_results();
        discard_pending_buff_results();
        clear_buff_plan_cache();
        publish_empty_tracker();
        publish_stop_plan();
      }
    }

    const auto current_exposure_ms = set_main_camera_exposure(current_mode);
    main_camera.read(img, t);
    if (!buff_time_origin_ready) {
      buff_time_origin = t;
      buff_time_origin_ready = true;
    }
    auto img_time = t - exposure_center_offset(current_exposure_ms);

    if (current_mode == SentryMode::IDLE) {
      discard_pending_yolo_results();
      discard_pending_buff_results();
      clear_buff_plan_cache();
      target_queue.push({});
      recorder.record(img, nav_gs->q, img_time);
      publish_empty_tracker();
      publish_stop_plan();
      continue;
    }

    auto img_q = ros2.subscribe_gimbal_q(img_time);
    if (!img_q) {
      if (current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF) {
        discard_pending_buff_results();
        clear_buff_plan_cache();
      }
      target_queue.push({});
      publish_empty_tracker();
      publish_stop_plan();
      warn_waiting_gimbal_state();
      continue;
    }
    // recorder.record(img, *img_q, img_time);

    if (current_mode == SentryMode::AUTO_AIM && use_async_yolo) {
      const auto frame_count = yolo_frame_count++;
      auto * primary_yolo = main_yolo_instances[frame_count % 2];
      auto * fallback_yolo = main_yolo_instances[(frame_count + 1) % 2];
      if (
        primary_yolo->submit(img, frame_count, img_time) ||
        fallback_yolo->submit(img, frame_count, img_time)) {
        async_submitted_count++;
        async_frame_qs[frame_count] = *img_q;
        while (async_frame_qs.size() > max_async_frame_qs) {
          async_frame_qs.erase(async_frame_qs.begin());
        }
      } else {
        async_busy_drop_count++;
      }

      std::vector<auto_aim::YOLOAsyncResult> ready_results;
      for (auto * yolo : main_yolo_instances) {
        auto_aim::YOLOAsyncResult yolo_result;
        while (yolo->try_fetch(yolo_result)) {
          ready_results.emplace_back(std::move(yolo_result));
        }
      }
      std::sort(ready_results.begin(), ready_results.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.frame_count < rhs.frame_count;
      });

      for (auto & yolo_result : ready_results) {
        if (yolo_result.stamp < auto_aim_mode_enter_t) {
          async_stale_drop_count++;
          continue;
        }

        if (has_last_auto_aim_result_t && yolo_result.stamp <= last_auto_aim_result_t) {
          async_stale_drop_count++;
          continue;
        }

        has_last_auto_aim_result_t = true;
        last_auto_aim_result_t = yolo_result.stamp;
        async_result_count++;
        async_detect_time_sum += yolo_result.detect_dt;

        auto frame_q_it = async_frame_qs.find(yolo_result.frame_count);
        if (frame_q_it == async_frame_qs.end()) {
          async_stale_drop_count++;
          continue;
        }

        auto result_q = frame_q_it->second;
        async_frame_qs.erase(async_frame_qs.begin(), std::next(frame_q_it));
        process_auto_aim_detection(std::move(yolo_result.armors), yolo_result.stamp, result_q);
      }

      update_async_fps(std::chrono::steady_clock::now());
    } else if (current_mode == SentryMode::AUTO_AIM) {
      auto detect_start = std::chrono::steady_clock::now();
      auto armors = main_yolo_primary.detect(img, yolo_frame_count++);
      auto detect_end = std::chrono::steady_clock::now();

      process_auto_aim_detection(std::move(armors), img_time, *img_q);
      update_fps(std::chrono::steady_clock::now(), tools::delta_time(detect_end, detect_start));
    } else if (current_mode == SentryMode::SMALL_BUFF || current_mode == SentryMode::BIG_BUFF) {
      discard_pending_yolo_results();
      target_queue.push({});
      publish_empty_tracker();
      const auto frame_count = buff_frame_count++;
      if (buff_detector.submit(img, frame_count, img_time)) {
        async_submitted_count++;
        buff_frame_qs[frame_count] = *img_q;
        while (buff_frame_qs.size() > max_async_frame_qs) {
          buff_frame_qs.erase(buff_frame_qs.begin());
          async_stale_drop_count++;
        }
      } else {
        async_busy_drop_count++;
      }

      auto_aim::Plan buff_plan{};
      bool has_fresh_buff_plan = false;
      std::optional<auto_buff::PowerRune> power_runes;
      cv::Mat result_img;
      int result_frame_count = -1;
      double detect_dt_ms = 0.0;
      std::chrono::steady_clock::time_point result_t;
      while (buff_detector.fetch(
        power_runes, result_img, result_frame_count, detect_dt_ms, false, nullptr, &result_t)) {
        if (has_last_buff_result_t && result_t <= last_buff_result_t) {
          async_stale_drop_count++;
          continue;
        }

        const auto frame_q_it = buff_frame_qs.find(result_frame_count);
        if (frame_q_it == buff_frame_qs.end()) {
          async_stale_drop_count++;
          if (has_fresh_buff_plan) buff_plan.fire = false;
          buff_aimer.notify_observation_missed();
          continue;
        }

        const auto result_q = frame_q_it->second;
        buff_frame_qs.erase(buff_frame_qs.begin(), std::next(frame_q_it));
        has_last_buff_result_t = true;
        last_buff_result_t = result_t;
        async_result_count++;
        async_detect_time_sum += detect_dt_ms;

        buff_solver.set_R_gimbal2world(result_q);
        buff_solver.solve(power_runes,&result_t);
        if (power_runes) {
          const double observed_time_abs = tools::delta_time(result_t, buff_time_origin);
          const double now_time_abs =
            tools::delta_time(std::chrono::steady_clock::now(), buff_time_origin);
          power_runes->last_observed_time = observed_time_abs;
          const auto buff_mode = current_mode == SentryMode::SMALL_BUFF ? auto_buff::BuffMode::SMALL
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
        } else {
          if (has_fresh_buff_plan) buff_plan.fire = false;
          buff_aimer.notify_observation_missed();
        }
      }

      if (!has_fresh_buff_plan) buff_plan = cached_buff_plan();
      ros2.publish_auto_aim_plan(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, 0, 0, buff_plan.pitch, 0, 0);
      update_async_fps(std::chrono::steady_clock::now());
    } else {
      discard_pending_yolo_results();
      discard_pending_buff_results();
      target_queue.push({});
      publish_empty_tracker();
      publish_stop_plan();
    }
  }

  quit = true;
  if (omni_thread.joinable()) omni_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  ros2.publish_auto_aim_plan(false, false, 0, 0, 0, 0, 0, 0);
  ros2.publish_omni_target(0);
  publish_empty_tracker();

  return 0;
}
