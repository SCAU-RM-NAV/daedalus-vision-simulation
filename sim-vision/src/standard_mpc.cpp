#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/runtime.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

constexpr std::uintmax_t kMinimumRecordFreeSpace = 10ull * 1024 * 1024 * 1024;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::RecorderOptions recorder_options;
  recorder_options.rotation_slot_count = 30;
  recorder_options.min_available_space_bytes = kMinimumRecordFreeSpace;
  tools::Recorder recorder(recorder_options);

  auto yaml = tools::load(config_path);
  const YAML::Node combat_camera_yaml = yaml["combat_camera"] ? yaml["combat_camera"] : yaml;
  const YAML::Node idle_camera_yaml = yaml["idle_camera"];
  const bool has_idle_camera = static_cast<bool>(idle_camera_yaml);

  const auto exposure_ms = tools::read<double>(combat_camera_yaml, "exposure_ms");
  const auto buff_exposure_ms = combat_camera_yaml["buff_exposure"]
                                  ? combat_camera_yaml["buff_exposure"].as<double>()
                                  : exposure_ms;
  auto exposure_center_offset = [](double exposure_ms) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  };
  tools::logger()->info(
    "Image q timestamp offset: auto aim -{:.3f} ms, buff -{:.3f} ms (exposure center)",
    exposure_ms * 0.5, buff_exposure_ms * 0.5);

  io::Gimbal gimbal(config_path);
  const auto initial_mode = gimbal.mode();
  io::Camera camera(combat_camera_yaml, !has_idle_camera || initial_mode != io::GimbalMode::IDLE);
  std::optional<io::Camera> idle_camera;
  if (has_idle_camera) {
    idle_camera.emplace(idle_camera_yaml, initial_mode == io::GimbalMode::IDLE);
    tools::logger()->info("Dual camera enabled: idle mode uses the long-focus camera.");
  }
  io::Camera * active_camera = &camera;
  if (has_idle_camera && initial_mode == io::GimbalMode::IDLE) active_camera = &idle_camera.value();

  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Runtime auto_aim_runtime(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  const bool has_buff =
    !yaml["trajectory_model"] || yaml["trajectory_model"].as<std::string>() != "hero";

  std::optional<auto_buff::Buff_Detector> buff_detector;
  std::optional<auto_buff::Solver> buff_solver;
  std::optional<auto_buff::Aimer> buff_aimer;
  if (has_buff) {
    buff_detector.emplace(config_path);
    buff_solver.emplace(config_path);
    buff_aimer.emplace(config_path);
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point buff_time_origin;
  bool buff_time_origin_ready = false;

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};
  auto fps_report_stamp = std::chrono::steady_clock::now();
  double detect_time_sum = 0.0;
  int fps_frame_count = 0;

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
  auto auto_aim_mode_enter_t = std::chrono::steady_clock::now();
  constexpr size_t max_async_frame_qs = 128;
  std::map<int, Eigen::Quaterniond> async_frame_qs;
  std::map<int, Eigen::Quaterniond> buff_frame_qs;

  const bool use_async_yolo = yolo.supports_async();
  tools::logger()->info(
    "YOLO pipeline mode: {}", use_async_yolo ? "OpenVINO async thread-pool" : "sync fallback");

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
        "detect avg: {:.2f} fps, loop: {:.2f} fps", fps_frame_count / detect_time_sum,
        fps_frame_count / report_dt);
      fps_report_stamp = now;
      detect_time_sum = 0.0;
      fps_frame_count = 0;
    }
  };

  auto update_async_fps = [&](const std::chrono::steady_clock::time_point & now) {
    auto report_dt = tools::delta_time(now, fps_report_stamp);
    if (report_dt < 1.0) return;

    // if (async_result_count > 0) {
    //   tools::logger()->info(
    //     "async submit: {:.2f} fps, async result: {:.2f} fps, avg latency: {:.2f} ms, busy_drop: {}, stale_drop: {}",
    //     async_submitted_count / report_dt, async_result_count / report_dt,
    //     async_detect_time_sum / async_result_count, async_busy_drop_count, async_stale_drop_count);
    // } else {
    //   tools::logger()->info(
    //     "async submit: {:.2f} fps, async result: 0.00 fps, busy_drop: {}, stale_drop: {}",
    //     async_submitted_count / report_dt, async_busy_drop_count, async_stale_drop_count);
    // }

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
    while (yolo.try_fetch(dropped_result)) {
      async_stale_drop_count++;
    }
    async_frame_qs.clear();
  };

  auto discard_pending_buff_results = [&]() {
    if (!has_buff) return;
    async_stale_drop_count += static_cast<int>(buff_detector->discard_pending_results());
    buff_frame_qs.clear();
    buff_aimer->notify_observation_missed();
  };

  auto activate_camera_for_mode = [&](io::GimbalMode current_mode) -> io::Camera & {
    auto * desired_camera = &camera;
    if (current_mode == io::GimbalMode::IDLE && idle_camera.has_value()) {
      desired_camera = &idle_camera.value();
    }

    if (active_camera != desired_camera) {
      active_camera->set_active(false);
      desired_camera->set_active(true);
      active_camera = desired_camera;
      tools::logger()->info(
        "Active camera: {}", desired_camera == &camera ? "combat" : "idle-long-focus");
    }

    return *active_camera;
  };

  auto exposure_for_mode = [&](io::GimbalMode current_mode) {
    if (current_mode == io::GimbalMode::SMALL_BUFF || current_mode == io::GimbalMode::BIG_BUFF) {
      return buff_exposure_ms;
    }

    return exposure_ms;
  };

  double current_combat_exposure_ms = -1.0;
  auto set_combat_camera_exposure = [&](io::GimbalMode current_mode) {
    const auto desired_exposure_ms = exposure_for_mode(current_mode);
    if (current_combat_exposure_ms != desired_exposure_ms) {
      camera.set_exposure_ms(desired_exposure_ms);
      current_combat_exposure_ms = desired_exposure_ms;
      tools::logger()->info("Set combat camera exposure: {:.3f} ms", desired_exposure_ms);
    }
    return desired_exposure_ms;
  };

  auto process_auto_aim_detection = [&](std::list<auto_aim::Armor> armors,
                                        const std::chrono::steady_clock::time_point & img_time,
                                        const Eigen::Quaterniond & img_q) {
    auto gs = gimbal.state();

    auto_aim::RuntimeFrame runtime_frame;
    runtime_frame.timestamp = img_time;
    runtime_frame.imu_quaternion = img_q;
    runtime_frame.camp = gs.camp;
    auto targets = auto_aim_runtime.track(armors, runtime_frame);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);
  };

  auto plan_thread = std::thread([&]() {
    const auto plan_send_period = std::chrono::microseconds(10000);  // ~100 Hz
    auto next_send = std::chrono::steady_clock::now();

    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        next_send += plan_send_period;

        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = auto_aim_runtime.plan(target, gs.bullet_speed, gs.pitch);

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        auto now = std::chrono::steady_clock::now();
        if (now > next_send + plan_send_period) {
          next_send = now;
        }
        std::this_thread::sleep_until(next_send);
      } else {
        next_send = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(20ms);
      }
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
      reset_fps();

      if (mode.load() == io::GimbalMode::AUTO_AIM) {
        auto_aim_mode_enter_t = std::chrono::steady_clock::now();
        has_last_auto_aim_result_t = false;
        discard_pending_buff_results();
        if (buff_aimer) buff_aimer->reset();
      } else if (
        mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
        target_queue.push(std::nullopt);
        discard_pending_yolo_results();
        discard_pending_buff_results();
        if (buff_aimer) buff_aimer->reset();
        has_last_buff_result_t = false;
      } else {
        target_queue.push(std::nullopt);
        discard_pending_yolo_results();
        discard_pending_buff_results();
        if (buff_aimer) buff_aimer->reset();
      }
    }

    const auto current_mode = mode.load();
    auto & frame_camera = activate_camera_for_mode(current_mode);
    const auto current_exposure_ms = (&frame_camera == &camera)
                                       ? set_combat_camera_exposure(current_mode)
                                       : exposure_ms;
    frame_camera.read(img, t);
    if (!buff_time_origin_ready) {
      buff_time_origin = t;
      buff_time_origin_ready = true;
    }
    auto img_time = t - exposure_center_offset(current_exposure_ms);

    if (current_mode == io::GimbalMode::IDLE) {
      discard_pending_yolo_results();
      target_queue.push(std::nullopt);
      recorder.record(img, Eigen::Quaterniond::Identity(), img_time);
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
      continue;
    }
    auto img_q = gimbal.q(img_time);
    auto gs = gimbal.state();
    recorder.record(img, img_q, img_time);

    /// 自瞄
    if (current_mode == io::GimbalMode::AUTO_AIM) {
      if (use_async_yolo) {
        const auto frame_count = yolo_frame_count++;
        if (yolo.submit(img, frame_count, img_time)) {
          async_submitted_count++;
          async_frame_qs[frame_count] = img_q;
          while (async_frame_qs.size() > max_async_frame_qs) {
            async_frame_qs.erase(async_frame_qs.begin());
          }
        } else {
          async_busy_drop_count++;
        }

        auto_aim::YOLOAsyncResult yolo_result;
        while (yolo.try_fetch(yolo_result)) {
          // 多 InferRequest 异步完成顺序不保证；tracker 只能吃时间单调递增的结果。
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

          auto img_q = frame_q_it->second;
          async_frame_qs.erase(async_frame_qs.begin(), std::next(frame_q_it));
          process_auto_aim_detection(std::move(yolo_result.armors), yolo_result.stamp, img_q);
        }

        update_async_fps(std::chrono::steady_clock::now());
      } else {
        auto detect_start = std::chrono::steady_clock::now();
        auto armors = yolo.detect(img, yolo_frame_count++);
        auto detect_end = std::chrono::steady_clock::now();

        process_auto_aim_detection(std::move(armors), img_time, img_q);
        update_fps(
          std::chrono::steady_clock::now(), tools::delta_time(detect_end, detect_start));
      }
    }

    /// 打符
    else if (
      (current_mode == io::GimbalMode::SMALL_BUFF || current_mode == io::GimbalMode::BIG_BUFF) &&
      has_buff) {
      discard_pending_yolo_results();
      const auto frame_count = buff_frame_count++;
      if (buff_detector->submit(img, frame_count, img_time)) {
        async_submitted_count++;
        buff_frame_qs[frame_count] = img_q;
        while (buff_frame_qs.size() > max_async_frame_qs) {
          buff_frame_qs.erase(buff_frame_qs.begin());
          async_stale_drop_count++;
        }
      } else {
        async_busy_drop_count++;
      }

      auto_aim::Plan buff_plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
      std::optional<auto_buff::PowerRune> power_runes;
      cv::Mat result_img;
      int result_frame_count = -1;
      double detect_dt_ms = 0.0;
      std::chrono::steady_clock::time_point result_t;
      while (buff_detector->fetch(
        power_runes, result_img, result_frame_count, detect_dt_ms, false, nullptr, &result_t)) {
        if (has_last_buff_result_t && result_t <= last_buff_result_t) {
          async_stale_drop_count++;
          continue;
        }

        const auto frame_q_it = buff_frame_qs.find(result_frame_count);
        if (frame_q_it == buff_frame_qs.end()) {
          async_stale_drop_count++;
          buff_aimer->notify_observation_missed();
          continue;
        }

        const auto result_q = frame_q_it->second;
        buff_frame_qs.erase(buff_frame_qs.begin(), std::next(frame_q_it));
        has_last_buff_result_t = true;
        last_buff_result_t = result_t;
        async_result_count++;
        async_detect_time_sum += detect_dt_ms;

        buff_solver->set_R_gimbal2world(result_q);
        buff_solver->solve(power_runes,&result_t);
        if (power_runes) {
          const double observed_time_abs = tools::delta_time(result_t, buff_time_origin);
          const double now_time_abs =
            tools::delta_time(std::chrono::steady_clock::now(), buff_time_origin);
          power_runes->last_observed_time = observed_time_abs;
          const auto buff_mode = current_mode == io::GimbalMode::SMALL_BUFF
                                   ? auto_buff::BuffMode::SMALL
                                   : auto_buff::BuffMode::BIG;
          buff_plan =
            buff_aimer->mpc_aim(*power_runes, observed_time_abs, now_time_abs, gs, buff_mode);
        } else {
          buff_aimer->notify_observation_missed();
        }

        gimbal.send(
          buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
          buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);
      }
      update_async_fps(std::chrono::steady_clock::now());

    } else {
      discard_pending_yolo_results();
      discard_pending_buff_results();
      target_queue.push(std::nullopt);
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
    }
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
