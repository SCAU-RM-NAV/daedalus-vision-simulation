#include <atomic>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <list>
#include <map>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "hero_deploy_vt/DeployVtSender.hpp"
#include "io/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "io/gimbal/gimbal.hpp"
#include <rclcpp/rclcpp.hpp>
#include <robot_msgs/msg/remote_hero_fdb.hpp>
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

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
  tools::Recorder recorder;

  auto yaml = tools::load(config_path);
  const auto navi_yaw_offset = tools::read<double>(yaml, "navi_yaw_offset");
  const auto navi_pitch_offset = tools::read<double>(yaml, "navi_pitch_offset");
  const YAML::Node combat_camera_yaml = yaml["combat_camera"] ? yaml["combat_camera"] : yaml;
  const YAML::Node idle_camera_yaml = yaml["idle_camera"];
  const bool has_idle_camera = static_cast<bool>(idle_camera_yaml);
  const auto combat_exposure_ms = tools::read<double>(combat_camera_yaml, "exposure_ms");
  const auto idle_exposure_ms =
    has_idle_camera ? tools::read<double>(idle_camera_yaml, "exposure_ms") : combat_exposure_ms;
  auto exposure_center_offset = [](double exposure_ms) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  };
  tools::logger()->info(
    "Image q timestamp offset: combat -{:.3f} ms, idle -{:.3f} ms (exposure center)",
    combat_exposure_ms * 0.5, idle_exposure_ms * 0.5);

  io::Gimbal gimbal(config_path);
  rclcpp::init(argc, argv);
  auto fdb_node = std::make_shared<rclcpp::Node>("remote_hero_fdb_subscriber");
  [[maybe_unused]] auto fdb_subscription =
    fdb_node->create_subscription<robot_msgs::msg::RemoteHeroFdb>(
      "fdb", rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
      [&gimbal, navi_yaw_offset, navi_pitch_offset](
        const robot_msgs::msg::RemoteHeroFdb::SharedPtr msg) {
        const auto yaw = msg->yaw + navi_yaw_offset;
        const auto pitch = msg->pitch + navi_pitch_offset;
        // tools::logger()->info(
        //   "[Remote Hero FDB] yaw: {:.3f} + {:.3f} = {:.3f}, pitch: {:.3f} + {:.3f} = {:.3f}",
        //   msg->yaw, navi_yaw_offset, yaw, msg->pitch, navi_pitch_offset, pitch);
        gimbal.send(false, false, yaw, 0, 0, pitch, 0, 0);
      });
  tools::logger()->info("[Remote Hero FDB] subscribed to /fdb");
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
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  hdvt::DeployVtSender vt_sender(yaml, gimbal);

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
  auto auto_aim_mode_enter_t = std::chrono::steady_clock::now();
  constexpr size_t max_async_frame_qs = 128;
  std::map<int, Eigen::Quaterniond> async_frame_qs;

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

    if (async_result_count > 0) {
      tools::logger()->info(
        "async submit: {:.2f} fps, async result: {:.2f} fps, avg latency: {:.2f} ms, busy_drop: "
        "{}, stale_drop: {}",
        async_submitted_count / report_dt, async_result_count / report_dt,
        async_detect_time_sum / async_result_count, async_busy_drop_count, async_stale_drop_count);
    } else {
      tools::logger()->info(
        "async submit: {:.2f} fps, async result: 0.00 fps, busy_drop: {}, stale_drop: {}",
        async_submitted_count / report_dt, async_busy_drop_count, async_stale_drop_count);
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
    while (yolo.try_fetch(dropped_result)) {
      async_stale_drop_count++;
    }
    async_frame_qs.clear();
  };

  auto uses_idle_camera = [&](io::GimbalMode current_mode) {
    return current_mode == io::GimbalMode::IDLE && idle_camera.has_value();
  };

  auto exposure_for_mode = [&](io::GimbalMode current_mode) {
    return uses_idle_camera(current_mode) ? idle_exposure_ms : combat_exposure_ms;
  };

  auto activate_camera_for_mode = [&](io::GimbalMode current_mode) -> io::Camera & {
    auto * desired_camera = &camera;
    if (uses_idle_camera(current_mode)) {
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

  auto process_auto_aim_detection = [&](
                                      std::list<auto_aim::Armor> armors,
                                      const std::chrono::steady_clock::time_point & img_time,
                                      const Eigen::Quaterniond & img_q) {
    auto gs = gimbal.state();

    tracker.set_enemy_color_from_camp(gs.camp);
    solver.set_R_gimbal2world(img_q);

    auto targets = tracker.track(armors, img_time);
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
        auto plan = planner.plan(target, gs.bullet_speed, gs.pitch);

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
      } else {
        target_queue.push(std::nullopt);
        discard_pending_yolo_results();
      }
    }

    const auto current_mode = mode.load();
    activate_camera_for_mode(current_mode).read(img, t);
    auto img_time = t - exposure_center_offset(exposure_for_mode(current_mode));

    if (current_mode == io::GimbalMode::IDLE) {
      discard_pending_yolo_results();
      target_queue.push(std::nullopt);

      vt_sender.tryPushImg(img);
      
      rclcpp::spin_some(fdb_node);
      continue;
    }  //坤的地盘！！！！！！！！！！！！！！！！！！

    if (current_mode != io::GimbalMode::AUTO_AIM) {
      discard_pending_yolo_results();
      target_queue.push(std::nullopt);
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
      continue;
    }

    auto img_q = gimbal.q(img_time);
    // recorder.record(img, img_q, img_time);

    /// 自瞄
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
      update_fps(std::chrono::steady_clock::now(), tools::delta_time(detect_end, detect_start));
    }
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  rclcpp::shutdown();

  return 0;
}
