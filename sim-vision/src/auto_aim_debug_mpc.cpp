#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <iterator>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  auto yaml = tools::load(config_path);
  const auto exposure_ms = tools::read<double>(yaml, "exposure_ms");
  const auto exposure_center_offset =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  tools::logger()->info(
    "Image q timestamp offset: -{:.3f} ms (exposure center)", exposure_ms * 0.5);

  const YAML::Node combat_camera_yaml = yaml["combat_camera"] ? yaml["combat_camera"] : yaml;
  const YAML::Node idle_camera_yaml = yaml["idle_camera"];
  const bool has_idle_camera = static_cast<bool>(idle_camera_yaml);

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
  auto_aim::Solver solver(config_path);
  solver.set_debug_xyz_optimize_log(true);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  std::atomic<bool> quit = false;
  std::atomic<io::GimbalMode> mode{initial_mode};
  auto last_mode = initial_mode;
  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    const auto plan_send_period = std::chrono::microseconds(10000);  // ~100 Hz
    auto next_send = std::chrono::steady_clock::now();

    while (!quit) {
      if (mode.load() != io::GimbalMode::AUTO_AIM) {
        target_queue.push(std::nullopt);
        gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
        next_send = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(20ms);
        continue;
      }

      next_send += plan_send_period;

      auto target = target_queue.front();
      auto gs = gimbal.state();
      auto plan = planner.plan(target, gs.bullet_speed, gs.pitch);
      const auto gate = planner.fire_gate_debug();

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      auto fired = gs.bullet_count > last_bullet_count;
      last_bullet_count = gs.bullet_count;

      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

      data["gimbal_yaw"] = -(2 * M_PI - gs.yaw);
      data["gimbal_yaw_vel"] = gs.yaw_vel;
      data["gimbal_pitch"] = gs.pitch;
      data["gimbal_pitch_vel"] = gs.pitch_vel;

      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;

      data["plan_yaw"] = plan.yaw;
      data["plan_yaw_vel"] = plan.yaw_vel;
      data["plan_yaw_acc"] = plan.yaw_acc;

      data["plan_pitch"] = plan.pitch;
      data["plan_pitch_vel"] = plan.pitch_vel;
      data["plan_pitch_acc"] = plan.pitch_acc;

      data["fire"] = plan.fire ? 1 : 0;
      data["fired"] = fired ? 1 : 0;
      data["fire_gate_state"] = static_cast<int>(gate.state);
      data["fire_gate_ready"] = gate.ready ? 1 : 0;
      data["fire_gate_raw_fire"] = gate.raw_fire ? 1 : 0;
      data["fire_gate_actual_fire"] = gate.actual_fire ? 1 : 0;
      data["fire_gate_prediction_change_deg"] = gate.prediction_change_deg;
      data["fire_gate_stable_observations"] = gate.stable_observations;
      data["fire_gate_warmup_remaining_ms"] = gate.warmup_remaining_ms;
      data["fire_gate_observation_age_ms"] = gate.observation_age_ms;

      if (target.has_value()) {
        data["target_z"] = target->ekf_x()[4];   // z
        data["target_vz"] = target->ekf_x()[5];  // vz
        data["observed_z"] = target->last_observed_z();
        data["aim_z"] = planner.debug_xyza.z();
        data["z_observation_only"] = target->z_observation_only() ? 1 : 0;
      } else {
        data["observed_z"] = 0.0;
        data["aim_z"] = 0.0;
        data["z_observation_only"] = 0;
      }

      if (target.has_value()) {
        data["w"] = target->ekf_x()[7];
        data["last_id"] = target->last_id;
        data["residual_yaw"] = target->ekf().data.at("residual_yaw");
        data["residual_pitch"] = target->ekf().data.at("residual_pitch");
        data["residual_distance"] = target->ekf().data.at("residual_distance");
        data["residual_angle"] = target->ekf().data.at("residual_angle");
        data["nis"] = target->ekf().data.at("nis");
        data["process_v1"] = target->ekf().data.at("process_v1");
        data["process_v2"] = target->ekf().data.at("process_v2");
        data["adaptive_v1_boost"] = target->ekf().data.at("adaptive_v1_boost");
        data["adaptive_v2_boost"] = target->ekf().data.at("adaptive_v2_boost");
      } else {
        data["w"] = 0.0;
        data["last_id"] = -1;
        data["residual_yaw"] = 0.0;
        data["residual_pitch"] = 0.0;
        data["residual_distance"] = 0.0;
        data["residual_angle"] = 0.0;
        data["nis"] = 0.0;
        data["process_v1"] = 0.0;
        data["process_v2"] = 0.0;
        data["adaptive_v1_boost"] = 0.0;
        data["adaptive_v2_boost"] = 0.0;
      }

      plotter.plot(data);

      auto now = std::chrono::steady_clock::now();
        if (now > next_send + plan_send_period) {
          next_send = now;
        }
        std::this_thread::sleep_until(next_send);

      // std::this_thread::sleep_for(10ms);
    }
  });

  auto process_detection = [&](std::list<auto_aim::Armor> armors,
                                const std::chrono::steady_clock::time_point & img_time,
                                const Eigen::Quaterniond & img_q,
                                cv::Mat display_img) {
    auto gs = gimbal.state();
    tracker.set_enemy_color_from_camp(gs.camp);

    solver.set_R_gimbal2world(img_q);

    auto targets = tracker.track(armors, img_time);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧 target 更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(display_img, image_points, {0, 255, 0});
      }

      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(display_img, image_points, {0, 0, 255});
    }

    cv::resize(display_img, display_img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", display_img);
  };

  auto draw_detection = [&](const cv::Mat & raw_img, const std::list<auto_aim::Armor> & armors,
                            int frame_id) {
    constexpr auto detection_display_period = 16ms;  // ~60 Hz
    static auto last_detection_display = std::chrono::steady_clock::time_point{};

    auto now = std::chrono::steady_clock::now();
    if (
      last_detection_display.time_since_epoch().count() != 0 &&
      now - last_detection_display < detection_display_period) {
      return;
    }
    last_detection_display = now;

    auto detection = raw_img.clone();
    tools::draw_text(detection, fmt::format("[{}]", frame_id), {10, 30}, {255, 255, 255});
    for (const auto & armor : armors) {
      auto info = fmt::format(
        "{:.2f} {} {} {}", armor.confidence, auto_aim::COLORS[armor.color],
        auto_aim::ARMOR_NAMES[armor.name], auto_aim::ARMOR_TYPES[armor.type]);
      tools::draw_points(detection, armor.points, {0, 255, 0});
      tools::draw_text(detection, info, armor.center, {0, 255, 0});
    }

    cv::resize(detection, detection, {}, 0.5, 0.5);
    cv::imshow("detection", detection);
  };

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  auto fps_report_stamp = std::chrono::steady_clock::now();

  double detect_time_sum = 0.0;
  int sync_frame_count = 0;

  int frame_count = 0;
  int async_submitted_count = 0;
  int async_result_count = 0;
  int async_busy_drop_count = 0;
  int async_stale_drop_count = 0;
  double async_detect_time_sum = 0.0;
  bool has_last_result_time = false;
  std::chrono::steady_clock::time_point last_result_time;
  auto auto_aim_mode_enter_t = std::chrono::steady_clock::now();
  constexpr size_t max_async_frame_qs = 128;
  std::map<int, Eigen::Quaterniond> async_frame_qs;

  const bool use_async_yolo = yolo.supports_async();
  tools::logger()->info(
    "YOLO pipeline mode: {}", use_async_yolo ? "OpenVINO async thread-pool" : "sync fallback");

  auto discard_pending_yolo_results = [&]() {
    if (!use_async_yolo) return;

    auto_aim::YOLOAsyncResult dropped_result;
    while (yolo.try_fetch(dropped_result)) {
      async_stale_drop_count++;
    }
    async_frame_qs.clear();
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

  auto show_image_only = [](const std::string & window_name, const cv::Mat & raw_img) {
    if (raw_img.empty()) return;

    auto display = raw_img.clone();
    cv::resize(display, display, {}, 0.5, 0.5);
    cv::imshow(window_name, display);
  };

  while (!exiter.exit()) {
    mode = gimbal.mode();
    const auto current_mode = mode.load();
    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", gimbal.str(current_mode));
      last_mode = current_mode;
      target_queue.push(std::nullopt);
      discard_pending_yolo_results();
      has_last_result_time = false;
      if (current_mode == io::GimbalMode::AUTO_AIM) {
        auto_aim_mode_enter_t = std::chrono::steady_clock::now();
      }
    }

    activate_camera_for_mode(current_mode).read(img, t);
    auto img_time = t - exposure_center_offset;

    if (current_mode != io::GimbalMode::AUTO_AIM) {
      target_queue.push(std::nullopt);
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
      show_image_only(current_mode == io::GimbalMode::IDLE ? "idle_camera" : "combat_camera", img);

      auto key = cv::waitKey(1);
      if (key == 'q') break;
      continue;
    }

    auto img_q = gimbal.q(img_time);

    if (use_async_yolo) {
      const auto submitted_frame_count = frame_count++;
      if (yolo.submit(img, submitted_frame_count, img_time)) {
        async_submitted_count++;
        async_frame_qs[submitted_frame_count] = img_q;
        while (async_frame_qs.size() > max_async_frame_qs) {
          async_frame_qs.erase(async_frame_qs.begin());
        }
      } else {
        async_busy_drop_count++;
      }

      auto_aim::YOLOAsyncResult result;
      while (yolo.fetch(result)) {
        // OpenVINO does not guarantee multi-request completion order.
        // Tracker must see monotonic timestamps; stale results are dropped here.
        if (result.timestamp < auto_aim_mode_enter_t) {
          async_stale_drop_count++;
          continue;
        }

        if (has_last_result_time && result.timestamp <= last_result_time) {
          async_stale_drop_count++;
          continue;
        }
        has_last_result_time = true;
        last_result_time = result.timestamp;
        async_result_count++;
        async_detect_time_sum += result.detect_dt;

        auto frame_q_it = async_frame_qs.find(result.frame_count);
        if (frame_q_it == async_frame_qs.end()) {
          async_stale_drop_count++;
          continue;
        }

        auto result_q = frame_q_it->second;
        async_frame_qs.erase(async_frame_qs.begin(), std::next(frame_q_it));
        draw_detection(result.img, result.armors, result.frame_count);
        process_detection(
          std::move(result.armors), result.timestamp, result_q, std::move(result.img));
      }
    } else {
      auto detect_start = std::chrono::steady_clock::now();
      auto armors = yolo.detect(img, frame_count++);
      auto detect_end = std::chrono::steady_clock::now();
      detect_time_sum += tools::delta_time(detect_end, detect_start);
      sync_frame_count++;

      draw_detection(img, armors, frame_count - 1);
      process_detection(std::move(armors), img_time, img_q, img.clone());
    }

    auto fps_now = std::chrono::steady_clock::now();
    auto report_dt = tools::delta_time(fps_now, fps_report_stamp);
    if (report_dt >= 1.0) {
      if (use_async_yolo) {
        if (async_result_count > 0) {
          tools::logger()->info(
            "async submit: {:.2f} fps, async result: {:.2f} fps, avg latency: {:.2f} ms, "
            "busy_drop: {}, stale_drop: {}",
            async_submitted_count / report_dt, async_result_count / report_dt,
            async_detect_time_sum / async_result_count, async_busy_drop_count,
            async_stale_drop_count);
        } else {
          tools::logger()->info(
            "async submit: {:.2f} fps, async result: 0.00 fps, busy_drop: {}, stale_drop: {}",
            async_submitted_count / report_dt, async_busy_drop_count, async_stale_drop_count);
        }
        async_submitted_count = 0;
        async_result_count = 0;
        async_busy_drop_count = 0;
        async_stale_drop_count = 0;
        async_detect_time_sum = 0.0;
      } else if (detect_time_sum > 0.0) {
        tools::logger()->info(
          "detect avg: {:.2f} fps, loop: {:.2f} fps", sync_frame_count / detect_time_sum,
          sync_frame_count / report_dt);
        detect_time_sum = 0.0;
        sync_frame_count = 0;
      }
      fps_report_stamp = fps_now;
    }

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
