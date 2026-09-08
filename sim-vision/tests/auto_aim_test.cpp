#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{fast-forward-seconds | 30          | seconds per fast-forward key }"
  "{@input-path    | assets/demo/demo  | avi和txt文件的路径}";

namespace
{
constexpr int kBaseWaitMs = 30;
constexpr double kFallbackFps = 30.0;
constexpr double kNormalPlaybackSpeed = 1.0;
constexpr double kSlowPlaybackSpeed = 0.25;
constexpr int kLeftArrowLinux = 65361;
constexpr int kLeftArrowWindows = 2424832;
constexpr int kRightArrowLinux = 65363;
constexpr int kRightArrowWindows = 2555904;

struct PlaybackUi
{
  cv::Rect rewind_button;
  cv::Rect fast_forward_button;
  cv::Rect slow_button;
  cv::Rect normal_button;
  cv::Rect pause_button;
  cv::Rect fullscreen_button;
  cv::Rect progress_bar;
  bool rewind_requested = false;
  bool fast_forward_requested = false;
  bool slow_requested = false;
  bool normal_requested = false;
  bool pause_requested = false;
  bool fullscreen_requested = false;
  bool progress_seek_requested = false;
  double progress_seek_ratio = 0;
};

bool read_pose(std::ifstream & text, double & t, double & w, double & x, double & y, double & z)
{
  return static_cast<bool>(text >> t >> w >> x >> y >> z);
}

void skip_pose_records(std::ifstream & text, int count)
{
  double t, w, x, y, z;
  for (int i = 0; i < count && read_pose(text, t, w, x, y, z); i++) {
  }
}

int count_pose_records(const std::string & text_path)
{
  std::ifstream input(text_path);
  int count = 0;
  double t, w, x, y, z;
  while (read_pose(input, t, w, x, y, z)) count++;
  return count;
}

bool is_fast_forward_key(int key)
{
  return key == 'f' || key == 'F' || key == 'd' || key == 'D' ||
         key == kRightArrowLinux || key == kRightArrowWindows;
}

bool is_rewind_key(int key)
{
  return key == 'b' || key == 'B' || key == 'a' || key == 'A' || key == kLeftArrowLinux ||
         key == kLeftArrowWindows;
}

bool is_slow_key(int key) { return key == 's' || key == 'S'; }

bool is_normal_speed_key(int key) { return key == 'n' || key == 'N' || key == '1'; }

bool is_pause_key(int key) { return key == ' '; }

bool is_fullscreen_key(int key) { return key == 'v' || key == 'V' || key == 10 || key == 13; }

int playback_wait_ms(double playback_speed)
{
  return std::max(1, static_cast<int>(std::round(kBaseWaitMs / playback_speed)));
}

void on_mouse(int event, int x, int y, int, void * userdata)
{
  if (event != cv::EVENT_LBUTTONDOWN || userdata == nullptr) return;

  auto * ui = static_cast<PlaybackUi *>(userdata);
  if (ui->rewind_button.contains(cv::Point{x, y})) ui->rewind_requested = true;
  if (ui->fast_forward_button.contains(cv::Point{x, y})) ui->fast_forward_requested = true;
  if (ui->slow_button.contains(cv::Point{x, y})) ui->slow_requested = true;
  if (ui->normal_button.contains(cv::Point{x, y})) ui->normal_requested = true;
  if (ui->pause_button.contains(cv::Point{x, y})) ui->pause_requested = true;
  if (ui->fullscreen_button.contains(cv::Point{x, y})) ui->fullscreen_requested = true;
  if (ui->progress_bar.contains(cv::Point{x, y}) && ui->progress_bar.width > 1) {
    ui->progress_seek_ratio = std::clamp(
      static_cast<double>(x - ui->progress_bar.x) / (ui->progress_bar.width - 1), 0.0, 1.0);
    ui->progress_seek_requested = true;
  }
}

void draw_speed_button(
  cv::Mat & img, const cv::Rect & button, const std::string & text, bool active)
{
  const cv::Scalar fill = active ? cv::Scalar{55, 105, 55} : cv::Scalar{35, 35, 35};
  cv::rectangle(img, button, fill, cv::FILLED);
  cv::rectangle(img, button, {240, 240, 240}, 1);
  cv::putText(
    img, text, {button.x + 10, button.y + 22}, cv::FONT_HERSHEY_SIMPLEX, 0.58, {255, 255, 255}, 2);
}

void draw_playback_controls(
  cv::Mat & img, PlaybackUi & ui, int fast_forward_seconds, double playback_speed,
  int frame_count, int start_index, int playback_end_frame, double physical_time, bool paused,
  bool fullscreen)
{
  const int control_y = std::max(10, img.rows - 72);
  const int text_y = control_y + 22;
  ui.rewind_button = {10, control_y, 112, 32};
  ui.fast_forward_button = {130, control_y, 112, 32};
  ui.slow_button = {250, control_y, 64, 32};
  ui.normal_button = {322, control_y, 64, 32};
  ui.pause_button = {394, control_y, 64, 32};
  ui.fullscreen_button = {466, control_y, 64, 32};

  cv::rectangle(img, ui.rewind_button, {35, 35, 35}, cv::FILLED);
  cv::rectangle(img, ui.rewind_button, {240, 240, 240}, 1);
  cv::putText(
    img, fmt::format("<< {}s", fast_forward_seconds), {22, text_y},
    cv::FONT_HERSHEY_SIMPLEX, 0.62, {255, 255, 255}, 2);
  cv::rectangle(img, ui.fast_forward_button, {35, 35, 35}, cv::FILLED);
  cv::rectangle(img, ui.fast_forward_button, {240, 240, 240}, 1);
  cv::putText(
    img, fmt::format(">> {}s", fast_forward_seconds), {142, text_y},
    cv::FONT_HERSHEY_SIMPLEX, 0.62, {255, 255, 255}, 2);
  draw_speed_button(
    img, ui.slow_button, "0.25x", std::abs(playback_speed - kSlowPlaybackSpeed) < 1e-3);
  draw_speed_button(
    img, ui.normal_button, "1x",   std::abs(playback_speed - kNormalPlaybackSpeed) < 1e-3);
  draw_speed_button(img, ui.pause_button, paused ? ">" : "||", paused);
  draw_speed_button(img, ui.fullscreen_button, fullscreen ? "exit" : "full", fullscreen);
  cv::putText(
    img, "q a/b f/d s/n space v", {538, text_y}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
    {255, 255, 255}, 1);

  const int progress_y = img.rows - 18;
  ui.progress_bar = {10, progress_y, std::max(1, img.cols - 20), 10};
  const bool has_progress_range = playback_end_frame > start_index;
  const double progress = has_progress_range
                            ? std::clamp(
                                static_cast<double>(frame_count - start_index) /
                                  (playback_end_frame - start_index),
                                0.0, 1.0)
                            : 0.0;

  cv::rectangle(img, ui.progress_bar, {35, 35, 35}, cv::FILLED);
  cv::rectangle(img, ui.progress_bar, {240, 240, 240}, 1);
  const int inner_width = std::max(0, ui.progress_bar.width - 2);
  const int filled_width = static_cast<int>(std::round(inner_width * progress));
  if (filled_width > 0) {
    cv::rectangle(
      img,
      {ui.progress_bar.x + 1, ui.progress_bar.y + 1, filled_width, ui.progress_bar.height - 2},
      {0, 190, 255}, cv::FILLED);
  }

  const auto time_text = has_progress_range
                           ? fmt::format(
                               "physical time: {:.3f}s | frame {}/{}", physical_time, frame_count,
                               playback_end_frame)
                           : fmt::format(
                               "physical time: {:.3f}s | frame {}", physical_time, frame_count);
  cv::putText(
    img, time_text, {10, progress_y - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.52, {255, 255, 255}, 1);
}

bool seek_video(cv::VideoCapture & video, int target_frame, double fps)
{
  if (video.set(cv::CAP_PROP_POS_FRAMES, target_frame)) return true;
  return fps > 0 && video.set(cv::CAP_PROP_POS_MSEC, target_frame * 1000.0 / fps);
}

void seek_pose_records(std::ifstream & text, int target_frame)
{
  text.clear();
  text.seekg(0, std::ios::beg);
  skip_pose_records(text, target_frame);
}

int jump_playback(
  cv::VideoCapture & video, std::ifstream & text, int frame_count, int start_index, int end_index,
  int total_frames, double fps, int jump_frames, int seek_seconds)
{
  if (jump_frames == 0) return frame_count;

  int target_frame = frame_count + jump_frames;
  target_frame = std::max(target_frame, start_index);
  if (end_index > 0) target_frame = std::min(target_frame, end_index);
  if (total_frames > 0) target_frame = std::min(target_frame, total_frames - 1);
  if (target_frame == frame_count) return frame_count;

  auto seeked = seek_video(video, target_frame, fps);
  if (!seeked && target_frame > frame_count) {
    int grabbed = 0;
    const int frames_to_grab = target_frame - frame_count - 1;
    while (grabbed < frames_to_grab && video.grab()) grabbed++;
    target_frame = frame_count + grabbed + 1;
    seeked = target_frame > frame_count;
  }

  if (!seeked) {
    tools::logger()->warn("Seek failed: frame {} -> {}", frame_count, target_frame);
    return frame_count;
  }

  seek_pose_records(text, target_frame);

  const auto * action = jump_frames > 0 ? "Fast forward" : "Rewind";
  tools::logger()->info(
    "{} {}s: frame {} -> {}", action, seek_seconds, frame_count, target_frame);

  return target_frame - 1;
}
}  // namespace

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  auto fast_forward_seconds = std::max(1, cli.get<int>("fast-forward-seconds"));

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);
  auto fps = video.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = kFallbackFps;
  auto total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
  if (total_frames <= 0) total_frames = count_pose_records(text_path);
  auto playback_end_frame = end_index > 0 ? end_index : total_frames - 1;
  if (total_frames > 0) playback_end_frame = std::min(playback_end_frame, total_frames - 1);
  auto fast_forward_frames = static_cast<int>(std::round(fps * fast_forward_seconds));
  const std::string window_name = "reprojection";

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img, drawing;
  auto t0 = std::chrono::steady_clock::now();

  auto_aim::Target last_target;
  io::Command last_command{};
  double last_t = -1;
  PlaybackUi playback_ui;
  double playback_speed = kNormalPlaybackSpeed;
  bool paused = false;
  bool fullscreen = false;

  cv::namedWindow(window_name, cv::WINDOW_NORMAL);
  cv::resizeWindow(window_name, 960, 720);
  cv::setMouseCallback(window_name, on_mouse, &playback_ui);

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  skip_pose_records(text, start_index);

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    if (!read_pose(text, t, w, x, y, z)) break;
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    /// 自瞄核心逻辑

    solver.set_R_gimbal2world({w, x, y, z});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker->track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2 &&
      std::abs(command.pitch - last_command.pitch) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;
    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});

    Eigen::Quaternion gimbal_q = {w, x, y, z};
    const auto gimbal_ypr_deg =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;
    tools::draw_text(
      img, fmt::format("gimbal yaw {:.2f}", gimbal_ypr_deg[0]),
      {10, 90}, {255, 255, 255});
    tools::draw_text(
      img, fmt::format("gimbal pitch {:.2f}", gimbal_ypr_deg[1]),
      {10, 120}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    data["gimbal_yaw"] = gimbal_ypr_deg[0];
    data["gimbal_pitch"] = gimbal_ypr_deg[1];
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    if (!targets.empty()) {
      auto target = targets.front();

      if (last_t == -1) {
        last_target = target;
        last_t = t;
        continue;
      }

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧target更新后
      armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
      data["process_v1"] = target.ekf().data.at("process_v1");
      data["process_v2"] = target.ekf().data.at("process_v2");
      data["adaptive_v1_boost"] = target.ekf().data.at("adaptive_v1_boost");
      data["adaptive_v2_boost"] = target.ekf().data.at("adaptive_v2_boost");
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    int key = -1;
    bool quit_requested = false;
    do {
      auto display = img.clone();
      draw_playback_controls(
        display, playback_ui, fast_forward_seconds, playback_speed, frame_count, start_index,
        playback_end_frame, t, paused, fullscreen);
      cv::imshow(window_name, display);
      key = cv::waitKeyEx(paused ? kBaseWaitMs : playback_wait_ms(playback_speed));

      if (key == 'q') {
        quit_requested = true;
        break;
      }
      if (
        is_fullscreen_key(key) || playback_ui.fullscreen_requested ||
        (key == 27 && fullscreen)) {
        fullscreen = !fullscreen;
        cv::setWindowProperty(
          window_name, cv::WND_PROP_FULLSCREEN,
          fullscreen ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        if (!fullscreen) cv::resizeWindow(window_name, 960, 720);
      }
      if (is_pause_key(key) || playback_ui.pause_requested) paused = !paused;
      if (is_slow_key(key) || playback_ui.slow_requested) playback_speed = kSlowPlaybackSpeed;
      if (is_normal_speed_key(key) || playback_ui.normal_requested)
        playback_speed = kNormalPlaybackSpeed;

      playback_ui.pause_requested = false;
      playback_ui.fullscreen_requested = false;
      playback_ui.slow_requested = false;
      playback_ui.normal_requested = false;
      if (
        playback_ui.progress_seek_requested || is_rewind_key(key) ||
        playback_ui.rewind_requested || is_fast_forward_key(key) ||
        playback_ui.fast_forward_requested)
        break;
    } while (paused);
    if (quit_requested) break;

    const auto rewind_requested = is_rewind_key(key) || playback_ui.rewind_requested;
    const auto fast_forward_requested =
      is_fast_forward_key(key) || playback_ui.fast_forward_requested;
    int jump_frames = 0;
    int seek_seconds = fast_forward_seconds;
    if (playback_ui.progress_seek_requested && playback_end_frame > start_index) {
      const auto target_frame = start_index + static_cast<int>(std::round(
                                                playback_ui.progress_seek_ratio *
                                                (playback_end_frame - start_index)));
      jump_frames = target_frame - frame_count;
      seek_seconds = static_cast<int>(std::round(std::abs(jump_frames) / fps));
    } else if (rewind_requested || fast_forward_requested) {
      jump_frames = fast_forward_requested ? fast_forward_frames : -fast_forward_frames;
    }

    playback_ui.progress_seek_requested = false;
    playback_ui.rewind_requested = false;
    playback_ui.fast_forward_requested = false;
    if (jump_frames != 0) {
      const auto next_frame_count = jump_playback(
        video, text, frame_count, start_index, end_index, total_frames, fps, jump_frames,
        seek_seconds);
      if (next_frame_count != frame_count) {
        tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
        last_command = {};
        last_t = -1;
        frame_count = next_frame_count;
      }
    }
  }

  return 0;
}
