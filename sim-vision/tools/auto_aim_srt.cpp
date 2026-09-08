#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/angle_estimate.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                   | Print this help message }"
  "{config-path c  | configs/demo.yaml | Replay configuration file }"
  "{mode           | common            | Replay mode: common or buff }"
  "{buff-mode      | auto              | Buff mode: auto, small or big }"
  "{output o       |                   | Output SRT path }"
  "{overlay-video  |                   | Rendered video with detection overlays }"
  "{start-index s  | 0                 | First video frame to process }"
  "{end-index e    | 0                 | Last video frame to process, 0 means all }"
  "{bullet-speed   | 27                | Bullet speed passed to the aimer }"
  "{@input-path    | assets/demo/demo  | Recording prefix or video path }";

namespace
{
constexpr double kFallbackFps = 30.0;

bool option_requires_value(const std::string & option)
{
  return option == "--config-path" || option == "-c" || option == "--output" || option == "-o" ||
         option == "--overlay-video" || option == "--start-index" || option == "-s" ||
         option == "--end-index" || option == "-e" || option == "--bullet-speed" ||
         option == "--mode" || option == "--buff-mode";
}

std::vector<std::string> normalize_command_line_args(int argc, char * argv[])
{
  std::vector<std::string> normalized_args;
  normalized_args.reserve(argc);
  for (int index = 0; index < argc; ++index) {
    std::string argument{argv[index]};
    if (argument == "-buff" || argument == "--buff") argument = "--mode=buff";
    if (
      option_requires_value(argument) && index + 1 < argc && argv[index + 1][0] != '-') {
      argument += "=";
      argument += argv[++index];
    }
    normalized_args.push_back(std::move(argument));
  }
  return normalized_args;
}

std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  return value;
}

enum class ReplayMode
{
  Common,
  Buff
};

ReplayMode parse_replay_mode(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "common") return ReplayMode::Common;
  if (normalized == "buff") return ReplayMode::Buff;
  throw std::invalid_argument(fmt::format("Unsupported replay mode: {}", value));
}

auto_buff::BuffMode parse_buff_mode(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "auto") return auto_buff::BuffMode::LOST;
  if (normalized == "small") return auto_buff::BuffMode::SMALL;
  if (normalized == "big") return auto_buff::BuffMode::BIG;
  throw std::invalid_argument(fmt::format("Unsupported buff mode: {}", value));
}

struct ReplayPaths
{
  std::filesystem::path video;
  std::filesystem::path pose;
  std::filesystem::path subtitle;
};

bool is_video_path(const std::filesystem::path & path)
{
  const auto extension = path.extension().string();
  return extension == ".avi" || extension == ".mp4" || extension == ".mkv" || extension == ".mov";
}

ReplayPaths resolve_replay_paths(const std::string & input_path, const std::string & output_path)
{
  const std::filesystem::path input{input_path};
  ReplayPaths paths;

  if (is_video_path(input)) {
    paths.video = input;
    paths.pose = input;
    paths.pose.replace_extension(".txt");
    paths.subtitle = input;
    paths.subtitle.replace_extension(".srt");
  } else {
    const auto avi_path = std::filesystem::path{input_path + ".avi"};
    const auto mp4_path = std::filesystem::path{input_path + ".mp4"};
    paths.video = std::filesystem::exists(avi_path) ? avi_path : mp4_path;
    paths.pose = std::filesystem::path{input_path + ".txt"};
    paths.subtitle = std::filesystem::path{input_path + ".srt"};
  }

  if (!output_path.empty()) paths.subtitle = output_path;
  return paths;
}

bool read_pose(
  std::ifstream & input, double & timestamp, double & w, double & x, double & y, double & z)
{
  return static_cast<bool>(input >> timestamp >> w >> x >> y >> z);
}

bool skip_pose_records(std::ifstream & input, int count)
{
  double timestamp, w, x, y, z;
  for (int index = 0; index < count; ++index) {
    if (!read_pose(input, timestamp, w, x, y, z)) return false;
  }
  return true;
}

std::int64_t frame_time_ms(int frame_index, double fps)
{
  return static_cast<std::int64_t>(std::llround(frame_index * 1000.0 / fps));
}

std::string srt_timestamp(std::int64_t time_ms)
{
  const auto hours = time_ms / 3600000;
  time_ms %= 3600000;
  const auto minutes = time_ms / 60000;
  time_ms %= 60000;
  const auto seconds = time_ms / 1000;
  const auto milliseconds = time_ms % 1000;
  return fmt::format("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, milliseconds);
}

std::string subtitle_text(const io::Command & command, const Eigen::Vector3d & gimbal_ypr_deg)
{
  return fmt::format(
    "command is {},{:.2f},{:.2f},shoot:{}\n"
    "gimbal yaw {:.2f}\n"
    "gimbal pitch {:.2f}",
    command.control, command.yaw * 57.3, command.pitch * 57.3, command.shoot, gimbal_ypr_deg[0],
     gimbal_ypr_deg[1]);
}

void draw_frame_index(cv::Mat & frame, int frame_index)
{
  const auto label = fmt::format("[{}]", frame_index);
  constexpr double font_scale = 0.8;
  constexpr int thickness = 2;
  int baseline = 0;
  const auto label_size = cv::getTextSize(
    label, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
  tools::draw_text(
    frame, label, {std::max(10, frame.cols - label_size.width - 10), label_size.height + 10},
    {255, 255, 255}, font_scale, thickness);
}

void draw_detection_overlay(
  cv::Mat & frame, const std::list<auto_aim::Armor> & armors, int frame_index)
{
  draw_frame_index(frame, frame_index);
  for (const auto & armor : armors) {
    const auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, auto_aim::COLORS[armor.color],
      auto_aim::ARMOR_NAMES[armor.name], auto_aim::ARMOR_TYPES[armor.type]);
    tools::draw_points(frame, armor.points, {0, 255, 0});
    tools::draw_text(frame, info, armor.center, {0, 255, 0});
  }
}

void draw_target_overlay(
  cv::Mat & frame, const auto_aim::Target & target, const auto_aim::Solver & solver,
  const auto_aim::AimPoint & aim_point)
{
  for (const Eigen::Vector4d & xyza : target.armor_xyza_list()) {
    const auto image_points =
      solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
    tools::draw_points(frame, image_points, {255, 255, 0});
  }

  if (!aim_point.valid) return;
  const auto image_points = solver.reproject_armor(
    aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
  tools::draw_points(frame, image_points, {0, 0, 255});
}

void draw_command_overlay(
  cv::Mat & frame, const io::Command & command, const Eigen::Vector3d & gimbal_ypr_deg)
{
  tools::draw_text(
    frame,
    fmt::format(
      "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
      command.pitch * 57.3, command.shoot),
    {10, 60}, {154, 50, 205});
  tools::draw_text(
    frame, fmt::format("gimbal yaw {:.2f}", gimbal_ypr_deg[0]), {10, 90}, {255, 255, 255});
  tools::draw_text(
    frame, fmt::format("gimbal pitch {:.2f}", gimbal_ypr_deg[1]), {10, 120}, {255, 255, 255});
}

int output_fourcc(const std::filesystem::path & output_path)
{
  if (output_path.extension() == ".mp4") return cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  return cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
}

void write_subtitle(
  std::ofstream & output, int sequence, int frame_index, double fps, const std::string & text)
{
  const auto start_time_ms = frame_time_ms(frame_index, fps);
  const auto end_time_ms = std::max(start_time_ms + 1, frame_time_ms(frame_index + 1, fps));
  output << sequence << "\n";
  output << srt_timestamp(start_time_ms) << " --> " << srt_timestamp(end_time_ms) << "\n";
  output << text << "\n\n";
}

struct BuffGimbalHistory
{
  bool valid = false;
  double timestamp = 0.0;
  double yaw = 0.0;
  double pitch = 0.0;
};

io::GimbalState buff_gimbal_state(
  const Eigen::Quaterniond & q, double timestamp, double bullet_speed, BuffGimbalHistory & history)
{
  const auto ypr = tools::eulers(q.toRotationMatrix(), 2, 1, 0);
  io::GimbalState state{};
  state.yaw = static_cast<float>(ypr[0]);
  state.pitch = static_cast<float>(ypr[1]);
  state.bullet_speed = static_cast<float>(bullet_speed);
  if (history.valid) {
    const double dt = timestamp - history.timestamp;
    if (std::isfinite(dt) && dt > 1e-6) {
      state.yaw_vel = static_cast<float>(tools::limit_rad(ypr[0] - history.yaw) / dt);
      state.pitch_vel = static_cast<float>((ypr[1] - history.pitch) / dt);
    }
  }
  history = {true, timestamp, ypr[0], ypr[1]};
  return state;
}

cv::Scalar buff_blade_color(
  const auto_buff::FanBlade & blade, int observed_leaf_id, int attack_leaf_id)
{
  const bool is_observed = observed_leaf_id >= 0 && blade.leaf_id == observed_leaf_id;
  const bool is_attack = attack_leaf_id >= 0 && blade.leaf_id == attack_leaf_id;
  if (is_observed && is_attack) return {255, 255, 0};
  if (is_attack) return {0, 255, 255};
  if (is_observed) return {0, 255, 0};
  if (blade.type == auto_buff::_unlight) return {96, 96, 96};
  return {255, 180, 0};
}

void draw_buff_detection_overlay(
  cv::Mat & frame, const std::optional<auto_buff::PowerRune> & rune, int frame_index,
  const auto_buff::RCenterRefineDebug & r_center_debug)
{
  draw_frame_index(frame, frame_index);
  if (r_center_debug.refined_valid || r_center_debug.coarse_r_center != cv::Point2f{}) {
    tools::draw_point(frame, r_center_debug.coarse_r_center, {0, 255, 255}, 4);
    tools::draw_text(
      frame, "R coarse", r_center_debug.coarse_r_center + cv::Point2f(8.0f, -8.0f), {0, 255, 255},
      0.5, 1);
    if (r_center_debug.refined_valid) {
      tools::draw_point(frame, r_center_debug.refined_r_center, {0, 0, 255}, 5);
      tools::draw_text(
        frame, "R refined", r_center_debug.refined_r_center + cv::Point2f(8.0f, 14.0f),
        {0, 0, 255}, 0.5, 1);
    }
  }
  if (!rune) return;

  for (std::size_t index = 0; index < rune->fanblades.size(); ++index) {
    const auto & blade = rune->fanblades[index];
    if (blade.type == auto_buff::_unlight || blade.points.size() < 4) continue;
    const bool is_target = index == 0 || blade.type == auto_buff::_target;
    const cv::Scalar color = is_target ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    tools::draw_points(frame, blade.points, color, is_target ? 3 : 2);
    tools::draw_point(frame, blade.points[0], {0, 0, 255}, is_target ? 5 : 4);
    for (int point_index = 1; point_index < 4; ++point_index) {
      tools::draw_point(frame, blade.points[point_index], color, is_target ? 5 : 4);
    }

    std::string label =
      fmt::format("{} conf:{:.2f}", is_target ? "target" : "leaf", blade.confidence);
    if (blade.leaf_angle_valid) {
      label += fmt::format(" id:{} angle:{:.1f}", blade.leaf_id, blade.leaf_angle * 57.3);
    }
    tools::draw_text(frame, label, blade.center + cv::Point2f(8.0f, -8.0f), color, 0.5, 1);
  }
}

std::vector<cv::Point2f> draw_buff_reprojection(
  cv::Mat & frame, auto_buff::Solver & solver, const Eigen::Vector3d & r_center_world,
  const Eigen::Matrix3d & rotation_world, const cv::Scalar & color)
{
  const auto image_points = solver.reproject_buff(r_center_world, rotation_world);
  if (image_points.size() < 5) return image_points;
  tools::draw_points(
    frame, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), color, 2);
  tools::draw_points(
    frame, std::vector<cv::Point2f>(image_points.begin() + 4, image_points.end()), color, 2);
  return image_points;
}

void draw_buff_result_overlay(
  cv::Mat & frame, const auto_buff::PowerRune & rune, auto_buff::Solver & solver,
  const auto_buff::Aimer & aimer)
{
  const auto & result = aimer.last_ballistic_result();
  const int observed_leaf_id =
    result.observed_leaf_id >= 0 ? result.observed_leaf_id : rune.target_leaf_id;
  const int attack_leaf_id = result.attack_leaf_id;
  for (const auto & blade : rune.fanblades) {
    if (blade.points.empty()) continue;
    const auto color = buff_blade_color(blade, observed_leaf_id, attack_leaf_id);
    tools::draw_points(frame, blade.points, color, blade.type == auto_buff::_unlight ? 1 : 2);
    tools::draw_point(frame, blade.center, color, 4);
  }
  tools::draw_point(frame, rune.r_center, {0, 0, 255}, 4);
  tools::draw_text(
    frame, fmt::format("observed leaf: {}  attack leaf: {}", observed_leaf_id, attack_leaf_id),
    {10, 28}, {255, 255, 255}, 0.7, 2);
  draw_buff_reprojection(frame, solver, rune.xyz_in_world, rune.rotation_world, {0, 255, 0});

  if (!result.valid || !std::isfinite(result.predicted_angle)) return;
  Eigen::Vector3d predicted_ypr = rune.ypr_in_world;
  predicted_ypr[2] = result.predicted_angle + CV_PI / 2.0;
  const auto image_points = draw_buff_reprojection(
    frame, solver, rune.xyz_in_world, tools::rotation_matrix(predicted_ypr), {0, 0, 255});
  const int predicted_leaf_id =
    result.attack_leaf_id >= 0 ? result.attack_leaf_id : observed_leaf_id;
  tools::draw_text(
    frame,
    fmt::format("predict attack leaf: {}  dt:{:.1f}ms", predicted_leaf_id,
                std::max(0.0, (result.predict_time_abs - result.observed_time_abs) * 1e3)),
    {10, 84}, {0, 0, 255}, 0.7, 2);
  if (image_points.size() >= 5) tools::draw_point(frame, image_points.back(), {0, 0, 255}, 5);
}

std::string buff_subtitle_text(
  const std::optional<auto_buff::PowerRune> & rune, const auto_aim::Plan & plan,
  const auto_buff::BuffBallisticResult & result, const Eigen::Vector3d & gimbal_ypr_deg)
{
  const bool pnp_valid = rune.has_value() && rune->pnp_valid;
  const int observed_leaf_id =
    rune ? (result.observed_leaf_id >= 0 ? result.observed_leaf_id : rune->target_leaf_id) : -1;
  return fmt::format(
    "buff detected:{} pnp:{} observed leaf:{} attack leaf:{}\n"
    "plan control:{} yaw:{:.2f} pitch:{:.2f} fire:{}\n"
    "gimbal yaw:{:.2f} pitch:{:.2f}",
    rune.has_value(), pnp_valid, observed_leaf_id, result.attack_leaf_id, plan.control,
    plan.yaw * 57.3, plan.pitch * 57.3, plan.fire, gimbal_ypr_deg[0], gimbal_ypr_deg[1]);
}

int export_buff_replay(
  cv::VideoCapture & video, std::ifstream & pose_input, std::ofstream & subtitle_output,
  const std::filesystem::path & subtitle_path, const std::filesystem::path & overlay_video_path,
  const std::string & config_path, int start_index, int end_index, double fps, double bullet_speed,
  auto_buff::BuffMode buff_mode)
{
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::Aimer aimer(config_path);
  const auto replay_origin = std::chrono::steady_clock::now();
  BuffGimbalHistory gimbal_history;
  cv::VideoWriter overlay_writer;
  int sequence = 1;

  for (int frame_index = start_index; ; ++frame_index) {
    if (end_index > 0 && frame_index > end_index) break;

    cv::Mat frame;
    if (!video.read(frame)) break;

    double capture_time, w, x, y, z;
    if (!read_pose(pose_input, capture_time, w, x, y, z)) {
      std::cerr << "Pose file ends before video at frame: " << frame_index << '\n';
      return 1;
    }

    const Eigen::Quaterniond q{w, x, y, z};
    const auto timestamp = replay_origin +
                           std::chrono::microseconds(static_cast<std::int64_t>(capture_time * 1e6));
    const auto gimbal_state = buff_gimbal_state(q, capture_time, bullet_speed, gimbal_history);
    auto_buff::RCenterRefineDebug r_center_debug;
    auto rune = detector.detect(frame, &r_center_debug);
    solver.set_R_gimbal2world(q);
    solver.solve(rune, &timestamp);

    auto_aim::Plan plan{false, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (rune) {
      rune->last_observed_time = capture_time;
      plan = aimer.mpc_aim(*rune, capture_time, capture_time, gimbal_state, buff_mode);
    } else {
      aimer.notify_observation_missed();
    }

    const auto gimbal_ypr_deg = tools::eulers(q.toRotationMatrix(), 2, 1, 0) * 57.3;
    if (!overlay_video_path.empty()) {
      if (!overlay_writer.isOpened()) {
        overlay_writer.open(
          overlay_video_path.string(), output_fourcc(overlay_video_path), fps, frame.size());
        if (!overlay_writer.isOpened()) {
          std::cerr << "Failed to create overlay video: " << overlay_video_path << '\n';
          return 1;
        }
      }
      draw_buff_detection_overlay(frame, rune, frame_index, r_center_debug);
      if (rune && rune->pnp_valid) draw_buff_result_overlay(frame, *rune, solver, aimer);
      tools::draw_text(
        frame,
        fmt::format("expect yaw:{:.2f} pitch:{:.2f} control:{} fire:{}", plan.yaw * 57.3,
                    plan.pitch * 57.3, plan.control, plan.fire),
        {10, 140}, {255, 255, 255}, 0.65, 2);
      tools::draw_text(
        frame, fmt::format("actual yaw:{:.2f} pitch:{:.2f}", gimbal_ypr_deg[0], gimbal_ypr_deg[1]),
        {10, 168}, {255, 255, 255}, 0.65, 2);
      overlay_writer.write(frame);
    }
    write_subtitle(
      subtitle_output, sequence++, frame_index, fps,
      buff_subtitle_text(rune, plan, aimer.last_ballistic_result(), gimbal_ypr_deg));
  }

  std::cout << "Wrote " << sequence - 1 << " subtitles to " << subtitle_path << '\n';
  if (!overlay_video_path.empty()) {
    std::cout << "Wrote overlay video to " << overlay_video_path << '\n';
  }
  return 0;
}
}  // namespace

int main(int argc, char * argv[])
{
  auto command_line_args = normalize_command_line_args(argc, argv);
  std::vector<char *> command_line_arg_ptrs;
  command_line_arg_ptrs.reserve(command_line_args.size());
  for (auto & argument : command_line_args) command_line_arg_ptrs.push_back(argument.data());
  cv::CommandLineParser cli(
    static_cast<int>(command_line_arg_ptrs.size()), command_line_arg_ptrs.data(), keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return 1;
  }

  const auto paths = resolve_replay_paths(cli.get<std::string>(0), cli.get<std::string>("output"));
  const auto config_path = cli.get<std::string>("config-path");
  const auto start_index = std::max(0, cli.get<int>("start-index"));
  const auto requested_end_index = cli.get<int>("end-index");
  const auto bullet_speed = cli.get<double>("bullet-speed");
  const std::filesystem::path overlay_video_path{cli.get<std::string>("overlay-video")};
  ReplayMode replay_mode;
  auto_buff::BuffMode buff_mode;
  try {
    replay_mode = parse_replay_mode(cli.get<std::string>("mode"));
    buff_mode = parse_buff_mode(cli.get<std::string>("buff-mode"));
  } catch (const std::invalid_argument & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  if (!std::filesystem::exists(paths.video)) {
    std::cerr << "Video file does not exist: " << paths.video << '\n';
    return 1;
  }
  if (!std::filesystem::exists(paths.pose)) {
    std::cerr << "Pose file does not exist: " << paths.pose << '\n';
    return 1;
  }
  if (
    !overlay_video_path.empty() &&
    std::filesystem::absolute(overlay_video_path).lexically_normal() ==
      std::filesystem::absolute(paths.video).lexically_normal()) {
    std::cerr << "Overlay video must not overwrite the replay video.\n";
    return 1;
  }

  cv::VideoCapture video(paths.video.string());
  if (!video.isOpened()) {
    std::cerr << "Failed to open video: " << paths.video << '\n';
    return 1;
  }

  std::ifstream pose_input(paths.pose);
  if (!pose_input) {
    std::cerr << "Failed to open pose file: " << paths.pose << '\n';
    return 1;
  }

  auto fps = video.get(cv::CAP_PROP_FPS);
  if (!std::isfinite(fps) || fps <= 0.0) fps = kFallbackFps;
  const auto total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
  if (total_frames > 0 && start_index >= total_frames) {
    std::cerr << "Start frame is outside the video: " << start_index << '\n';
    return 1;
  }

  int end_index = requested_end_index;
  if (total_frames > 0 && (end_index <= 0 || end_index >= total_frames)) {
    end_index = total_frames - 1;
  }
  if (end_index > 0 && end_index < start_index) {
    std::cerr << "End frame precedes start frame.\n";
    return 1;
  }

  const auto output_parent = paths.subtitle.parent_path();
  if (!output_parent.empty()) std::filesystem::create_directories(output_parent);
  const auto overlay_parent = overlay_video_path.parent_path();
  if (!overlay_parent.empty()) std::filesystem::create_directories(overlay_parent);
  std::ofstream output(paths.subtitle);
  if (!output) {
    std::cerr << "Failed to create subtitle file: " << paths.subtitle << '\n';
    return 1;
  }

  if (!video.set(cv::CAP_PROP_POS_FRAMES, start_index)) {
    std::cerr << "Failed to seek video to frame: " << start_index << '\n';
    return 1;
  }
  if (!skip_pose_records(pose_input, start_index)) {
    std::cerr << "Pose file ends before start frame: " << start_index << '\n';
    return 1;
  }

  if (replay_mode == ReplayMode::Buff) {
    return export_buff_replay(
      video, pose_input, output, paths.subtitle, overlay_video_path, config_path, start_index,
      end_index, fps, bullet_speed, buff_mode);
  }

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto tracker = std::make_unique<auto_aim::Tracker>(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  const auto replay_origin = std::chrono::steady_clock::now();
  io::Command last_command{};
  cv::VideoWriter overlay_writer;
  int sequence = 1;

  for (int frame_index = start_index; ; ++frame_index) {
    if (end_index > 0 && frame_index > end_index) break;

    cv::Mat frame;
    if (!video.read(frame)) break;

    double capture_time, w, x, y, z;
    if (!read_pose(pose_input, capture_time, w, x, y, z)) {
      std::cerr << "Pose file ends before video at frame: " << frame_index << '\n';
      return 1;
    }

    const auto timestamp = replay_origin +
                           std::chrono::microseconds(static_cast<std::int64_t>(capture_time * 1e6));
    solver.set_R_gimbal2world({w, x, y, z});
    auto armors = yolo.detect(frame, frame_index);
    const auto targets = tracker->track(armors, timestamp);
    auto command = aimer.aim(targets, timestamp, bullet_speed, false);

    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2 &&
      std::abs(command.pitch - last_command.pitch) * 57.3 < 2)
      command.shoot = true;
    if (command.control) last_command = command;

    const Eigen::Quaterniond gimbal_q{w, x, y, z};
    const auto gimbal_ypr_deg = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;
    if (!overlay_video_path.empty()) {
      if (!overlay_writer.isOpened()) {
        overlay_writer.open(
          overlay_video_path.string(), output_fourcc(overlay_video_path), fps, frame.size());
        if (!overlay_writer.isOpened()) {
          std::cerr << "Failed to create overlay video: " << overlay_video_path << '\n';
          return 1;
        }
      }
      draw_detection_overlay(frame, armors, frame_index);
      if (!targets.empty()) {
        draw_target_overlay(frame, targets.front(), solver, aimer.debug_aim_point);
      }
      draw_command_overlay(frame, command, gimbal_ypr_deg);
      overlay_writer.write(frame);
    }
    write_subtitle(output, sequence++, frame_index, fps, subtitle_text(command, gimbal_ypr_deg));
  }

  std::cout << "Wrote " << sequence - 1 << " subtitles to " << paths.subtitle << '\n';
  if (!overlay_video_path.empty()) {
    std::cout << "Wrote overlay video to " << overlay_video_path << '\n';
  }
  return 0;
}
