#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <optional>
#include <string>

#include "io/sim/talos_ipc.hpp"
#include "sim/debug_visualizer.hpp"
#include "tasks/auto_aim/runtime.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

namespace
{
using json = nlohmann::json;

#ifdef VISION_SIM_DEBUG_BUILD
#define VISION_SIM_GUI_DEBUG_DEFAULT "true"
#else
#define VISION_SIM_GUI_DEBUG_DEFAULT "false"
#endif

const std::string keys =
  "{help h usage ? | | Show command line help }"
  "{@config-path   | configs/daedalus_overlay.yaml | Simulation-only calibration overlay }"
  "{meta-path      | /tmp/talos_ipc_meta | Talos v3 metadata mmap }"
  "{image-pool-path| /tmp/talos_ipc_image_pool | Talos v3 image mmap }"
  "{session-dir    | sim_reports | Report root; a new child directory is created }"
  "{scenario       | daedalus_manual | Scenario name recorded in the report }"
  "{dry-run        | false | Print the Daedalus Talos session configuration without starting }"
  "{gui-debug      | " VISION_SIM_GUI_DEBUG_DEFAULT " | Enable OpenCV diagnostic windows and PlotJuggler output }"
  "{record         | false | Record raw RGB frames in the session directory }"
  "{record-debug   | false | Record diagnostic overlay frames in the session directory }"
  "{truth-overlay  | false | Display ground truth only in diagnostic output }"
  "{enable-fire    | true | Let Daedalus launch a simulated projectile when the planner requests fire }"
  "{device         | CPU | Override OpenVINO device in a session-local config copy }"
  "{frame-timeout-ms | 5000 | Stop and report an error when Talos produces no new frame }"
  "{max-frames     | 0 | Stop after accepted frames; 0 means unlimited }";

#undef VISION_SIM_GUI_DEBUG_DEFAULT

constexpr int kTalosFrameRate = 100;
constexpr const char * kRobotConfigPath = "configs/sentry.yaml";
constexpr double kSimBuffFireToleranceRad = 0.12;
constexpr auto kSimBuffFireInterval = std::chrono::milliseconds(50);
constexpr auto kSimBuffFirePulse = std::chrono::milliseconds(30);

struct Stats
{
  std::uint64_t polled_frames = 0;
  std::uint64_t accepted_frames = 0;
  std::uint64_t rejected_frames = 0;
  std::uint64_t auto_aim_detections = 0;
  std::uint64_t auto_aim_frames = 0;
  std::uint64_t tracked_frames = 0;
  std::uint64_t detection_truth_matches = 0;
  std::uint64_t pnp_samples = 0;
  std::uint64_t gimbal_follow_samples = 0;
  std::uint64_t auto_aim_commands = 0;
  std::uint64_t small_buff_commands = 0;
  std::uint64_t big_buff_commands = 0;
  std::uint64_t hit_events = 0;
  std::uint64_t correct_hits = 0;
  std::uint64_t armor_hits = 0;
  std::uint64_t rune_hits = 0;
  std::uint64_t rune_correct_target_hits = 0;
  std::uint64_t rune_wrong_target_hits = 0;
  std::uint64_t rune_timeout_hits = 0;
  std::uint64_t last_frame_seq = 0;
  std::uint64_t last_command_seq = 0;
  double end_to_end_latency_ms_sum = 0.0;
  double pnp_position_error_m_sum = 0.0;
  double pnp_orientation_error_rad_sum = 0.0;
  double gimbal_follow_error_rad_sum = 0.0;
};

std::string make_session_directory(const std::string & root, const std::string & scenario)
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  const auto path = std::filesystem::path(root) / (scenario + "-" + std::to_string(milliseconds));
  std::filesystem::create_directories(path);
  return path.string();
}

std::string shell_text(const std::string & command)
{
  std::array<char, 128> buffer{};
  std::string result;
  auto * stream = popen(command.c_str(), "r");
  if (stream == nullptr) return result;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), stream) != nullptr)
    result += buffer.data();
  pclose(stream);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
  return result;
}

std::string repository_sha()
{
  constexpr const char * git_path = "/usr/bin/git";
  if (!std::filesystem::exists(git_path)) return "unavailable";
  return shell_text(
    "/usr/bin/git -C /home/kop/burn-your-bridges/sim-vision rev-parse HEAD 2>/dev/null");
}

std::uint64_t fnv1a64(const std::string & text)
{
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto byte : text) {
    hash ^= static_cast<unsigned char>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string read_text_file(const std::filesystem::path & path)
{
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void override_device(YAML::Node node, const std::string & device)
{
  if (!node || device.empty()) return;
  if (node.IsMap()) {
    for (auto iterator = node.begin(); iterator != node.end(); ++iterator) {
      if (iterator->first.as<std::string>() == "device") {
        iterator->second = device;
      } else {
        override_device(iterator->second, device);
      }
    }
  } else if (node.IsSequence()) {
    for (auto entry : node) override_device(entry, device);
  }
}

void copy_simulation_override(YAML::Node & target, const YAML::Node & overlay, const char * key)
{
  if (overlay[key]) target[key] = YAML::Clone(overlay[key]);
}

void copy_simulation_section_overrides(
  YAML::Node & target, const YAML::Node & overlay, const char * section,
  std::initializer_list<const char *> keys)
{
  if (!overlay[section]) return;
  for (const auto * key : keys) {
    if (overlay[section][key]) {
      target[section][key] = YAML::Clone(overlay[section][key]);
    }
  }
}

std::string session_config_path(
  const std::string & overlay_config_path, const std::string & session_dir,
  const std::string & device)
{
  auto yaml = YAML::LoadFile(kRobotConfigPath);
  const auto overlay = YAML::LoadFile(overlay_config_path);
  // Algorithm parameters always come from sentry.yaml. Only geometry/assembly values that
  // describe the ideal Talos camera may differ from the physical robot.
  for (const auto * key : {
         "yolov5_backend", "yaw_offset", "pitch_offset", "left_yaw_offset", "right_yaw_offset",
         "R_gimbal2imubody", "camera_matrix", "distort_coeffs", "R_camera2gimbal",
         "t_camera2gimbal"}) {
    copy_simulation_override(yaml, overlay, key);
  }
  // The rendered Daedalus image domain uses its own detector frontend. Keep PnP/solver,
  // tracking, prediction and fire policy from the real-robot sentry.yaml configuration.
  if (overlay["buff_detector"]) {
    for (const auto * key : {
           "backend", "model", "conf_threshold", "r_center_refine",
           "r_center_refine_threshold", "r_center_refine_radius_scale",
           "r_center_refine_kernel_size", "r_center_refine_hierarchy_filter",
           "r_center_refine_max_child_area_ratio"}) {
      if (overlay["buff_detector"][key]) {
        yaml["buff_detector"][key] = YAML::Clone(overlay["buff_detector"][key]);
      }
    }
  }
  // Simulation response tuning. These narrowly scoped values restore the previously
  // validated Daedalus warm-up, tracking and fire-gate behavior without replacing the
  // real-robot PnP, ballistic solver or any complete configuration subtree.
  if (overlay["fire_gap_time"]) {
    yaml["fire_gap_time"] = YAML::Clone(overlay["fire_gap_time"]);
  }
  copy_simulation_section_overrides(yaml, overlay, "predictor", {"small_predictor"});
  copy_simulation_section_overrides(
    yaml, overlay, "fitting",
    {"big_min_fit_size", "big_direction_lock_threshold", "small_direction_lock_threshold"});
  copy_simulation_section_overrides(
    yaml, overlay, "command_guard",
    {"max_yaw_rate", "max_pitch_rate", "max_fire_yaw_error", "max_fire_pitch_error",
     "big_single_leaf_fire_guard_enabled"});
  copy_simulation_section_overrides(
    yaml, overlay, "ballistic", {"energy_yaw_bias", "energy_pitch_bias"});
  override_device(yaml, device);
  YAML::Emitter emitter;
  emitter << yaml;
  const auto path = std::filesystem::path(session_dir) / "runtime_config.yaml";
  std::ofstream output(path);
  output << emitter.c_str() << '\n';
  return path.string();
}

std::string configured_inference_device(
  const std::string & config_path, const std::string & requested_device)
{
  if (!requested_device.empty()) return requested_device;
  const auto yaml = YAML::LoadFile(config_path);
  return yaml["device"] ? yaml["device"].as<std::string>() : "CPU";
}

std::string resolve_inference_device(const std::string & requested_device)
{
  if (requested_device != "GPU") return requested_device;
  try {
    ov::Core core;
    const auto devices = core.get_available_devices();
    const auto gpu_available = std::any_of(
      devices.begin(), devices.end(),
      [](const std::string & device) { return device == "GPU" || device.rfind("GPU.", 0) == 0; });
    if (gpu_available) return "GPU";
  } catch (const std::exception &) {
  }
  tools::logger()->warn(
    "[vision_sim_runner] OpenVINO GPU is unavailable; using CPU for this session. "
    "Use sim_model_probe to verify a future TensorRT/OpenVINO deployment.");
  return "CPU";
}

Eigen::Matrix3d matrix3(const std::array<float, 9> & values)
{
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col) result(row, col) = values.at(row * 3 + col);
  return result;
}

Eigen::Matrix3d quaternion_matrix(const std::array<float, 4> & values)
{
  return Eigen::Quaterniond(values[0], values[1], values[2], values[3])
    .normalized()
    .toRotationMatrix();
}

cv::Mat camera_matrix(const io::sim::CameraCalibration & calibration)
{
  return (
    cv::Mat_<double>(3, 3) << calibration.intrinsics[0], 0.0, calibration.intrinsics[2], 0.0,
    calibration.intrinsics[1], calibration.intrinsics[3], 0.0, 0.0, 1.0);
}

cv::Mat distortion(const io::sim::CameraCalibration & calibration)
{
  cv::Mat result(1, static_cast<int>(calibration.distortion.size()), CV_64F);
  for (int index = 0; index < result.cols; ++index)
    result.at<double>(0, index) = calibration.distortion.at(index);
  return result;
}

io::GimbalState feedback_as_gimbal_state(const io::sim::GimbalFeedback & feedback)
{
  return {
    feedback.yaw_rad,
    feedback.yaw_velocity_radps,
    feedback.pitch_rad,
    feedback.pitch_velocity_radps,
    feedback.bullet_speed_mps,
    static_cast<std::uint16_t>(feedback.projectile_count),
    feedback.camp};
}

io::sim::VisionCommand make_command(
  const auto_aim::Plan & plan, const io::sim::FramePacket & frame, std::uint64_t command_seq)
{
  return {
    frame.data.image.frame_seq,
    command_seq,
    frame.data.image.timestamp_ns,
    plan.yaw,
    plan.pitch,
    plan.yaw_vel,
    plan.pitch_vel,
    plan.yaw_acc,
    plan.pitch_acc,
    0.0F,
    static_cast<std::uint8_t>(plan.control),
    static_cast<std::uint8_t>(plan.fire),
    {}};
}

std::uint8_t team_for_color(auto_aim::Color color) { return color == auto_aim::Color::red ? 0 : 1; }

std::uint8_t label_for_armor_name(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::ArmorName::one:
      return 1;
    case auto_aim::ArmorName::two:
      return 2;
    case auto_aim::ArmorName::three:
      return 3;
    case auto_aim::ArmorName::four:
      return 4;
    case auto_aim::ArmorName::five:
      return 5;
    case auto_aim::ArmorName::sentry:
      return 0;
    case auto_aim::ArmorName::outpost:
      return 6;
    case auto_aim::ArmorName::base:
      return 7;
    default:
      return 255;
  }
}

double orientation_error_rad(
  const Eigen::Vector3d & observed_ypr, const std::array<float, 4> & truth_wxyz)
{
  const auto truth = Eigen::Quaterniond(truth_wxyz[0], truth_wxyz[1], truth_wxyz[2], truth_wxyz[3])
                       .normalized()
                       .toRotationMatrix();
  const auto truth_ypr = tools::eulers(truth, 2, 1, 0);
  Eigen::Vector3d delta;
  for (int axis = 0; axis < 3; ++axis)
    delta[axis] = tools::limit_rad(observed_ypr[axis] - truth_ypr[axis]);
  return delta.norm();
}

void evaluate_armors(
  const std::list<auto_aim::Armor> & armors, const io::sim::GroundTruthBatch & truth,
  const io::sim::Pose & gimbal_world, Stats & stats)
{
  for (const auto & armor : armors) {
    if (!armor.xyz_in_world.allFinite()) continue;
    const auto label = label_for_armor_name(armor.name);
    if (label == 255) continue;
    const auto team = team_for_color(armor.color);
    const io::sim::GroundTruthTarget * nearest = nullptr;
    double nearest_distance = std::numeric_limits<double>::infinity();
    for (std::uint32_t index = 0; index < truth.target_count; ++index) {
      const auto & candidate = truth.targets.at(index);
      if (candidate.team != team) continue;
      const auto relative =
        io::sim::world_position_relative_to_gimbal(candidate.position_m, gimbal_world);
      const Eigen::Vector3d position(relative[0], relative[1], relative[2]);
      const double distance = (armor.xyz_in_world - position).norm();
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest = &candidate;
      }
    }
    if (nearest == nullptr) continue;
    ++stats.pnp_samples;
    stats.pnp_position_error_m_sum += nearest_distance;
    stats.pnp_orientation_error_rad_sum +=
      orientation_error_rad(armor.ypr_in_world, nearest->quaternion_wxyz);
    if (nearest->armor_label == label) ++stats.detection_truth_matches;
  }
}

void write_report(
  const std::string & session_dir, const std::string & config_path, const std::string & scenario,
  bool hero_trajectory_model, const Stats & stats, std::uint64_t overflow_count, bool frame_timeout)
{
  const auto delivery = stats.polled_frames == 0
                          ? 0.0
                          : static_cast<double>(stats.accepted_frames) / stats.polled_frames;
  const auto latency =
    stats.accepted_frames == 0 ? 0.0 : stats.end_to_end_latency_ms_sum / stats.accepted_frames;
  const auto hit_rate =
    stats.hit_events == 0 ? 0.0 : static_cast<double>(stats.correct_hits) / stats.hit_events;
  const auto rune_correct_target_rate =
    stats.rune_hits == 0 ? 0.0
                         : static_cast<double>(stats.rune_correct_target_hits) / stats.rune_hits;
  const auto rune_wrong_target_rate =
    stats.rune_hits == 0 ? 0.0
                         : static_cast<double>(stats.rune_wrong_target_hits) / stats.rune_hits;
  const auto rune_timeout_rate =
    stats.rune_hits == 0 ? 0.0 : static_cast<double>(stats.rune_timeout_hits) / stats.rune_hits;
  const auto detection_match_rate =
    stats.auto_aim_detections == 0
      ? 0.0
      : static_cast<double>(stats.detection_truth_matches) / stats.auto_aim_detections;
  const auto pnp_position_error =
    stats.pnp_samples == 0 ? 0.0 : stats.pnp_position_error_m_sum / stats.pnp_samples;
  const auto pnp_orientation_error =
    stats.pnp_samples == 0 ? 0.0 : stats.pnp_orientation_error_rad_sum / stats.pnp_samples;
  const auto tracking_continuity =
    stats.auto_aim_frames == 0 ? 0.0
                               : static_cast<double>(stats.tracked_frames) / stats.auto_aim_frames;
  const auto gimbal_follow_error =
    stats.gimbal_follow_samples == 0
      ? 0.0
      : stats.gimbal_follow_error_rad_sum / stats.gimbal_follow_samples;
  json report = {
    {"kind", "simulation-report"},
    {"disclaimer", "simulation result; not a competition-grade real-robot hit-rate claim"},
    {"scenario", scenario},
    {"config_path", config_path},
    {"protocol", "talos-v3"},
    {"simulator", "Daedalus"},
    {"hero_trajectory_model", hero_trajectory_model},
    {"frame_timeout", frame_timeout},
    {"metrics",
     {{"image_delivery_rate", delivery},
      {"end_to_end_latency_ms_mean", latency},
      {"detection_match_rate", detection_match_rate},
      {"pnp_position_error_m", pnp_position_error},
      {"pnp_orientation_error_rad", pnp_orientation_error},
      {"tracking_continuity", tracking_continuity},
      {"gimbal_follow_error_rad", gimbal_follow_error},
      {"hit_rate", hit_rate},
      {"correct_hits", stats.correct_hits},
      {"armor_hits", stats.armor_hits},
      {"rune_hits", stats.rune_hits},
      {"rune_correct_target_rate", rune_correct_target_rate},
      {"rune_wrong_target_rate", rune_wrong_target_rate},
      {"rune_timeout_rate", rune_timeout_rate},
      {"hit_event_ring_overflow", overflow_count}}},
    {"counters",
     {{"polled_frames", stats.polled_frames},
      {"accepted_frames", stats.accepted_frames},
      {"rejected_frames", stats.rejected_frames},
      {"auto_aim_detections", stats.auto_aim_detections},
      {"detection_truth_matches", stats.detection_truth_matches},
      {"pnp_samples", stats.pnp_samples},
      {"gimbal_follow_samples", stats.gimbal_follow_samples},
      {"auto_aim_commands", stats.auto_aim_commands},
      {"small_buff_commands", stats.small_buff_commands},
      {"big_buff_commands", stats.big_buff_commands},
      {"rune_correct_target_hits", stats.rune_correct_target_hits},
      {"rune_wrong_target_hits", stats.rune_wrong_target_hits},
      {"rune_timeout_hits", stats.rune_timeout_hits},
      {"last_frame_seq", stats.last_frame_seq},
      {"last_command_seq", stats.last_command_seq}}}};
  std::ofstream output(std::filesystem::path(session_dir) / "summary.json");
  output << report.dump(2) << '\n';
}

void write_metadata(
  const std::string & session_dir, const std::string & config_path, const std::string & scenario,
  const std::string & inference_device, bool physical_fire_enabled)
{
  const auto config_text = read_text_file(config_path);
  json metadata = {
    {"kind", "simulation-session-metadata"},
    {"scenario", scenario},
    {"burn_your_bridges_sha", repository_sha()},
    {"algorithm_config_hash_fnv1a64", fnv1a64(config_text)},
    {"inference_device", inference_device},
    {"physical_fire_enabled", physical_fire_enabled},
    {"simulator", "Daedalus"},
    {"protocol", "talos-v3"},
    {"units", "ROS Z-up, m, s, rad, wxyz"},
    {"policy", "session artifacts are append-only; no baseline or source is modified"}};
  std::ofstream output(std::filesystem::path(session_dir) / "metadata.json");
  output << metadata.dump(2) << '\n';
}

void print_dry_run(
  const std::string & config_path, const std::string & scenario, const std::string & meta_path,
  const std::string & image_pool_path, const std::string & inference_device, bool physical_fire_enabled)
{
  const json dry_run = {
    {"kind", "daedalus-talos-dry-run"},
    {"simulator", "Daedalus"},
    {"scenario", scenario},
    {"algorithm_config_path", kRobotConfigPath},
    {"simulation_overlay_path", config_path},
    {"meta_path", meta_path},
    {"image_pool_path", image_pool_path},
    {"inference_device_override", inference_device},
    {"physical_fire_enabled", physical_fire_enabled},
    {"required_protocol", "Talos v3: 1440x1080 RGB, ROS Z-up, m, s, rad, wxyz"},
    {"truth_policy", "ground truth is diagnostic-only and never feeds RGB-mode algorithms"}};
  std::cout << dry_run.dump(2) << '\n';
}

json truth_targets_json(const io::sim::GroundTruthBatch & truth)
{
  json targets = json::array();
  for (std::uint32_t index = 0; index < truth.target_count; ++index) {
    const auto & target = truth.targets.at(index);
    targets.push_back(
      {{"target_id", target.target_id},
       {"team", target.team},
       {"armor_label", target.armor_label},
       {"robot_type", target.robot_type},
       {"is_outpost", target.is_outpost},
       {"position_m", target.position_m},
       {"quaternion_wxyz", target.quaternion_wxyz},
       {"velocity_mps", target.velocity_mps},
       {"yaw_rate_radps", target.yaw_rate_radps}});
  }
  return targets;
}

json truth_runes_json(const io::sim::GroundTruthBatch & truth)
{
  json runes = json::array();
  for (std::uint32_t index = 0; index < truth.rune_count; ++index) {
    const auto & rune = truth.runes.at(index);
    runes.push_back(
      {{"rune_id", rune.rune_id},
       {"team", rune.team},
       {"mode", rune.mode},
       {"mechanism_state", rune.mechanism_state},
       {"direction", rune.direction},
       {"center_m", rune.center_m},
       {"quaternion_wxyz", rune.quaternion_wxyz},
       {"radius_m", rune.radius_m},
       {"angle_rad", rune.angle_rad},
       {"angular_velocity_radps", rune.angular_velocity_radps},
       {"sine_amplitude", rune.sine_amplitude},
       {"sine_omega", rune.sine_omega},
       {"sine_phase", rune.sine_phase},
       {"active_blade_id", rune.active_blade_id},
       {"target_activations", rune.target_activations}});
  }
  return runes;
}

void write_geometry_sample(std::ofstream & output, const io::sim::FrameData & frame)
{
  const json sample = {
    {"frame_seq", frame.image.frame_seq},         {"timestamp_ns", frame.image.timestamp_ns},
    {"target_count", frame.truth.target_count},   {"rune_count", frame.truth.rune_count},
    {"targets", truth_targets_json(frame.truth)}, {"runes", truth_runes_json(frame.truth)}};
  output << sample.dump() << '\n';
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>("@config-path");
  const auto scenario = cli.get<std::string>("scenario");
  const auto meta_path = cli.get<std::string>("meta-path");
  const auto image_pool_path = cli.get<std::string>("image-pool-path");
  const auto requested_device = cli.get<std::string>("device");
  const auto configured_device = configured_inference_device(config_path, requested_device);
  const bool physical_fire_enabled = cli.get<bool>("enable-fire");
  if (cli.get<bool>("dry-run")) {
    print_dry_run(
      config_path, scenario, meta_path, image_pool_path, configured_device,
      cli.get<bool>("enable-fire"));
    return 0;
  }
  const auto inference_device = resolve_inference_device(configured_device);
  const auto session_dir = make_session_directory(cli.get<std::string>("session-dir"), scenario);
  const auto max_frames = cli.get<int>("max-frames");
  const auto frame_timeout_ms = cli.get<int>("frame-timeout-ms");
  if (frame_timeout_ms < 0) {
    std::cerr << "--frame-timeout-ms must be non-negative\n";
    return 3;
  }
  const auto runtime_config_path = session_config_path(config_path, session_dir, inference_device);
  write_metadata(
    session_dir, runtime_config_path, scenario, inference_device, physical_fire_enabled);
  std::ofstream frame_log(std::filesystem::path(session_dir) / "frames.jsonl");
  std::ofstream command_log(std::filesystem::path(session_dir) / "commands.jsonl");
  std::ofstream truth_log(std::filesystem::path(session_dir) / "ground_truth.jsonl");
  std::ofstream hit_log(std::filesystem::path(session_dir) / "hits.jsonl");
  std::ofstream geometry_log(std::filesystem::path(session_dir) / "geometry.jsonl");

  io::sim::TalosClient client(meta_path, image_pool_path);
  if (!client.connected()) {
    tools::logger()->error("[vision_sim_runner] {}", client.error());
    return 2;
  }

  auto yaml = tools::load(runtime_config_path);
  const bool hero =
    yaml["trajectory_model"] && yaml["trajectory_model"].as<std::string>() == "hero";
  std::optional<auto_aim::YOLO> yolo;
  std::optional<auto_aim::Runtime> auto_aim_runtime;
  std::optional<auto_buff::Buff_Detector> buff_detector;
  std::optional<auto_buff::Solver> buff_solver;
  std::optional<auto_buff::Aimer> buff_aimer;
  const bool gui_debug = cli.get<bool>("gui-debug");
  const bool record_raw = cli.get<bool>("record");
  const bool record_debug = cli.get<bool>("record-debug");
  const bool truth_overlay = cli.get<bool>("truth-overlay");
  const bool debug_enabled = gui_debug || record_debug;
  std::optional<tools::Recorder> raw_recorder;
  std::optional<tools::Recorder> debug_recorder;
  if (record_raw) {
    tools::RecorderOptions options;
    options.fps = kTalosFrameRate;
    options.output_mode = tools::RecorderOutputMode::session_dir;
    options.output_path = (std::filesystem::path(session_dir) / "raw").string();
    options.config_path = runtime_config_path;
    raw_recorder.emplace(options);
  }
  if (record_debug) {
    tools::RecorderOptions options;
    options.fps = kTalosFrameRate;
    options.output_mode = tools::RecorderOutputMode::session_dir;
    options.output_path = (std::filesystem::path(session_dir) / "debug").string();
    options.config_path = runtime_config_path;
    debug_recorder.emplace(options);
  }
  std::optional<tools::Plotter> plotter;
  if (gui_debug) plotter.emplace();

  Stats stats;
  std::map<std::uint64_t, std::pair<float, float>> commanded_angles;
  auto last_frame_received = std::chrono::steady_clock::now();
  auto last_sim_buff_fire = std::chrono::steady_clock::now() - kSimBuffFireInterval;
  auto sim_buff_fire_until = std::chrono::steady_clock::time_point::min();
  bool frame_timeout = false;
  tools::Exiter exiter;
  while (!exiter.exit()) {
    ++stats.polled_frames;
    auto frame = client.try_read_frame();
    if (!frame.has_value()) {
      ++stats.rejected_frames;
      if (
        frame_timeout_ms > 0 && std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - last_frame_received)
                                    .count() >= frame_timeout_ms) {
        tools::logger()->error(
          "[vision_sim_runner] no new Talos frame for {} ms; check that Daedalus is running with "
          "the talos feature and DAEDALUS_FORCE_TALOS_CAPTURE=1",
          frame_timeout_ms);
        frame_timeout = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    last_frame_received = std::chrono::steady_clock::now();
    ++stats.accepted_frames;
    stats.last_frame_seq = frame->data.image.frame_seq;
    write_geometry_sample(geometry_log, frame->data);
    frame_log << json({{"frame_seq", frame->data.image.frame_seq},
                       {"timestamp_ns", frame->data.image.timestamp_ns},
                       {"mode", static_cast<std::uint8_t>(frame->data.feedback.mode)},
                       {"camp", frame->data.feedback.camp},
                       {"robot_type", static_cast<std::uint8_t>(frame->data.feedback.robot_type)},
                       {"projectile_count", frame->data.feedback.projectile_count},
                       {"gimbal_yaw_rad", frame->data.feedback.yaw_rad},
                       {"gimbal_pitch_rad", frame->data.feedback.pitch_rad},
                       {"gimbal_yaw_velocity_radps", frame->data.feedback.yaw_velocity_radps},
                       {"gimbal_pitch_velocity_radps", frame->data.feedback.pitch_velocity_radps},
                       {"simulation_subscription_enabled",
                        frame->data.feedback.simulation_subscription_enabled},
                       {"last_command_seq", frame->data.feedback.last_command_seq},
                       {"gimbal_world_position_m", frame->data.gimbal_world.position_m},
                       {"gimbal_world_quaternion_wxyz",
                        frame->data.gimbal_world.quaternion_wxyz}})
                   .dump()
              << '\n';
    truth_log << json({{"frame_seq", frame->data.truth.frame_seq},
                       {"timestamp_ns", frame->data.truth.timestamp_ns},
                       {"target_count", frame->data.truth.target_count},
                       {"rune_count", frame->data.truth.rune_count},
                       {"targets", truth_targets_json(frame->data.truth)},
                       {"runes", truth_runes_json(frame->data.truth)}})
                   .dump()
              << '\n';
    const auto & calibration = frame->data.camera;
    const auto feedback_command = commanded_angles.find(frame->data.feedback.last_command_seq);
    if (feedback_command != commanded_angles.end()) {
      const auto yaw_error = tools::limit_rad(
        static_cast<double>(frame->data.feedback.yaw_rad - feedback_command->second.first));
      const auto pitch_error = tools::limit_rad(
        static_cast<double>(frame->data.feedback.pitch_rad - feedback_command->second.second));
      stats.gimbal_follow_error_rad_sum += std::hypot(yaw_error, pitch_error);
      ++stats.gimbal_follow_samples;
    }
    const auto R_camera2gimbal = matrix3(calibration.R_camera2gimbal_row_major);
    const Eigen::Vector3d t_camera2gimbal(
      calibration.t_camera2gimbal_m[0], calibration.t_camera2gimbal_m[1],
      calibration.t_camera2gimbal_m[2]);
    const auto R_gimbal2world = quaternion_matrix(frame->data.gimbal_world.quaternion_wxyz);
    const auto K = camera_matrix(calibration);
    const auto D = distortion(calibration);
    auto plan = auto_aim::Plan{};
    const auto mode = frame->data.feedback.mode;
    cv::Mat debug_image;
    std::optional<auto_aim::RuntimeDebugResult> auto_aim_debug;
    std::optional<auto_buff::PowerRune> buff_debug;
    std::optional<json> buff_command_debug;
    if (mode == io::sim::AlgorithmMode::auto_aim) {
      if (!yolo.has_value()) {
        auto_aim_runtime.emplace(runtime_config_path);
        yolo.emplace(runtime_config_path, false);
      }
      ++stats.auto_aim_frames;
      auto armors = yolo->detect(frame->image_bgr, static_cast<int>(frame->data.image.frame_seq));
      stats.auto_aim_detections += armors.size();
      const auto gimbal = feedback_as_gimbal_state(frame->data.feedback);
      auto runtime_frame = auto_aim::RuntimeFrame{};
      runtime_frame.timestamp = frame->timestamp;
      runtime_frame.pose_source = auto_aim::RuntimePoseSource::gimbal_to_world_matrix;
      runtime_frame.R_gimbal2world = R_gimbal2world;
      runtime_frame.bullet_speed = gimbal.bullet_speed;
      runtime_frame.pitch = gimbal.pitch;
      runtime_frame.camp = gimbal.camp;
      runtime_frame.camera_matrix = &K;
      runtime_frame.distort_coeffs = &D;
      runtime_frame.R_camera2gimbal = &R_camera2gimbal;
      runtime_frame.t_camera2gimbal = &t_camera2gimbal;
      auto_aim_debug.emplace(auto_aim_runtime->process_with_debug(armors, runtime_frame));
      evaluate_armors(armors, frame->data.truth, frame->data.gimbal_world, stats);
      plan = auto_aim_debug->plan;
      stats.tracked_frames += plan.control;
      ++stats.auto_aim_commands;
      if (debug_enabled) {
        debug_image = sim::draw_auto_aim_debug(
          frame->image_bgr, armors, static_cast<int>(frame->data.image.frame_seq),
          &auto_aim_debug->target, &auto_aim_runtime->solver(), &auto_aim_debug->aim_xyza,
          truth_overlay ? &frame->data.truth : nullptr);
      }
    } else if (
      !hero &&
      (mode == io::sim::AlgorithmMode::small_buff || mode == io::sim::AlgorithmMode::big_buff)) {
      if (!buff_detector.has_value()) {
        buff_detector.emplace(runtime_config_path);
        buff_solver.emplace(runtime_config_path);
        buff_aimer.emplace(runtime_config_path);
      }
      buff_solver->set_runtime_calibration(K, D, R_camera2gimbal, t_camera2gimbal);
      buff_solver->set_R_gimbal2world_matrix(R_gimbal2world);
      // camp is self camp: red self detects blue (label 1), blue self detects red (label 0).
      buff_detector->set_expected_color_label(frame->data.feedback.camp == 0 ? 1 : 0);
      auto image = frame->image_bgr;
      auto rune = buff_detector->detect(image);
      if (rune.has_value()) {
        buff_solver->solve(rune, &frame->timestamp);
        const auto elapsed =
          std::chrono::duration<double>(frame->timestamp.time_since_epoch()).count();
        const auto gimbal = feedback_as_gimbal_state(frame->data.feedback);
        const auto buff_mode = mode == io::sim::AlgorithmMode::small_buff
                                 ? auto_buff::BuffMode::SMALL
                                 : auto_buff::BuffMode::BIG;
        plan = buff_aimer->mpc_aim(*rune, elapsed, elapsed, gimbal, buff_mode);
        const auto prediction = buff_aimer->prediction_snapshot();
        const auto & ballistic = buff_aimer->last_ballistic_result();
        const double current_world_yaw = std::atan2(R_gimbal2world(1, 0), R_gimbal2world(0, 0));
        const double current_world_pitch = std::atan2(
          -R_gimbal2world(2, 0), std::hypot(R_gimbal2world(0, 0), R_gimbal2world(1, 0)));
        const double yaw_error =
          tools::limit_rad(static_cast<double>(plan.yaw) - current_world_yaw);
        const double pitch_error = static_cast<double>(plan.pitch) - current_world_pitch;
        const bool aligned = std::abs(yaw_error) <= kSimBuffFireToleranceRad &&
                             std::abs(pitch_error) <= kSimBuffFireToleranceRad;
        const bool planner_fire = plan.fire;
        const auto now = std::chrono::steady_clock::now();
        const bool sim_fire_ready =
          plan.control && prediction.fitted && ballistic.valid && aligned;
        if ((planner_fire || sim_fire_ready) &&
            now - last_sim_buff_fire >= kSimBuffFireInterval) {
          last_sim_buff_fire = now;
          sim_buff_fire_until = now + kSimBuffFirePulse;
        }
        plan.fire = now < sim_buff_fire_until;
        buff_command_debug = json{
          {"mode", mode == io::sim::AlgorithmMode::small_buff ? "SMALL" : "BIG"},
          {"prediction_fitted", prediction.fitted},
          {"prediction_samples", prediction.sample_count},
          {"ballistic_valid", ballistic.valid},
          {"current_world_yaw_rad", current_world_yaw},
          {"current_world_pitch_rad", current_world_pitch},
          {"yaw_error_rad", yaw_error},
          {"pitch_error_rad", pitch_error},
          {"aligned", aligned},
          {"planner_fire", planner_fire},
          {"sim_fire_ready", sim_fire_ready},
          {"sim_fire_pulse", plan.fire}};
      }
      buff_debug = rune;
      if (mode == io::sim::AlgorithmMode::small_buff)
        ++stats.small_buff_commands;
      else
        ++stats.big_buff_commands;
      if (debug_enabled) {
        debug_image = sim::draw_buff_debug(
          frame->image_bgr, buff_debug, static_cast<int>(frame->data.image.frame_seq),
          &*buff_solver, &*buff_aimer, truth_overlay ? &frame->data.truth : nullptr);
      }
    }

    const auto world_command = make_command(plan, *frame, ++stats.last_command_seq);
    auto command = io::sim::localize_world_command(
      world_command, frame->data, physical_fire_enabled);
    client.send_command(command);
    if (command.control != 0) {
      commanded_angles.emplace(
        command.command_seq, std::make_pair(command.yaw_rad, command.pitch_rad));
    }
    while (commanded_angles.size() > 1024) commanded_angles.erase(commanded_angles.begin());
    json command_sample = {
      {"frame_seq", command.frame_seq},
      {"command_seq", command.command_seq},
      {"timestamp_ns", command.timestamp_ns},
      {"control", command.control},
      {"fire", command.fire},
      {"planner_fire", world_command.fire},
      {"physical_fire_enabled", physical_fire_enabled},
      {"world_yaw_rad", world_command.yaw_rad},
      {"world_pitch_rad", world_command.pitch_rad},
      {"yaw_rad", command.yaw_rad},
      {"pitch_rad", command.pitch_rad},
      {"yaw_velocity_radps", command.yaw_velocity_radps},
      {"pitch_velocity_radps", command.pitch_velocity_radps},
      {"yaw_acceleration_radps2", command.yaw_acceleration_radps2},
      {"pitch_acceleration_radps2", command.pitch_acceleration_radps2}};
    if (auto_aim_debug.has_value() && auto_aim_debug->target.has_value()) {
      const auto & target = *auto_aim_debug->target;
      const auto ekf_x = target.ekf_x();
      json armor_xyza = json::array();
      for (const auto & xyza : target.armor_xyza_list()) {
        armor_xyza.push_back({xyza.x(), xyza.y(), xyza.z(), xyza.w()});
      }
      const json planner_aim_xyza = json::array(
        {auto_aim_debug->aim_xyza.x(), auto_aim_debug->aim_xyza.y(), auto_aim_debug->aim_xyza.z(),
         auto_aim_debug->aim_xyza.w()});
      command_sample["auto_aim_debug"] = {
        {"target_name", auto_aim::ARMOR_NAMES[target.name]},
        {"jumped", target.jumped},
        {"last_observed_armor_id", target.last_id},
        {"track_epoch", target.tracking_info().track_epoch},
        {"observation_seq", target.tracking_info().observation_seq},
        {"ekf_state",
         {{"center_x_m", ekf_x[0]},
          {"center_vx_mps", ekf_x[1]},
          {"center_y_m", ekf_x[2]},
          {"center_vy_mps", ekf_x[3]},
          {"center_z_m", ekf_x[4]},
          {"center_vz_mps", ekf_x[5]},
          {"armor_yaw_rad", ekf_x[6]},
          {"armor_yaw_rate_radps", ekf_x[7]},
          {"radius_1_m", ekf_x[8]},
          {"radius_delta_m", ekf_x[9]},
          {"height_delta_m", ekf_x[10]}}},
        {"armor_xyza", armor_xyza},
         {"planner_aim_xyza", planner_aim_xyza}};
    }
    if (buff_command_debug.has_value()) command_sample["buff_debug"] = *buff_command_debug;
    command_log << command_sample.dump() << '\n';
    const Eigen::Quaterniond camera_q(R_gimbal2world);
    if (raw_recorder) raw_recorder->record(frame->image_bgr, camera_q, frame->timestamp);
    if (debug_recorder && !debug_image.empty()) {
      debug_recorder->record(debug_image, camera_q, frame->timestamp);
    }
    if (gui_debug && !debug_image.empty()) {
      const auto enemy_color = frame->data.feedback.camp == 0 ? "BLUE" : "RED";
      if (
        mode == io::sim::AlgorithmMode::small_buff ||
        mode == io::sim::AlgorithmMode::big_buff) {
        const auto buff_mode_text =
          mode == io::sim::AlgorithmMode::small_buff ? "buff-mode=SMALL" : "buff-mode=BIG";
        cv::putText(
          debug_image, buff_mode_text, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
          cv::Scalar(80, 255, 255), 2, cv::LINE_AA);
      }
      cv::putText(
        debug_image, std::string("detect-color=") + enemy_color,
        mode == io::sim::AlgorithmMode::auto_aim ? cv::Point(20, 40) : cv::Point(20, 80),
        cv::FONT_HERSHEY_SIMPLEX, 0.9,
        frame->data.feedback.camp == 0 ? cv::Scalar(255, 80, 80) : cv::Scalar(80, 80, 255), 2,
        cv::LINE_AA);
      cv::Mat display;
      cv::resize(debug_image, display, {}, 0.5, 0.5);
      cv::imshow(mode == io::sim::AlgorithmMode::auto_aim ? "sim auto aim" : "sim buff", display);
      if (cv::waitKey(1) == 'q') break;
      if (plotter) {
        json plot_data = {
          {"frame_seq", frame->data.image.frame_seq},
          {"mode", static_cast<int>(mode)},
          {"plan_yaw", plan.yaw},
          {"plan_yaw_vel", plan.yaw_vel},
          {"plan_yaw_acc", plan.yaw_acc},
          {"plan_pitch", plan.pitch},
          {"plan_pitch_vel", plan.pitch_vel},
          {"plan_pitch_acc", plan.pitch_acc},
          {"control", plan.control ? 1 : 0},
          {"fire", plan.fire ? 1 : 0},
          {"gimbal_yaw", frame->data.feedback.yaw_rad},
          {"gimbal_yaw_vel", frame->data.feedback.yaw_velocity_radps},
          {"gimbal_pitch", frame->data.feedback.pitch_rad},
          {"gimbal_pitch_vel", frame->data.feedback.pitch_velocity_radps},
          {"bullet_speed", frame->data.feedback.bullet_speed_mps}};
        if (auto_aim_debug.has_value() && auto_aim_debug->target.has_value()) {
          const auto & target = *auto_aim_debug->target;
          const auto x = target.ekf_x();
          plot_data["target_x"] = x[0];
          plot_data["target_vx"] = x[1];
          plot_data["target_y"] = x[2];
          plot_data["target_vy"] = x[3];
          plot_data["target_z"] = x[4];
          plot_data["target_vz"] = x[5];
          plot_data["target_yaw"] = x[6];
          plot_data["target_yaw_vel"] = x[7];
          plot_data["target_radius"] = x[8];
          plot_data["target_radius_delta"] = x[9];
          plot_data["target_height_delta"] = x[10];
          plot_data["target_last_id"] = target.last_id;

          const auto & ekf_data = target.ekf().data;
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

          plot_data["planner_aim_x"] = auto_aim_debug->aim_xyza[0];
          plot_data["planner_aim_y"] = auto_aim_debug->aim_xyza[1];
          plot_data["planner_aim_z"] = auto_aim_debug->aim_xyza[2];
        }
        if (buff_aimer.has_value()) {
          const auto snapshot = buff_aimer->prediction_snapshot();
          plot_data["buff_samples"] = snapshot.sample_count;
          plot_data["buff_a"] = snapshot.energy_tri.a;
          plot_data["buff_w"] = snapshot.energy_tri.w;
        }
        plotter->plot(plot_data);
      }
    }
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
    stats.end_to_end_latency_ms_sum +=
      std::max<double>(0.0, (now_ns - frame->data.image.timestamp_ns) / 1e6);
    for (const auto & event : client.read_hit_events()) {
      ++stats.hit_events;
      stats.correct_hits += event.correct != 0;
      stats.armor_hits += event.hit_type == io::sim::HitType::armor;
      stats.rune_hits += event.hit_type == io::sim::HitType::rune;
      if (event.hit_type == io::sim::HitType::rune) {
        stats.rune_correct_target_hits += event.outcome >= 2;
        stats.rune_wrong_target_hits += event.outcome == 1;
        stats.rune_timeout_hits += event.outcome == 0;
      }
      hit_log << json({{"event_seq", event.event_seq},
                       {"command_seq", event.command_seq},
                       {"hit_timestamp_ns", event.hit_timestamp_ns},
                       {"target_id", event.target_id},
                       {"blade_id", event.blade_id},
                       {"hit_type", static_cast<std::uint8_t>(event.hit_type)},
                       {"correct", event.correct},
                       {"outcome", event.outcome}})
                   .dump()
              << '\n';
    }
    if (max_frames > 0 && stats.accepted_frames >= static_cast<std::uint64_t>(max_frames)) break;
  }
  write_report(
    session_dir, runtime_config_path, scenario, hero, stats, client.hit_event_overflow_count(),
    frame_timeout);
  tools::logger()->info("[vision_sim_runner] wrote report to {}", session_dir);
  if (gui_debug) cv::destroyAllWindows();
  return frame_timeout ? 2 : 0;
}
