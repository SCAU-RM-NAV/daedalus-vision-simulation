#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tasks/auto_buff/angle_estimate.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

// 代码执行
//cd /home/lcg/burn-your-bridges
//./build/auto_buff_detect_test /home/lcg/下载/red_buffs.mp4 --config-path configs/standard3.yaml --mode single

namespace
{
constexpr double kFallbackVideoFps = 30.0;
constexpr double kDefaultBulletSpeed = 24.0;
constexpr double kLeafAngleStep = 2.0 * CV_PI / 5.0;
constexpr bool kEnablePerfLog = true;      // 是否输出性能日志（仅当前测试入口生效）
constexpr double kPerfLogPeriodSec = 1.0;  // 性能日志输出周期，单位：秒

const std::string keys =
  "{help h usage ? | | Show command line help }"
  "{source         | video | Input source: video or camera }"
  "{pipeline       | full | Pipeline: detect or full }"
  "{view           | result | View: none, detect, refine or result }"
  "{r-center-view  | off | R-center view: off, roi, binary, contours or all }"
  "{config-path c  | | Config YAML path }"
  "{input-prefix   | | Legacy replay prefix for <path>.avi and <path>.txt }"
  "{record-dir     | | Session directory for replay or recording }"
  "{start-index s  | 0 | Start frame index }"
  "{end-index e    | 0 | End frame index, inclusive. 0 means no limit }"
  "{max-frames     | 0 | Maximum number of input frames to process }"
  "{buff-mode      | auto | Buff mode: small, big or auto }"
  "{record         | off | Record switch: off or on }"
  "{self-test      | off | Run deterministic auto_buff regression checks }"
  "{@legacy-input  | | Deprecated positional legacy replay prefix }";

enum class SourceKind
{
  Video,
  Camera
};

enum class PipelineKind
{
  Detect,
  Full
};

enum class ViewKind
{
  None,
  Detect,
  Refine,
  Result
};

enum class BuffModeArg
{
  Auto,
  Small,
  Big
};

struct CliOptions
{
  SourceKind source = SourceKind::Video;
  PipelineKind pipeline = PipelineKind::Full;
  ViewKind view = ViewKind::Result;
  auto_buff::RCenterDebugView r_center_view = auto_buff::RCenterDebugView::Off;
  std::string config_path;
  std::string input_prefix;
  std::string record_dir;
  int start_index = 0;
  int end_index = 0;
  int max_frames = 0;
  BuffModeArg buff_mode = BuffModeArg::Auto;
  bool record_enabled = false; 
  bool self_test = false;
};

struct ReplaySample
{
  int frame_index = -1;
  double t_capture_sec = 0.0;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  double bullet_speed = kDefaultBulletSpeed;
  std::string vision_mode = "unknown";
};

struct SessionMetadata
{
  std::string config_path;
  double fps = 0.0;
  int record_version = 0;
};

struct VideoReplaySession
{
  cv::VideoCapture video;
  std::vector<ReplaySample> samples;
  SessionMetadata metadata;
  double fps = kFallbackVideoFps;
  int total_frames = 0;
};

struct FullPipelineContext
{
  tools::Plotter & plotter;
  auto_buff::Solver & solver;
  auto_buff::Aimer & aimer;
};

const char * predictor_mode_text(auto_buff::PredictorBackend backend)
{
  switch (backend) {
    case auto_buff::PredictorBackend::BIG_CERES:
      return "大符";
    case auto_buff::PredictorBackend::SMALL_CERES:
    case auto_buff::PredictorBackend::SMALL_TARGET_EKF:
      return "小符";
  }
  return "未知";
}

const char * predictor_backend_text(auto_buff::PredictorBackend backend)
{
  switch (backend) {
    case auto_buff::PredictorBackend::BIG_CERES:
    case auto_buff::PredictorBackend::SMALL_CERES:
      return "Ceres";
    case auto_buff::PredictorBackend::SMALL_TARGET_EKF:
      return "EKF";
  }
  return "未知";
}

struct PredictorPerfBucket
{
  int updates = 0;
  double total_time_ms = 0.0;
  std::size_t latest_sample_count = 0;
};

struct BuffPerfLogState
{
  int submitted = 0;
  int results = 0;
  int busy_drop = 0;
  int empty_results = 0;
  int fit_input_frames = 0;
  double total_detect_ms = 0.0;
  double preprocess_ms = 0.0;
  double infer_ms = 0.0;
  double postprocess_ms = 0.0;
  std::size_t last_stale_drop_total = 0;
  std::size_t last_out_of_order_total = 0;
  PredictorPerfBucket big_ceres;
  PredictorPerfBucket small_ceres;
  PredictorPerfBucket small_ekf;

  void record_submit(bool ok)
  {
    if (ok) {
      ++submitted;
    } else {
      ++busy_drop;
    }
  }

  void record_detect(
    double detect_dt_ms, const auto_buff::BuffDetectPerfStats & perf_stats, bool empty_result)
  {
    ++results;
    total_detect_ms += detect_dt_ms;
    preprocess_ms += perf_stats.preprocess_dt_ms;
    infer_ms += perf_stats.infer_dt_ms;
    postprocess_ms += perf_stats.postprocess_dt_ms;
    if (empty_result) ++empty_results;
  }

  void record_fit_input() { ++fit_input_frames; }

  void record_predictor(const auto_buff::AngleEstimate::Snapshot & snapshot)
  {
    double used_time_ms = 0.0;
    PredictorPerfBucket * bucket = nullptr;
    switch (snapshot.predictor_backend) {
      case auto_buff::PredictorBackend::BIG_CERES:
        if (!snapshot.has_ceres_solve_time) return;
        used_time_ms = snapshot.last_ceres_solve_time_ms;
        bucket = &big_ceres;
        break;
      case auto_buff::PredictorBackend::SMALL_CERES:
        if (!snapshot.has_ceres_solve_time) return;
        used_time_ms = snapshot.last_ceres_solve_time_ms;
        bucket = &small_ceres;
        break;
      case auto_buff::PredictorBackend::SMALL_TARGET_EKF:
        used_time_ms = snapshot.last_small_ekf_update_time_ms;
        bucket = &small_ekf;
        break;
    }
    if (used_time_ms <= 0.0 || bucket == nullptr) return;
    ++bucket->updates;
    bucket->total_time_ms += used_time_ms;
    bucket->latest_sample_count = snapshot.sample_count;
  }

  void log_predictor_bucket(
    auto_buff::PredictorBackend backend, const PredictorPerfBucket & bucket) const
  {
    if (bucket.updates <= 0) return;
    tools::logger()->info(
      "[能量机关][预测性能] 模式={} 后端={} 周期更新={} 平均耗时={:.2f}ms 最新样本数={}",
      predictor_mode_text(backend), predictor_backend_text(backend), bucket.updates,
      bucket.total_time_ms / static_cast<double>(bucket.updates), bucket.latest_sample_count);
  }

  void log_and_reset(
    const auto_buff::Buff_Detector & detector, double report_dt_sec, bool include_predictor)
  {
    const auto stale_total = detector.async_stale_drop_count();
    const auto out_of_order_total = detector.async_out_of_order_drop_count();
    const auto stale_delta = stale_total - last_stale_drop_total;
    const auto out_of_order_delta = out_of_order_total - last_out_of_order_total;
    last_stale_drop_total = stale_total;
    last_out_of_order_total = out_of_order_total;

    const double avg_total_ms = results > 0 ? total_detect_ms / static_cast<double>(results) : 0.0;
    const double avg_preprocess_ms =
      results > 0 ? preprocess_ms / static_cast<double>(results) : 0.0;
    const double avg_infer_ms = results > 0 ? infer_ms / static_cast<double>(results) : 0.0;
    const double avg_postprocess_ms =
      results > 0 ? postprocess_ms / static_cast<double>(results) : 0.0;

    tools::logger()->info(
      "[能量机关][检测性能] 模式=异步 提交帧率={:.2f}fps 返回帧率={:.2f}fps 总耗时={:.2f}ms "
      "预处理={:.2f}ms 推理={:.2f}ms 后处理={:.2f}ms 忙丢帧={} 空结果={} 陈旧丢弃={} 乱序完成={}",
      submitted / report_dt_sec, results / report_dt_sec, avg_total_ms, avg_preprocess_ms,
      avg_infer_ms, avg_postprocess_ms, busy_drop, empty_results, stale_delta, out_of_order_delta);
    if (include_predictor) {
      tools::logger()->info(
        "[能量机关][拟合输入] 检测交接帧数={} 交接帧率={:.2f}fps", fit_input_frames,
        fit_input_frames / report_dt_sec);
      log_predictor_bucket(auto_buff::PredictorBackend::BIG_CERES, big_ceres);
      log_predictor_bucket(auto_buff::PredictorBackend::SMALL_CERES, small_ceres);
      log_predictor_bucket(auto_buff::PredictorBackend::SMALL_TARGET_EKF, small_ekf);
    }

    submitted = 0;
    results = 0;
    busy_drop = 0;
    empty_results = 0;
    fit_input_frames = 0;
    total_detect_ms = 0.0;
    preprocess_ms = 0.0;
    infer_ms = 0.0;
    postprocess_ms = 0.0;
    big_ceres = {};
    small_ceres = {};
    small_ekf = {};
  }
};

double wrap_angle(double value) { return std::atan2(std::sin(value), std::cos(value)); }

double zero_angle(double angle, int leaf_id)
{
  if (leaf_id < 0) return angle;
  return angle - static_cast<double>(leaf_id) * kLeafAngleStep;
}

std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::chrono::steady_clock::time_point replay_timestamp(
  const std::chrono::steady_clock::time_point & origin, double seconds)
{
  return origin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(seconds));
}

SourceKind parse_source(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "camera") return SourceKind::Camera;
  if (normalized == "video") return SourceKind::Video;
  throw std::runtime_error(fmt::format("Unsupported source: {}", value));
}

PipelineKind parse_pipeline(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "detect") return PipelineKind::Detect;
  if (normalized == "full") return PipelineKind::Full;
  throw std::runtime_error(fmt::format("Unsupported pipeline: {}", value));
}

ViewKind parse_view(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "none") return ViewKind::None;
  if (normalized == "detect") return ViewKind::Detect;
  if (normalized == "refine") return ViewKind::Refine;
  if (normalized == "result") return ViewKind::Result;
  throw std::runtime_error(fmt::format("Unsupported view: {}", value));
}

BuffModeArg parse_buff_mode_arg(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "auto") return BuffModeArg::Auto;
  if (normalized == "small") return BuffModeArg::Small;
  if (normalized == "big") return BuffModeArg::Big;
  throw std::runtime_error(fmt::format("Unsupported buff mode: {}", value));
}

bool has_named_option(int argc, char * argv[], const std::string & option)
{
  const std::string prefix = option + "=";
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == option || argument.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

bool parse_record_switch(const std::string & value)
{
  const auto normalized = lower(value);
  if (normalized == "on" || normalized == "true" || normalized == "1") return true;
  if (normalized == "off" || normalized == "false" || normalized == "0") return false;
  throw std::runtime_error(fmt::format("Unsupported record switch: {}", value));
}

std::string mode_to_string(io::Mode mode)
{
  const auto index = static_cast<std::size_t>(mode);
  if (index < io::MODES.size()) return io::MODES[index];
  return "unknown";
}

auto_buff::BuffMode detect_mode_from_string(const std::string & vision_mode)
{
  const auto normalized = lower(vision_mode);
  if (normalized.find("small") != std::string::npos) return auto_buff::BuffMode::SMALL;
  if (normalized.find("big") != std::string::npos) return auto_buff::BuffMode::BIG;
  return auto_buff::BuffMode::LOST;
}

auto_buff::BuffMode resolve_buff_mode(BuffModeArg cli_mode, const std::string & vision_mode)
{
  if (cli_mode == BuffModeArg::Small) return auto_buff::BuffMode::SMALL;
  if (cli_mode == BuffModeArg::Big) return auto_buff::BuffMode::BIG;
  return detect_mode_from_string(vision_mode);
}

CliOptions parse_cli(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    std::exit(0);
  }

  CliOptions options;
  options.source = parse_source(cli.get<std::string>("source"));
  options.pipeline = parse_pipeline(cli.get<std::string>("pipeline"));
  options.view = parse_view(cli.get<std::string>("view"));
  const auto parsed_r_center_view =
    auto_buff::parse_r_center_debug_view(cli.get<std::string>("r-center-view"));
  if (!parsed_r_center_view.has_value()) {
    throw std::runtime_error(
      fmt::format("Unsupported R-center view: {}", cli.get<std::string>("r-center-view")));
  }
  options.r_center_view = *parsed_r_center_view;
  options.config_path = cli.get<std::string>("config-path");
  options.input_prefix = cli.get<std::string>("input-prefix");
  options.record_dir = cli.get<std::string>("record-dir");
  options.start_index = std::max(0, cli.get<int>("start-index"));
  options.end_index = std::max(0, cli.get<int>("end-index"));
  options.max_frames = std::max(0, cli.get<int>("max-frames"));
  options.buff_mode = parse_buff_mode_arg(cli.get<std::string>("buff-mode"));
  options.record_enabled = parse_record_switch(cli.get<std::string>("record"));
  options.self_test = parse_record_switch(cli.get<std::string>("self-test"));

  const auto positional_input = cli.get<std::string>(0);
  if (options.input_prefix.empty()) options.input_prefix = positional_input;

  if (!options.self_test && options.source == SourceKind::Video) {
    if (options.input_prefix.empty() && options.record_dir.empty()) {
      throw std::runtime_error(
        "Video mode requires --input-prefix=<legacy_prefix> or --record-dir=<session_dir>.");
    }
  }

  if (options.pipeline == PipelineKind::Detect && options.view == ViewKind::Result) {
    tools::logger()->warn(
      "Detect pipeline does not provide result overlays. Falling back to detect view.");
    options.view = ViewKind::Detect;
  }

  if (options.view == ViewKind::Refine) {
    if (!has_named_option(argc, argv, "--r-center-view")) {
      options.r_center_view = auto_buff::RCenterDebugView::All;
    }
    options.view = ViewKind::Detect;
  }

  return options;
}

void require_self_test(bool condition, const std::string & message)
{
  if (!condition) throw std::runtime_error(message);
}

YAML::Node require_yaml_mapping(
  const YAML::Node & parent, const std::string & key, const std::string & path)
{
  const YAML::Node node = parent[key];
  require_self_test(node && node.IsMap(), path + " must be a mapping.");
  return node;
}

template <typename T>
void require_yaml_scalar(const YAML::Node & parent, const std::string & key, const std::string & path)
{
  const YAML::Node node = parent[key];
  require_self_test(node && node.IsScalar(), path + " must be a scalar.");
  try {
    static_cast<void>(node.as<T>());
  } catch (const YAML::Exception &) {
    require_self_test(false, path + " has an invalid type.");
  }
}

void run_buff_yaml_schema_self_test()
{
  const auto source_root = std::filesystem::path(__FILE__).parent_path().parent_path();
  auto config_root = source_root / "configs";
  if (!std::filesystem::exists(config_root)) {
    for (auto candidate = std::filesystem::current_path(); !candidate.empty();
         candidate = candidate.parent_path()) {
      if (std::filesystem::exists(candidate / "configs")) {
        config_root = candidate / "configs";
        break;
      }
      if (candidate == candidate.root_path()) break;
    }
  }
  const std::vector<std::string> config_names = {
    "ascento.yaml", "calibration.yaml", "camera.yaml", "demo.yaml", "example.yaml",
    "mvs.yaml", "sentry.yaml", "standard3.yaml",
    "standard4.yaml", "uav.yaml"};
  const std::vector<std::string> detector_double_keys = {
    "conf_threshold", "nms_threshold", "kpt_conf_threshold", "r_center_refine_radius_scale",
    "r_center_refine_max_child_area_ratio"};
  const std::vector<std::string> detector_int_keys = {
    "buff_center_roi_size", "r_center_refine_threshold", "r_center_refine_kernel_size"};
  const std::vector<std::string> predictor_double_keys = {
    "min_time_change", "small_ekf_p0_angle", "small_ekf_p0_speed", "small_ekf_q_accel",
    "small_ekf_r_angle"};
  const std::vector<std::string> predictor_int_keys = {"big_queue_size", "small_queue_size"};
  const std::vector<std::string> id_tracking_double_keys = {
    "max_match_angle", "min_residual_margin", "max_id_angular_speed",
    "max_zero_phase_innovation_rad", "coast_timeout_sec", "reset_timeout_sec", "model_max_nis"};
  const std::vector<std::string> fitting_double_keys = {
    "timeout_sec", "big_a_min", "big_a_max", "big_w_min", "big_w_max", "huber_delta",
    "omega_a", "omega_w", "big_cost_threshold", "small_cost_threshold", "small_huber_delta"};
  const std::vector<std::string> fitting_int_keys = {
    "big_min_fit_size", "small_min_fit_size", "big_cold_start_coarse_steps",
    "big_cold_start_refine_steps", "max_iterations", "ceres_threads"};
  const std::vector<std::string> selection_double_keys = {"w_yaw", "w_pitch", "switch_margin"};
  const std::vector<std::string> ballistic_double_keys = {
    "small_bias_time_ms", "big_bias_time_ms", "default_bullet_speed", "energy_yaw_bias",
    "energy_pitch_bias", "target_radius", "min_bullet_speed", "max_bullet_speed",
    "hero_air_resistance_k", "hero_s_bias", "hero_imu_pitch", "ballistic_debug_interval"};
  const std::vector<std::string> pose_filter_double_keys = {
    "max_dt_sec", "p0_r_yaw", "p0_r_v_yaw", "p0_r_pitch", "p0_r_dist", "p0_buff_yaw",
    "q_r_yaw_accel", "q_r_pitch", "q_r_dist", "q_buff_yaw", "r_r_yaw", "r_r_pitch",
    "r_r_dist", "r_buff_yaw", "r_blade_yaw", "r_blade_pitch", "r_blade_dist"};

  for (const auto & config_name : config_names) {
    const auto config_path = config_root / config_name;
    require_self_test(std::filesystem::exists(config_path), "Missing Buff config: " + config_name);
    const YAML::Node yaml = YAML::LoadFile(config_path.string());
    const auto & detector = require_yaml_mapping(yaml, "buff_detector", config_name + ".buff_detector");
    const auto & openvino = require_yaml_mapping(
      detector, "openvino", config_name + ".buff_detector.openvino");
    require_yaml_scalar<std::string>(detector, "model", config_name + ".buff_detector.model");
    require_yaml_scalar<std::string>(openvino, "device", config_name + ".buff_detector.openvino.device");
    require_yaml_scalar<int>(openvino, "num_requests", config_name + ".buff_detector.openvino.num_requests");
    require_yaml_scalar<int>(
      openvino, "result_queue_size", config_name + ".buff_detector.openvino.result_queue_size");
    require_self_test(
      openvino["num_requests"].as<int>() > 0,
      config_name + ".buff_detector.openvino.num_requests must be positive.");
    require_self_test(
      openvino["result_queue_size"].as<int>() > 0,
      config_name + ".buff_detector.openvino.result_queue_size must be positive.");
    require_yaml_scalar<bool>(detector, "buff_center_roi_enable", config_name + ".buff_detector.buff_center_roi_enable");
    require_yaml_scalar<bool>(detector, "r_center_refine", config_name + ".buff_detector.r_center_refine");
    require_yaml_scalar<bool>(
      detector, "r_center_refine_hierarchy_filter",
      config_name + ".buff_detector.r_center_refine_hierarchy_filter");
    for (const auto & key : detector_double_keys) {
      require_yaml_scalar<double>(detector, key, config_name + ".buff_detector." + key);
    }
    for (const auto & key : detector_int_keys) {
      require_yaml_scalar<int>(detector, key, config_name + ".buff_detector." + key);
    }
    require_self_test(
      detector["buff_center_roi_size"].as<int>() > 0,
      config_name + ".buff_detector.buff_center_roi_size must be positive.");
    require_self_test(
      detector["conf_threshold"].as<double>() >= 0.0 && detector["conf_threshold"].as<double>() <= 1.0 &&
        detector["nms_threshold"].as<double>() >= 0.0 && detector["nms_threshold"].as<double>() <= 1.0 &&
        detector["kpt_conf_threshold"].as<double>() >= 0.0 &&
        detector["kpt_conf_threshold"].as<double>() <= 1.0,
      config_name + ".buff_detector confidence thresholds must be in [0, 1].");

    const auto & pnp = require_yaml_mapping(yaml, "pnp", config_name + ".pnp");
    require_yaml_scalar<double>(pnp, "max_reprojection_error", config_name + ".pnp.max_reprojection_error");
    require_yaml_scalar<double>(yaml, "fire_gap_time", config_name + ".fire_gap_time");
    const auto & predictor = require_yaml_mapping(yaml, "predictor", config_name + ".predictor");
    require_yaml_scalar<std::string>(predictor, "mode", config_name + ".predictor.mode");
    require_yaml_scalar<std::string>(predictor, "small_predictor", config_name + ".predictor.small_predictor");
    for (const auto & key : predictor_double_keys) {
      require_yaml_scalar<double>(predictor, key, config_name + ".predictor." + key);
    }
    for (const auto & key : predictor_int_keys) {
      require_yaml_scalar<int>(predictor, key, config_name + ".predictor." + key);
    }

    const auto & id_tracking = require_yaml_mapping(yaml, "id_tracking", config_name + ".id_tracking");
    for (const auto & key : id_tracking_double_keys) {
      require_yaml_scalar<double>(id_tracking, key, config_name + ".id_tracking." + key);
    }
    require_yaml_scalar<int>(id_tracking, "model_min_confirmed_frames", config_name + ".id_tracking.model_min_confirmed_frames");

    const auto & fitting = require_yaml_mapping(yaml, "fitting", config_name + ".fitting");
    for (const auto & key : fitting_double_keys) {
      require_yaml_scalar<double>(fitting, key, config_name + ".fitting." + key);
    }
    for (const auto & key : fitting_int_keys) {
      require_yaml_scalar<int>(fitting, key, config_name + ".fitting." + key);
    }

    const auto & selection = require_yaml_mapping(yaml, "target_selection", config_name + ".target_selection");
    for (const auto & key : selection_double_keys) {
      require_yaml_scalar<double>(selection, key, config_name + ".target_selection." + key);
    }
    require_yaml_scalar<int>(selection, "switch_frames", config_name + ".target_selection.switch_frames");

    const auto & guard = require_yaml_mapping(yaml, "command_guard", config_name + ".command_guard");
    require_yaml_scalar<double>(guard, "hold_last_leaf_sec", config_name + ".command_guard.hold_last_leaf_sec");
    require_yaml_scalar<int>(guard, "fire_rearm_confirmed_frames", config_name + ".command_guard.fire_rearm_confirmed_frames");
    require_yaml_scalar<double>(guard, "max_yaw_rate", config_name + ".command_guard.max_yaw_rate");
    require_yaml_scalar<double>(guard, "max_pitch_rate", config_name + ".command_guard.max_pitch_rate");
    require_yaml_scalar<double>(guard, "max_slew_dt_sec", config_name + ".command_guard.max_slew_dt_sec");

    const auto & ballistic = require_yaml_mapping(yaml, "ballistic", config_name + ".ballistic");
    require_yaml_scalar<std::string>(ballistic, "model", config_name + ".ballistic.model");
    for (const auto & key : ballistic_double_keys) {
      require_yaml_scalar<double>(ballistic, key, config_name + ".ballistic." + key);
    }
    require_yaml_scalar<int>(ballistic, "fixed_iterations", config_name + ".ballistic.fixed_iterations");
    require_yaml_scalar<int>(ballistic, "hero_rk_iter", config_name + ".ballistic.hero_rk_iter");
    require_yaml_scalar<bool>(ballistic, "attack_leaf_center", config_name + ".ballistic.attack_leaf_center");
    require_yaml_scalar<bool>(ballistic, "ballistic_debug", config_name + ".ballistic.ballistic_debug");

    const auto & pose_filter = require_yaml_mapping(yaml, "pose_filter", config_name + ".pose_filter");
    require_yaml_scalar<bool>(pose_filter, "enabled", config_name + ".pose_filter.enabled");
    for (const auto & key : pose_filter_double_keys) {
      require_yaml_scalar<double>(pose_filter, key, config_name + ".pose_filter." + key);
    }
    require_self_test(!yaml["model"], config_name + " retains the legacy root Buff model key.");
  }
}

void run_small_rune_direction_self_test()
{
  auto_buff::AngleEstimateConfig config;
  config.small_predictor = auto_buff::SmallRunePredictor::TARGET_EKF;
  config.min_time_change = 1e-6;
  config.max_match_angle = 0.5;
  config.max_zero_phase_innovation_rad = 0.5;
  config.max_id_angular_speed = 10.0;
  config.id_coast_timeout_sec = 10.0;
  config.id_reset_timeout_sec = 10.0;

  auto_buff::AngleEstimate estimator(config);
  double time = 0.0;
  double angle = 0.0;
  auto do_update = [&](double a) {
    const auto match = estimator.update(a, time, auto_buff::BuffMode::SMALL);
    require_self_test(match.confirmed(), "Small-rune direction self-test lost leaf-ID continuity.");
  };

  // Frame 0 — init
  do_update(angle);

  // After 1 positive step the cumulative sum (~0.05 rad) is below the lock threshold.
  time += 0.01;
  angle += 0.05;
  do_update(angle);
  require_self_test(
    !estimator.direction_locked(),
    "Direction must be unlocked after only one positive observation.");
  require_self_test(
    estimator.debug_snapshot().direction == 0,
    "debug_snapshot().direction must be 0 while direction is unlocked.");

  // Second positive step: cumulative sum ~0.10, still below threshold.
  time += 0.01;
  angle += 0.05;
  do_update(angle);
  require_self_test(
    !estimator.direction_locked(),
    "Direction must remain unlocked when cumulative displacement is still below threshold.");

  // Third positive step: cumulative sum ~0.15 reaches the lock threshold.
  time += 0.01;
  angle += 0.05;
  do_update(angle);
  require_self_test(
    estimator.direction_locked(),
    "Direction must lock after cumulative positive displacement reaches threshold.");
  require_self_test(
    estimator.debug_snapshot().direction == 1,
    "Direction must lock to +1 after positive cumulative displacement.");

  // Contrary observations — in the new design the locked direction must NOT flip.
  for (int i = 0; i < 10; ++i) {
    time += 0.01;
    angle -= 0.05;
    do_update(angle);
    require_self_test(
      estimator.direction_locked(), "Locked direction must stay locked under contrary observations.");
    require_self_test(
      estimator.debug_snapshot().direction == 1,
      "Locked direction must not flip under contrary observations.");
  }
}

double shortest_angle_distance(double from, double to)
{
  return std::atan2(std::sin(to - from), std::cos(to - from));
}

void run_big_rune_cold_start_self_test()
{
  auto_buff::AngleEstimateConfig config;
  config.max_iterations = 50;
  config.big_cost_threshold = 0.01;

  auto_buff::EnergyTri truth;
  truth.mode = auto_buff::BuffMode::BIG;
  truth.direction = 1;
  truth.a = 1.035;
  truth.w = 1.994;
  truth.c = 0.83;
  truth.t0 = 2.0 * CV_PI / truth.w - 0.015;

  std::deque<auto_buff::LeafData> observations;
  for (int index = 0; index < 180; ++index) {
    const double time = static_cast<double>(index) * 0.02;
    double angle = truth.getAngle(time);
    if (index == 70) angle += 0.70;
    observations.push_back({angle, angle, time, 0});
  }

  auto_buff::EnergyTri initial;
  initial.mode = auto_buff::BuffMode::BIG;
  initial.direction = 1;
  auto_buff::AngleEstimate estimator(config);
  auto_buff::AngleEstimate::AngleFitResult result;
  const bool fitted = estimator.angleFitOnSnapshot(
    observations, initial, auto_buff::AngleEstimate::State::FITTING, false, result);

  require_self_test(fitted && result.success, "Big-rune cold start failed on a bounded synthetic fit.");
  require_self_test(
    std::abs(result.energy_tri.a - truth.a) < 0.025,
    fmt::format(
      "Big-rune cold start did not recover amplitude near the configured bound: expected {}, got {}, "
      "cost {}.",
      truth.a, result.energy_tri.a, result.average_cost));
  require_self_test(
    std::abs(result.energy_tri.w - truth.w) < 0.012,
    fmt::format(
      "Big-rune cold start did not recover angular frequency near the configured bound: expected {}, "
      "got {}, cost {}.",
      truth.w, result.energy_tri.w, result.average_cost));
  const double future_time = 4.5;
  require_self_test(
    std::abs(shortest_angle_distance(result.energy_tri.getAngle(future_time), truth.getAngle(future_time))) <
      0.03,
    "Big-rune cold start selected the wrong phase basin after an outlier.");
}

std::deque<auto_buff::LeafData> make_small_rune_observations(
  double speed, double intercept, int outlier_index = -1)
{
  std::deque<auto_buff::LeafData> observations;
  for (int index = 0; index < 120; ++index) {
    const double time = static_cast<double>(index) * 0.02;
    double angle = speed * time + intercept;
    if (index == outlier_index) angle += 0.70;
    observations.push_back({angle, angle, time, 0});
  }
  return observations;
}

void run_small_rune_ceres_self_test()
{
  auto_buff::EnergyTri initial;
  initial.mode = auto_buff::BuffMode::SMALL;
  initial.direction = 1;

  auto_buff::AngleEstimateConfig robust_config;
  robust_config.small_cost_threshold = 0.001;
  auto_buff::AngleEstimate robust_estimator(robust_config);
  auto_buff::AngleEstimate::AngleFitResult robust_result;
  constexpr double expected_speed = 0.92;
  constexpr double expected_intercept = 0.18;
  const bool robust_fitted = robust_estimator.angleFitOnSnapshot(
    make_small_rune_observations(expected_speed, expected_intercept, 51), initial,
    auto_buff::AngleEstimate::State::FITTING, false, robust_result);
  require_self_test(
    robust_fitted && robust_result.success,
    "Small-rune Ceres did not reject a single measurement outlier robustly.");
  require_self_test(
    std::abs(robust_result.energy_tri.k - expected_speed) < 0.01,
    "Small-rune Ceres did not recover speed after an outlier.");
  require_self_test(
    std::abs(robust_result.energy_tri.b - expected_intercept) < 0.01,
    "Small-rune Ceres did not recover intercept after an outlier.");

  auto_buff::AngleEstimateConfig one_iteration_config;
  one_iteration_config.max_iterations = 0;
  one_iteration_config.small_cost_threshold = 0.001;
  auto_buff::AngleEstimate one_iteration_estimator(one_iteration_config);
  auto_buff::AngleEstimate::AngleFitResult one_iteration_result;
  const bool one_iteration_fitted = one_iteration_estimator.angleFitOnSnapshot(
    make_small_rune_observations(initial.k, 0.02), initial,
    auto_buff::AngleEstimate::State::FITTING, false, one_iteration_result);
  require_self_test(
    !one_iteration_fitted && !one_iteration_result.success,
    "Small-rune Ceres accepted a numerically good but unconverged solution.");
}

void run_ceres_phase_handoff_self_test()
{
  const auto make_fit_result = [](const auto_buff::EnergyTri & model) {
    auto_buff::AngleEstimate::AngleFitResult result;
    result.success = true;
    result.mode = model.mode;
    result.energy_tri = model;
    result.latest_t = 0.0;
    return result;
  };

  {
    auto_buff::AngleEstimate estimator;
    estimator.init(0.0, 2.0, auto_buff::BuffMode::BIG);

    auto_buff::EnergyTri first_model;
    first_model.mode = auto_buff::BuffMode::BIG;
    first_model.direction = 1;
    first_model.a = 0.85;
    first_model.w = 1.90;
    first_model.c = 0.25;
    estimator.applyAngleFitResult(make_fit_result(first_model));
    require_self_test(
      estimator.update(0.0, 2.1, auto_buff::BuffMode::BIG).confirmed(),
      "Big-rune phase handoff self-test could not add the latest observation.");
    const double before = estimator.predict(2.1);

    auto_buff::EnergyTri updated_model = first_model;
    updated_model.a = 1.10;
    updated_model.w = 2.05;
    updated_model.c = 0.70;
    estimator.applyAngleFitResult(make_fit_result(updated_model));
    require_self_test(
      std::abs(shortest_angle_distance(before, estimator.predict(2.1))) < 1e-12,
      "Big-rune Ceres handoff changed the phase at the latest observation.");
  }

  {
    auto_buff::AngleEstimate estimator;
    estimator.init(0.0, 3.0, auto_buff::BuffMode::SMALL);

    auto_buff::EnergyTri first_model;
    first_model.mode = auto_buff::BuffMode::SMALL;
    first_model.direction = -1;
    first_model.k = 1.00;
    first_model.b = 0.30;
    estimator.applyAngleFitResult(make_fit_result(first_model));
    require_self_test(
      estimator.update(0.0, 3.1, auto_buff::BuffMode::SMALL).confirmed(),
      "Small-rune phase handoff self-test could not add the latest observation.");
    const double before = estimator.predict(3.1);

    auto_buff::EnergyTri updated_model = first_model;
    updated_model.k = 0.85;
    updated_model.b = 0.90;
    updated_model.direction = 1;
    estimator.applyAngleFitResult(make_fit_result(updated_model));
    require_self_test(
      std::abs(shortest_angle_distance(before, estimator.predict(3.1))) < 1e-12,
      "Small-rune reverse-direction Ceres handoff changed the phase at the latest observation.");
  }
}

bool read_legacy_samples(const std::string & input_prefix, std::vector<ReplaySample> & samples)
{
  const auto text_path = fmt::format("{}.txt", input_prefix);
  std::ifstream text(text_path);
  if (!text.is_open()) {
    tools::logger()->error("Failed to open legacy IMU file: {}", text_path);
    return false;
  }

  samples.clear();
  double t = 0.0;
  double w = 0.0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  int frame_index = 0;
  while (text >> t >> w >> x >> y >> z) {
    ReplaySample sample;
    sample.frame_index = frame_index++;
    sample.t_capture_sec = t;
    sample.q = Eigen::Quaterniond(w, x, y, z);
    sample.bullet_speed = kDefaultBulletSpeed;
    sample.vision_mode = "unknown";
    samples.push_back(sample);
  }

  return !samples.empty();
}

bool read_session_metadata(const std::string & record_dir, SessionMetadata & metadata)
{
  const auto metadata_path = fmt::format("{}/metadata.yaml", record_dir);
  if (!std::ifstream(metadata_path).good()) {
    tools::logger()->warn("Session metadata not found: {}", metadata_path);
    metadata = {};
    return false;
  }

  const YAML::Node yaml = YAML::LoadFile(metadata_path);
  metadata.config_path = yaml["config_path"] ? yaml["config_path"].as<std::string>() : "";
  metadata.fps = yaml["fps"] ? yaml["fps"].as<double>() : 0.0;
  metadata.record_version = yaml["record_version"] ? yaml["record_version"].as<int>() : 0;
  return true;
}

bool read_session_samples(const std::string & record_dir, std::vector<ReplaySample> & samples)
{
  const auto csv_path = fmt::format("{}/imu.csv", record_dir);
  std::ifstream csv(csv_path);
  if (!csv.is_open()) {
    tools::logger()->error("Failed to open session IMU file: {}", csv_path);
    return false;
  }

  samples.clear();
  std::string line;
  bool header_skipped = false;
  while (std::getline(csv, line)) {
    if (line.empty()) continue;
    if (!header_skipped) {
      header_skipped = true;
      if (line.find("frame_index") != std::string::npos) continue;
    }

    std::vector<std::string> fields;
    std::string field;
    std::stringstream row(line);
    while (std::getline(row, field, ',')) fields.push_back(field);
    if (fields.size() < 8) {
      tools::logger()->warn("Skip malformed IMU row: {}", line);
      continue;
    }

    ReplaySample sample;
    sample.frame_index = std::stoi(fields[0]);
    sample.t_capture_sec = std::stod(fields[1]);
    sample.q = Eigen::Quaterniond(
      std::stod(fields[2]), std::stod(fields[3]), std::stod(fields[4]), std::stod(fields[5]));
    sample.bullet_speed = std::stod(fields[6]);
    sample.vision_mode = fields[7];
    samples.push_back(sample);
  }

  std::sort(samples.begin(), samples.end(), [](const ReplaySample & lhs, const ReplaySample & rhs) {
    return lhs.frame_index < rhs.frame_index;
  });
  return !samples.empty();
}

bool open_video_replay(
  const CliOptions & options, VideoReplaySession & session, std::string & config_path)
{
  std::string video_path;
  if (!options.record_dir.empty()) {
    video_path = fmt::format("{}/video.avi", options.record_dir);
    read_session_metadata(options.record_dir, session.metadata);
    if (!read_session_samples(options.record_dir, session.samples)) return false;
    if (config_path.empty() && !session.metadata.config_path.empty()) {
      config_path = session.metadata.config_path;
    }
    if (config_path.empty()) {
      tools::logger()->error(
        "Replay from --record-dir requires --config-path when metadata.yaml is missing or "
        "does not provide config_path.");
      return false;
    }
  } else {
    video_path = fmt::format("{}.avi", options.input_prefix);
    if (!read_legacy_samples(options.input_prefix, session.samples)) return false;
  }

  session.video = cv::VideoCapture(video_path);
  if (!session.video.isOpened()) {
    tools::logger()->error("Failed to open replay video: {}", video_path);
    return false;
  }

  session.fps = session.video.get(cv::CAP_PROP_FPS);
  if (session.fps <= 0.0) {
    session.fps = session.metadata.fps > 0.0 ? session.metadata.fps : kFallbackVideoFps;
  }
  session.total_frames = static_cast<int>(session.video.get(cv::CAP_PROP_FRAME_COUNT));
  return true;
}

cv::Scalar blade_color(
  const auto_buff::PowerRune & rune, const auto_buff::FanBlade & blade, std::size_t index,
  const auto_buff::BuffBallisticResult * ballistic)
{
  const int observed_leaf_id = ballistic != nullptr && ballistic->observed_leaf_id >= 0
                                 ? ballistic->observed_leaf_id
                                 : rune.target_leaf_id;
  const int attack_leaf_id = ballistic != nullptr ? ballistic->attack_leaf_id : -1;
  const bool is_attack_leaf = attack_leaf_id >= 0 && blade.leaf_id == attack_leaf_id;
  const bool has_observed_leaf = observed_leaf_id >= 0 && blade.leaf_id >= 0;
  const bool is_observed_leaf = has_observed_leaf
                                  ? blade.leaf_id == observed_leaf_id
                                  : (index == 0 || blade.type == auto_buff::_target);
  if (is_observed_leaf && is_attack_leaf) return {255, 255, 0};
  if (is_attack_leaf) return {0, 255, 255};
  if (is_observed_leaf) return {0, 255, 0};
  if (blade.type == auto_buff::_unlight) return {96, 96, 96};
  return {255, 180, 0};
}

void draw_blade_annotations(
  cv::Mat & image, const auto_buff::PowerRune & rune,
  const auto_buff::BuffBallisticResult * ballistic = nullptr)
{
  for (std::size_t i = 0; i < rune.fanblades.size(); ++i) {
    const auto & blade = rune.fanblades[i];
    if (blade.points.empty()) continue;
    const auto color = blade_color(rune, blade, i, ballistic);
    tools::draw_points(image, blade.points, color, blade.type == auto_buff::_unlight ? 1 : 2);
    tools::draw_point(image, blade.center, color, 4);

    std::string label = blade.type == auto_buff::_unlight ? "dark" : "light";
    // 观测片用于PnP/预测更新，击打片才是最终弹道目标，颜色和标签必须分开。
    const int observed_leaf_id = ballistic != nullptr && ballistic->observed_leaf_id >= 0
                                   ? ballistic->observed_leaf_id
                                   : rune.target_leaf_id;
    const int attack_leaf_id = ballistic != nullptr ? ballistic->attack_leaf_id : -1;
    const bool is_observed = (observed_leaf_id >= 0 && blade.leaf_id == observed_leaf_id) ||
                             (observed_leaf_id < 0 && blade.type == auto_buff::_target);
    const bool is_attack = attack_leaf_id >= 0 && blade.leaf_id == attack_leaf_id;
    if (is_observed && is_attack) {
      label = "observed/attack";
    } else if (is_attack) {
      label = "attack";
    } else if (is_observed) {
      label = "observed";
    }
    if (blade.leaf_angle_valid) {
      label += fmt::format(" id:{} angle:{:.1f}", blade.leaf_id, blade.leaf_angle * 57.3);
    }
    label += fmt::format(" conf:{:.2f}", blade.confidence);
    tools::draw_text(
      image, label,
      cv::Point(static_cast<int>(blade.center.x + 8.0f), static_cast<int>(blade.center.y - 8.0f)),
      color, 0.5, 1);
  }
}

void draw_detection_overlay(
  cv::Mat & image, const std::optional<auto_buff::PowerRune> & rune, int frame_index,
  const auto_buff::RCenterRefineDebug & refine_debug)
{
  if (image.empty()) return;

  tools::draw_text(image, fmt::format("frame {}", frame_index), {10, 28}, {255, 255, 255}, 0.7, 2);
  tools::draw_text(
    image,
    fmt::format(
      "R refine: {:.3f} ms ({})", refine_debug.refine_dt_ms,
      refine_debug.refined_valid ? "valid" : "fallback"),
    {10, 56}, refine_debug.refined_valid ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 165, 255), 0.6,
    2);
  if (refine_debug.coarse_r_center != cv::Point2f{} || refine_debug.refined_valid) {
    tools::draw_point(image, refine_debug.coarse_r_center, {0, 255, 255}, 4);
    tools::draw_text(
      image, "R coarse",
      cv::Point(
        static_cast<int>(refine_debug.coarse_r_center.x + 8.0f),
        static_cast<int>(refine_debug.coarse_r_center.y - 8.0f)),
      {0, 255, 255}, 0.5, 1);
    const cv::Scalar refined_color =
      refine_debug.refined_valid ? cv::Scalar(255, 255, 0) : cv::Scalar(160, 160, 160);
    tools::draw_point(image, refine_debug.refined_r_center, refined_color, 5);
    tools::draw_text(
      image, refine_debug.refined_valid ? "R refined" : "R fallback",
      cv::Point(
        static_cast<int>(refine_debug.refined_r_center.x + 8.0f),
        static_cast<int>(refine_debug.refined_r_center.y + 14.0f)),
      refined_color, 0.5, 1);
  }

  if (!rune) return;
  draw_blade_annotations(image, *rune);
}

std::vector<cv::Point2f> draw_reprojection(
  cv::Mat & image, auto_buff::Solver & solver, const auto_buff::PowerRune & rune,
  const Eigen::Matrix3d & rotation_world,
  const cv::Scalar & color)
{
  auto image_points = solver.reproject_buff(rune.xyz_in_world, rotation_world);
  if (image_points.size() < 5) return image_points;
  tools::draw_points(
    image, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), color, 2);
  tools::draw_points(
    image, std::vector<cv::Point2f>(image_points.begin() + 4, image_points.end()), color, 2);
  return image_points;
}

void draw_result_overlay(
  cv::Mat & image, const auto_buff::PowerRune & rune, auto_buff::Solver & solver,
  auto_buff::Aimer & aimer)
{
  const auto & ballistic = aimer.last_ballistic_result();
  draw_blade_annotations(image, rune, &ballistic);
  const int observed_leaf_id =
    ballistic.observed_leaf_id >= 0 ? ballistic.observed_leaf_id : rune.target_leaf_id;
  const int attack_leaf_id = ballistic.attack_leaf_id;
  tools::draw_point(image, rune.r_center, {0, 0, 255}, 4);
  tools::draw_text(
    image, "R",
    cv::Point(static_cast<int>(rune.r_center.x + 8.0f), static_cast<int>(rune.r_center.y - 8.0f)),
    {0, 0, 255}, 0.6, 2);
  tools::draw_text(
    image, fmt::format("observed leaf: {}  attack leaf: {}", observed_leaf_id, attack_leaf_id),
    {10, 28}, {255, 255, 255}, 0.7, 2);

  draw_reprojection(image, solver, rune, rune.rotation_world, {0, 255, 0});

  if (
    !ballistic.valid || ballistic.predict_time_abs <= 0.0 ||
    !std::isfinite(ballistic.predicted_angle)) {
    return;
  }
  const double attack_angle = ballistic.predicted_angle;
  const int predicted_leaf_id =
    ballistic.attack_leaf_id >= 0 ? ballistic.attack_leaf_id : observed_leaf_id;
  tools::draw_text(
    image,
    fmt::format(
      "predict attack leaf: {}  dt:{:.1f}ms", predicted_leaf_id,
      std::max(0.0, (ballistic.predict_time_abs - ballistic.observed_time_abs) * 1e3)),
    {10, 56}, {0, 0, 255}, 0.7, 2);
  Eigen::Vector3d predicted_ypr = rune.ypr_in_world;
  predicted_ypr[2] = attack_angle + CV_PI / 2.0;
  const auto predicted_rotation = tools::rotation_matrix(predicted_ypr);
  const auto image_points =
    draw_reprojection(image, solver, rune, predicted_rotation, {0, 0, 255});
  if (image_points.size() >= 5) {
    const auto & predicted_center = image_points.back();
    tools::draw_point(image, predicted_center, {0, 0, 255}, 5);
    tools::draw_text(
      image, fmt::format("pred attack id:{}", predicted_leaf_id),
      cv::Point(
        static_cast<int>(predicted_center.x + 8.0f), static_cast<int>(predicted_center.y + 18.0f)),
      {0, 0, 255}, 0.6, 2);
  }
}

void populate_plot_data(
  nlohmann::json & data, const auto_buff::PowerRune * rune, auto_buff::Solver & solver,
  auto_buff::Aimer & aimer, const io::Command & command,
  const auto_buff::RCenterRefineDebug & refine_debug)
{
  data["refine_dt_ms"] = refine_debug.refine_dt_ms;
  data["refine_valid"] = refine_debug.refined_valid ? 1 : 0;
  data["filtered_pose_valid"] = 0;
  data["used_filtered_pose"] = 0;

  if (rune != nullptr) {
    data["buff_R_yaw"] = rune->ypd_in_world[0];
    data["buff_R_pitch"] = rune->ypd_in_world[1];
    data["buff_R_dis"] = rune->ypd_in_world[2];
    data["buff_yaw"] = rune->ypr_in_world[0] * 57.3;
    data["buff_pitch"] = rune->ypr_in_world[1] * 57.3;
    data["buff_roll"] = rune->ypr_in_world[2] * 57.3;

    const auto & pose_snapshot = aimer.pose_snapshot();
    if (pose_snapshot.valid) {
      data["filtered_pose_valid"] = 1;
      data["filtered_buff_R_yaw"] = pose_snapshot.filtered_ypd_world[0] * 57.3;
      data["filtered_buff_R_pitch"] = pose_snapshot.filtered_ypd_world[1] * 57.3;
      data["filtered_buff_R_dist"] = pose_snapshot.filtered_ypd_world[2];
      data["filtered_buff_yaw"] = pose_snapshot.filtered_buff_yaw * 57.3;
    }

    const auto & ballistic = aimer.last_ballistic_result();
    data["used_filtered_pose"] = ballistic.used_filtered_pose ? 1 : 0;
    data["aim_source"] = static_cast<int>(ballistic.aim_source);
    if (
      ballistic.valid && ballistic.predict_time_abs > 0.0 &&
      std::isfinite(ballistic.predicted_angle)) {
      const double predicted_angle = ballistic.predicted_angle;
      const auto snapshot = aimer.prediction_snapshot();
      data["angle"] = predicted_angle * 57.3;
      data["observed_angle"] = rune->physical_angle * 57.3;
      data["delta_angle"] = (predicted_angle - rune->physical_angle) * 57.3;
      data["spd"] = aimer.prediction_speed(ballistic.predict_time_abs) * 57.3;
      data["predict_samples"] = snapshot.sample_count;
      data["predict_cost"] = snapshot.average_cost;
      data["predict_backend"] =
        snapshot.small_predictor == auto_buff::SmallRunePredictor::TARGET_EKF ? "TARGET_EKF"
                                                                              : "CERES";
      data["observed_leaf_id"] = ballistic.observed_leaf_id;
      data["attack_leaf_id"] = ballistic.attack_leaf_id;
      data["selection_cost"] = ballistic.selection_cost;
      data["a"] = snapshot.energy_tri.a;
      data["w"] = snapshot.energy_tri.w;
      data["c"] = snapshot.energy_tri.c;
      data["t0"] = snapshot.energy_tri.t0;
      if (snapshot.has_ceres_solve_time) {
        data["ceres_solve_ms"] = snapshot.last_ceres_solve_time_ms;
      }
      if (ballistic.observed_leaf_id >= 0 && ballistic.attack_leaf_id >= 0) {
        const double theta0_obs = zero_angle(rune->physical_angle, ballistic.observed_leaf_id);
        const double theta0_fit = zero_angle(predicted_angle, ballistic.attack_leaf_id);
        data["theta0_obs"] = theta0_obs * 57.3;
        data["theta0_fit"] = theta0_fit * 57.3;
        data["theta0_residual"] = wrap_angle(theta0_fit - theta0_obs) * 57.3;
      }
    }
  }

  const Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
  data["gimbal_yaw"] = ypr[0] * 57.3;
  data["gimbal_pitch"] = ypr[1] * 57.3;
  if (command.control) {
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["shoot"] = command.shoot ? 1 : 0;
  }
}

bool handle_visual_controls(ViewKind view, auto_buff::RCenterDebugView r_center_view)
{
  if (view == ViewKind::None && r_center_view == auto_buff::RCenterDebugView::Off) return false;

  int key = cv::waitKey(1);
  if (key == 'q' || key == 27) return true;
  while (key == ' ') {
    const int paused_key = cv::waitKey(30);
    if (paused_key == 'q' || paused_key == 27) return true;
    if (paused_key == ' ') break;
  }
  return false;
}

void show_full_pipeline_views(
  ViewKind view, const cv::Mat & detect_image, const cv::Mat & result_image)
{
  if (view == ViewKind::Detect && !detect_image.empty()) {
    cv::imshow("buff detect", detect_image);
  } else if (view == ViewKind::Result && !result_image.empty()) {
    cv::imshow("buff result", result_image);
  }
}

bool render_detect_only(
  ViewKind view, auto_buff::RCenterDebugView r_center_view, cv::Mat detect_image,
  const std::optional<auto_buff::PowerRune> & rune, int frame_index,
  const auto_buff::RCenterRefineDebug & refine_debug)
{
  if (view == ViewKind::None && r_center_view == auto_buff::RCenterDebugView::Off) return false;

  auto_buff::show_r_center_debug_views(r_center_view, detect_image, refine_debug);
  draw_detection_overlay(detect_image, rune, frame_index, refine_debug);
  if (view == ViewKind::Detect) cv::imshow("buff detect", detect_image);
  return handle_visual_controls(view, r_center_view);
}

bool process_full_pipeline(
  FullPipelineContext & context, BuffPerfLogState & perf_log_state, ViewKind view,
  auto_buff::RCenterDebugView r_center_view, cv::Mat detect_image, cv::Mat result_image,
  std::optional<auto_buff::PowerRune> & rune, const auto_buff::RCenterRefineDebug & refine_debug,
  int frame_index, const Eigen::Quaterniond & q, double observed_time_abs, double now_time_abs,
  double bullet_speed, auto_buff::BuffMode buff_mode, io::Command * issued_command = nullptr)
{
  context.solver.set_R_gimbal2world(q);
  context.solver.solve(rune);

  io::Command command{false, false, 0.0, 0.0};
  if (rune) {
    rune->last_observed_time = observed_time_abs;
    if (rune->pnp_valid) perf_log_state.record_fit_input();
    command = context.aimer.aim(*rune, observed_time_abs, now_time_abs, bullet_speed, buff_mode);
    if (view == ViewKind::Result && rune->pnp_valid) {
      draw_result_overlay(result_image, *rune, context.solver, context.aimer);
    }
  } else {
    context.aimer.notify_observation_missed();
  }
  if (issued_command != nullptr) *issued_command = command;

  nlohmann::json plot_data;
  populate_plot_data(
    plot_data, rune ? &rune.value() : nullptr, context.solver, context.aimer, command,
    refine_debug);
  context.plotter.plot(plot_data);

  auto_buff::show_r_center_debug_views(r_center_view, detect_image, refine_debug);
  draw_detection_overlay(detect_image, rune, frame_index, refine_debug);
  show_full_pipeline_views(view, detect_image, result_image);
  return handle_visual_controls(view, r_center_view);
}

bool lookup_sample(
  const std::unordered_map<int, ReplaySample> & sample_map, int frame_index, ReplaySample & sample)
{
  const auto it = sample_map.find(frame_index);
  if (it == sample_map.end()) return false;
  sample = it->second;
  return true;
}
}  // namespace

int main(int argc, char * argv[])
{
  try {
    CliOptions options = parse_cli(argc, argv);
    if (options.self_test) {
      run_buff_yaml_schema_self_test();
      run_small_rune_direction_self_test();
      run_big_rune_cold_start_self_test();
      run_small_rune_ceres_self_test();
      run_ceres_phase_handoff_self_test();
      fmt::print("auto_buff self-test passed\n");
      return 0;
    }
    std::string config_path = options.config_path;

    if (options.source == SourceKind::Video) {
      VideoReplaySession replay;
      if (!open_video_replay(options, replay, config_path)) return 1;
      if (config_path.empty()) config_path = "configs/sentry.yaml";

      auto_buff::Buff_Detector detector(config_path);
      BuffPerfLogState perf_log_state;
      auto perf_report_stamp = std::chrono::steady_clock::now();
      const bool enable_predictor_log = options.pipeline == PipelineKind::Full;
      std::optional<FullPipelineContext> full_context;
      tools::Plotter plotter;
      std::optional<auto_buff::Solver> solver;
      std::optional<auto_buff::Aimer> aimer;
      if (options.pipeline == PipelineKind::Full) {
        solver.emplace(config_path);
        aimer.emplace(config_path);
        full_context.emplace(FullPipelineContext{plotter, *solver, *aimer});
      }

      std::unordered_map<int, ReplaySample> sample_map;
      sample_map.reserve(replay.samples.size());
      for (const auto & sample : replay.samples) sample_map.emplace(sample.frame_index, sample);

      const int available_frames =
        replay.total_frames > 0
          ? std::min(replay.total_frames, static_cast<int>(replay.samples.size()))
          : static_cast<int>(replay.samples.size());
      if (options.start_index >= available_frames) {
        tools::logger()->error(
          "Start index {} exceeds available frames {}", options.start_index, available_frames);
        return 1;
      }

      replay.video.set(cv::CAP_PROP_POS_FRAMES, options.start_index);
      const auto replay_origin = std::chrono::steady_clock::now();
      int submitted_frames = 0;
      int dropped_frames = 0;
      bool should_quit = false;

      for (int playback_index = options.start_index;
           playback_index < available_frames && !should_quit; ++playback_index) {
        if (options.end_index > 0 && playback_index > options.end_index) break;
        if (options.max_frames > 0 && submitted_frames >= options.max_frames) break;

        cv::Mat frame;
        replay.video.read(frame);
        if (frame.empty()) break;

        const ReplaySample & sample = replay.samples[playback_index];

        std::optional<auto_buff::PowerRune> rune;
        cv::Mat result_image;
        int result_frame_index = -1;
        double detect_dt_ms = 0.0;
        auto_buff::BuffDetectPerfStats detect_perf_stats;
        auto_buff::RCenterRefineDebug refine_debug;
        while (detector.fetch(
          rune, result_image, result_frame_index, detect_dt_ms, true, &refine_debug, nullptr,
          &detect_perf_stats)) {
          perf_log_state.record_detect(detect_dt_ms, detect_perf_stats, !rune.has_value());
          ReplaySample result_sample;
          if (!lookup_sample(sample_map, result_frame_index, result_sample)) continue;

          cv::Mat detect_image = result_image.clone();
          if (options.pipeline == PipelineKind::Detect) {
            should_quit = render_detect_only(
              options.view, options.r_center_view, detect_image, rune, result_frame_index,
              refine_debug);
          } else if (full_context.has_value()) {
            const auto buff_mode = resolve_buff_mode(options.buff_mode, result_sample.vision_mode);
            should_quit = process_full_pipeline(
              *full_context, perf_log_state, options.view, options.r_center_view, detect_image,
              result_image, rune, refine_debug, result_frame_index, result_sample.q,
              result_sample.t_capture_sec, result_sample.t_capture_sec, result_sample.bullet_speed,
              buff_mode);
            if (rune && rune->pnp_valid) {
              perf_log_state.record_predictor(full_context->aimer.prediction_snapshot());
            }
          }
          if (should_quit) break;
        }
        if (should_quit) break;

        const bool submit_ok = detector.submit(
          frame, sample.frame_index, replay_timestamp(replay_origin, sample.t_capture_sec));
        perf_log_state.record_submit(submit_ok);
        if (submit_ok) {
          submitted_frames++;
        } else {
          dropped_frames++;
        }

        const auto perf_now = std::chrono::steady_clock::now();
        const double report_dt = tools::delta_time(perf_now, perf_report_stamp);
        if (kEnablePerfLog && report_dt >= kPerfLogPeriodSec) {
          perf_log_state.log_and_reset(detector, report_dt, enable_predictor_log);
          perf_report_stamp = perf_now;
        }
      }

      const auto replay_drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!should_quit && std::chrono::steady_clock::now() < replay_drain_deadline) {
        bool fetched_any = false;
        std::optional<auto_buff::PowerRune> rune;
        cv::Mat result_image;
        int result_frame_index = -1;
        double detect_dt_ms = 0.0;
        auto_buff::BuffDetectPerfStats detect_perf_stats;
        auto_buff::RCenterRefineDebug refine_debug;
        while (detector.fetch(
          rune, result_image, result_frame_index, detect_dt_ms, true, &refine_debug, nullptr,
          &detect_perf_stats)) {
          fetched_any = true;
          perf_log_state.record_detect(detect_dt_ms, detect_perf_stats, !rune.has_value());
          ReplaySample result_sample;
          if (!lookup_sample(sample_map, result_frame_index, result_sample)) continue;

          cv::Mat detect_image = result_image.clone();
          if (options.pipeline == PipelineKind::Detect) {
            should_quit = render_detect_only(
              options.view, options.r_center_view, detect_image, rune, result_frame_index,
              refine_debug);
          } else if (full_context.has_value()) {
            const auto buff_mode = resolve_buff_mode(options.buff_mode, result_sample.vision_mode);
            should_quit = process_full_pipeline(
              *full_context, perf_log_state, options.view, options.r_center_view, detect_image,
              result_image, rune, refine_debug, result_frame_index, result_sample.q,
              result_sample.t_capture_sec, result_sample.t_capture_sec, result_sample.bullet_speed,
              buff_mode);
            if (rune && rune->pnp_valid) {
              perf_log_state.record_predictor(full_context->aimer.prediction_snapshot());
            }
          }
          if (should_quit) break;
        }
        if (!fetched_any) std::this_thread::sleep_for(std::chrono::milliseconds(5));

        const auto perf_now = std::chrono::steady_clock::now();
        const double report_dt = tools::delta_time(perf_now, perf_report_stamp);
        if (kEnablePerfLog && report_dt >= kPerfLogPeriodSec) {
          perf_log_state.log_and_reset(detector, report_dt, enable_predictor_log);
          perf_report_stamp = perf_now;
        }
      }

      const double final_report_dt =
        tools::delta_time(std::chrono::steady_clock::now(), perf_report_stamp);
      if (kEnablePerfLog && final_report_dt > 1e-6) {
        perf_log_state.log_and_reset(detector, final_report_dt, enable_predictor_log);
      }
      tools::logger()->info("Replay submitted: {}, dropped: {}", submitted_frames, dropped_frames);
      cv::destroyAllWindows();
      return 0;
    }

    if (config_path.empty()) config_path = "configs/sentry.yaml";
    tools::Plotter plotter;
    tools::Exiter exiter;
    io::Camera camera(config_path);
    io::CBoard cboard(config_path);
    auto_buff::Buff_Detector detector(config_path);
    BuffPerfLogState perf_log_state;
    auto perf_report_stamp = std::chrono::steady_clock::now();
    const bool enable_predictor_log = options.pipeline == PipelineKind::Full;

    std::optional<auto_buff::Solver> solver;
    std::optional<auto_buff::Aimer> aimer;
    std::optional<FullPipelineContext> full_context;
    if (options.pipeline == PipelineKind::Full) {
      solver.emplace(config_path);
      aimer.emplace(config_path);
      full_context.emplace(FullPipelineContext{plotter, *solver, *aimer});
    }

    std::unique_ptr<tools::Recorder> recorder;
    if (options.record_enabled) {
      tools::RecorderOptions recorder_options;
      recorder_options.fps = 30.0;
      recorder_options.output_mode = tools::RecorderOutputMode::session_dir;
      recorder_options.output_path = options.record_dir;
      recorder_options.config_path = config_path;
      recorder = std::make_unique<tools::Recorder>(recorder_options);
    }

    int capture_frame_index = 0;
    int submit_frame_index = 0;
    int submitted_frames = 0;
    int dropped_frames = 0;
    std::unordered_map<int, ReplaySample> live_samples;
    std::chrono::steady_clock::time_point time_origin;
    bool time_origin_ready = false;
    bool should_quit = false;

    while (!exiter.exit() && !should_quit) {
      if (options.max_frames > 0 && capture_frame_index >= options.max_frames) break;

      cv::Mat frame;
      std::chrono::steady_clock::time_point timestamp;
      camera.read(frame, timestamp);
      if (frame.empty()) continue;

      if (!time_origin_ready) {
        time_origin = timestamp;
        time_origin_ready = true;
      }

      const std::string vision_mode = mode_to_string(cboard.mode);
      const Eigen::Quaterniond submit_q = cboard.imu_at(timestamp);
      const bool submit_ok = detector.submit(frame, submit_frame_index, timestamp);
      perf_log_state.record_submit(submit_ok);
      if (submit_ok) {
        ReplaySample submit_sample;
        submit_sample.frame_index = submit_frame_index;
        submit_sample.t_capture_sec = tools::delta_time(timestamp, time_origin);
        submit_sample.q = submit_q;
        submit_sample.bullet_speed = cboard.bullet_speed;
        submit_sample.vision_mode = vision_mode;
        live_samples[submit_frame_index] = submit_sample;

        if (recorder != nullptr) {
          tools::RecordSampleMetadata metadata;
          metadata.frame_index = submit_frame_index;
          metadata.bullet_speed = submit_sample.bullet_speed;
          metadata.vision_mode = submit_sample.vision_mode;
          recorder->record(frame, submit_sample.q, timestamp, metadata);
        }
        submitted_frames++;
        submit_frame_index++;
      } else {
        dropped_frames++;
      }

      std::optional<auto_buff::PowerRune> rune;
      cv::Mat result_image;
      int result_frame_index = -1;
      double detect_dt_ms = 0.0;
      auto_buff::BuffDetectPerfStats detect_perf_stats;
      auto_buff::RCenterRefineDebug refine_debug;
      std::chrono::steady_clock::time_point result_timestamp;
      while (detector.fetch(
        rune, result_image, result_frame_index, detect_dt_ms, true, &refine_debug,
        &result_timestamp, &detect_perf_stats)) {
        perf_log_state.record_detect(detect_dt_ms, detect_perf_stats, !rune.has_value());
        cv::Mat detect_image = result_image.clone();
        if (options.pipeline == PipelineKind::Detect) {
          should_quit = render_detect_only(
            options.view, options.r_center_view, detect_image, rune, result_frame_index,
            refine_debug);
        } else if (full_context.has_value()) {
          ReplaySample live_sample;
          const bool have_live_sample =
            lookup_sample(live_samples, result_frame_index, live_sample);
          if (have_live_sample) live_samples.erase(result_frame_index);

          const auto q = have_live_sample ? live_sample.q : cboard.imu_at(result_timestamp);
          const double observed_time_abs = have_live_sample
                                             ? live_sample.t_capture_sec
                                             : tools::delta_time(result_timestamp, time_origin);
          const double now_time_abs =
            tools::delta_time(std::chrono::steady_clock::now(), time_origin);
          const double bullet_speed =
            have_live_sample ? live_sample.bullet_speed : cboard.bullet_speed;
          const std::string sample_vision_mode =
            have_live_sample ? live_sample.vision_mode : vision_mode;
          const auto buff_mode = resolve_buff_mode(options.buff_mode, sample_vision_mode);
          io::Command send_cmd{false, false, 0.0, 0.0};
          should_quit = process_full_pipeline(
            *full_context, perf_log_state, options.view, options.r_center_view, detect_image,
            result_image, rune, refine_debug, result_frame_index, q, observed_time_abs,
            now_time_abs, bullet_speed, buff_mode, &send_cmd);
          if (rune && rune->pnp_valid) {
            perf_log_state.record_predictor(full_context->aimer.prediction_snapshot());
          }
          cboard.send(send_cmd);
        }
        if (should_quit) break;
      }

      const auto perf_now = std::chrono::steady_clock::now();
      const double report_dt = tools::delta_time(perf_now, perf_report_stamp);
      if (kEnablePerfLog && report_dt >= kPerfLogPeriodSec) {
        perf_log_state.log_and_reset(detector, report_dt, enable_predictor_log);
        perf_report_stamp = perf_now;
      }

      if (options.end_index > 0 && capture_frame_index >= options.end_index) break;
      capture_frame_index++;
    }

    const auto camera_drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!should_quit && std::chrono::steady_clock::now() < camera_drain_deadline) {
      bool fetched_any = false;
      std::optional<auto_buff::PowerRune> rune;
      cv::Mat result_image;
      int result_frame_index = -1;
      double detect_dt_ms = 0.0;
      auto_buff::BuffDetectPerfStats detect_perf_stats;
      auto_buff::RCenterRefineDebug refine_debug;
      std::chrono::steady_clock::time_point result_timestamp;
      while (detector.fetch(
        rune, result_image, result_frame_index, detect_dt_ms, true, &refine_debug,
        &result_timestamp, &detect_perf_stats)) {
        fetched_any = true;
        perf_log_state.record_detect(detect_dt_ms, detect_perf_stats, !rune.has_value());
        cv::Mat detect_image = result_image.clone();
        if (options.pipeline == PipelineKind::Detect) {
          should_quit = render_detect_only(
            options.view, options.r_center_view, detect_image, rune, result_frame_index,
            refine_debug);
        } else if (full_context.has_value()) {
          ReplaySample live_sample;
          const bool have_live_sample =
            lookup_sample(live_samples, result_frame_index, live_sample);
          if (have_live_sample) live_samples.erase(result_frame_index);

          const auto q = have_live_sample ? live_sample.q : cboard.imu_at(result_timestamp);
          const double observed_time_abs = have_live_sample
                                             ? live_sample.t_capture_sec
                                             : tools::delta_time(result_timestamp, time_origin);
          const double now_time_abs =
            tools::delta_time(std::chrono::steady_clock::now(), time_origin);
          const double bullet_speed =
            have_live_sample ? live_sample.bullet_speed : cboard.bullet_speed;
          const std::string sample_vision_mode =
            have_live_sample ? live_sample.vision_mode : mode_to_string(cboard.mode);
          const auto buff_mode = resolve_buff_mode(options.buff_mode, sample_vision_mode);
          io::Command send_cmd{false, false, 0.0, 0.0};
          should_quit = process_full_pipeline(
            *full_context, perf_log_state, options.view, options.r_center_view, detect_image,
            result_image, rune, refine_debug, result_frame_index, q, observed_time_abs,
            now_time_abs, bullet_speed, buff_mode, &send_cmd);
          if (rune && rune->pnp_valid) {
            perf_log_state.record_predictor(full_context->aimer.prediction_snapshot());
          }
          cboard.send(send_cmd);
        }
        if (should_quit) break;
      }
      if (!fetched_any) std::this_thread::sleep_for(std::chrono::milliseconds(5));

      const auto perf_now = std::chrono::steady_clock::now();
      const double report_dt = tools::delta_time(perf_now, perf_report_stamp);
      if (kEnablePerfLog && report_dt >= kPerfLogPeriodSec) {
        perf_log_state.log_and_reset(detector, report_dt, enable_predictor_log);
        perf_report_stamp = perf_now;
      }
    }

    const double final_report_dt =
      tools::delta_time(std::chrono::steady_clock::now(), perf_report_stamp);
    if (kEnablePerfLog && final_report_dt > 1e-6) {
      perf_log_state.log_and_reset(detector, final_report_dt, enable_predictor_log);
    }
    tools::logger()->info("Camera submitted: {}, dropped: {}", submitted_frames, dropped_frames);
    cv::destroyAllWindows();
    return 0;
  } catch (const std::exception & e) {
    fmt::print(stderr, "auto_buff_test failed: {}\n", e.what());
    return 1;
  }
}
