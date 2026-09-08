#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "io/camera.hpp"
#include "io/cboard.hpp"
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

namespace
{
constexpr bool kShowDetectorView = true;
constexpr bool kShowResultView = true;
constexpr bool kEnablePerfLog = true;      // 是否输出性能日志（仅当前调试入口生效）
constexpr double kPerfLogPeriodSec = 1.0;  // 性能日志输出周期，单位：秒
constexpr double kLeafAngleStep = 2.0 * CV_PI / 5.0;

double wrap_angle(double value) { return std::atan2(std::sin(value), std::cos(value)); }

double zero_angle(double angle, int leaf_id)
{
  if (leaf_id < 0) return angle;
  return angle - static_cast<double>(leaf_id) * kLeafAngleStep;
}

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

  void log_and_reset(const auto_buff::Buff_Detector & detector, double report_dt_sec)
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
    tools::logger()->info(
      "[能量机关][拟合输入] 检测交接帧数={} 交接帧率={:.2f}fps", fit_input_frames,
      fit_input_frames / report_dt_sec);
    log_predictor_bucket(auto_buff::PredictorBackend::BIG_CERES, big_ceres);
    log_predictor_bucket(auto_buff::PredictorBackend::SMALL_CERES, small_ceres);
    log_predictor_bucket(auto_buff::PredictorBackend::SMALL_TARGET_EKF, small_ekf);

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
constexpr auto kDisplayPeriod = std::chrono::milliseconds(16);

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{r-center-view  | off | R-center view: off, roi, binary, contours or all }"
  "{@config-path   | | yaml配置文件路径 }";

cv::Scalar blade_color(const auto_buff::FanBlade & blade, int observed_leaf_id, int attack_leaf_id)
{
  const bool is_observed = observed_leaf_id >= 0 && blade.leaf_id == observed_leaf_id;
  const bool is_attack = attack_leaf_id >= 0 && blade.leaf_id == attack_leaf_id;
  if (is_observed && is_attack) return {255, 255, 0};
  if (is_attack) return {0, 255, 255};
  if (is_observed) return {0, 255, 0};
  if (blade.type == auto_buff::_unlight) return {96, 96, 96};
  return {255, 180, 0};
}

void draw_blade_annotations(
  cv::Mat & image, const auto_buff::PowerRune & rune, int observed_leaf_id, int attack_leaf_id)
{
  for (const auto & blade : rune.fanblades) {
    if (blade.points.empty()) continue;

    const bool is_observed = observed_leaf_id >= 0 && blade.leaf_id == observed_leaf_id;
    const bool is_attack = attack_leaf_id >= 0 && blade.leaf_id == attack_leaf_id;
    const auto color = blade_color(blade, observed_leaf_id, attack_leaf_id);
    tools::draw_points(image, blade.points, color, blade.type == auto_buff::_unlight ? 1 : 2);
    tools::draw_point(image, blade.center, color, 4);

    std::string label = blade.type == auto_buff::_unlight ? "dark" : "light";
    if (is_observed && is_attack) {
      label = "observed/attack";
    } else if (is_attack) {
      label = "attack";
    } else if (is_observed) {
      label = "observed";
    }
    if (blade.leaf_id >= 0) label += fmt::format(" id:{}", blade.leaf_id);
    if (blade.leaf_angle_valid) label += fmt::format(" angle:{:.1f}", blade.leaf_angle * 57.3);
    label += fmt::format(" conf:{:.2f}", blade.confidence);
    tools::draw_text(
      image, label,
      cv::Point(static_cast<int>(blade.center.x + 8.0f), static_cast<int>(blade.center.y - 8.0f)),
      color, 0.5, 1);
  }
}

std::vector<cv::Point2f> draw_reprojection(
  cv::Mat & image, auto_buff::Solver & solver, const Eigen::Vector3d & r_center_world,
  const Eigen::Matrix3d & rotation_world,
  const cv::Scalar & color)
{
  auto image_points = solver.reproject_buff(r_center_world, rotation_world);
  if (image_points.size() < 5) return image_points;
  tools::draw_points(
    image, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), color, 2);
  tools::draw_points(
    image, std::vector<cv::Point2f>(image_points.begin() + 4, image_points.end()), color, 2);
  return image_points;
}

void draw_buff_detection(
  cv::Mat & image, const std::optional<auto_buff::PowerRune> & rune, int frame_count,
  const auto_buff::RCenterRefineDebug & r_center_debug)
{
  if (image.empty()) return;

  tools::draw_text(image, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255}, 0.8, 2);
  if (r_center_debug.refined_valid || r_center_debug.coarse_r_center != cv::Point2f{}) {
    tools::draw_point(image, r_center_debug.coarse_r_center, {0, 255, 255}, 4);
    tools::draw_text(
      image, "R coarse", r_center_debug.coarse_r_center + cv::Point2f(8.0f, -8.0f), {0, 255, 255},
      0.5, 1);
    if (r_center_debug.refined_valid) {
      tools::draw_point(image, r_center_debug.refined_r_center, {0, 0, 255}, 5);
      tools::draw_text(
        image, "R refined", r_center_debug.refined_r_center + cv::Point2f(8.0f, 14.0f), {0, 0, 255},
        0.5, 1);
    }
  }

  if (!rune) return;

  for (std::size_t i = 0; i < rune->fanblades.size(); ++i) {
    const auto & blade = rune->fanblades[i];
    if (blade.type == auto_buff::_unlight || blade.points.size() < 4) continue;

    const bool is_target = i == 0 || blade.type == auto_buff::_target;
    const cv::Scalar color = is_target ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    tools::draw_points(image, blade.points, color, is_target ? 3 : 2);
    tools::draw_point(image, blade.points[0], {0, 0, 255}, is_target ? 5 : 4);
    for (int j = 1; j < 4; ++j) {
      tools::draw_point(image, blade.points[j], color, is_target ? 5 : 4);
    }

    std::string info =
      fmt::format("{} conf:{:.2f}", is_target ? "target" : "leaf", blade.confidence);
    if (blade.leaf_angle_valid) {
      info += fmt::format(" id:{} angle:{:.1f}", blade.leaf_id, blade.leaf_angle * 180.0 / CV_PI);
    }
    tools::draw_text(image, info, blade.center + cv::Point2f(8.0f, -8.0f), color, 0.5, 1);
  }
}

void draw_result_overlay(
  cv::Mat & image, const auto_buff::PowerRune & rune, auto_buff::Solver & solver,
  auto_buff::Aimer & aimer)
{
  const auto & result = aimer.last_ballistic_result();
  const int observed_leaf_id =
    result.observed_leaf_id >= 0 ? result.observed_leaf_id : rune.target_leaf_id;
  const int attack_leaf_id = result.attack_leaf_id;

  draw_blade_annotations(image, rune, observed_leaf_id, attack_leaf_id);
  tools::draw_point(image, rune.r_center, {0, 0, 255}, 4);
  tools::draw_text(
    image, "R",
    cv::Point(static_cast<int>(rune.r_center.x + 8.0f), static_cast<int>(rune.r_center.y - 8.0f)),
    {0, 0, 255}, 0.6, 2);
  tools::draw_text(
    image, fmt::format("observed leaf: {}  attack leaf: {}", observed_leaf_id, attack_leaf_id),
    {10, 28}, {255, 255, 255}, 0.7, 2);

  const int candidate_count = std::clamp(
    rune.pnp_debug.candidate_count, 0, static_cast<int>(rune.pnp_debug.candidates.size()));
  const auto draw_candidate = [&](int index, const cv::Scalar & color) {
    if (index < 0 || index >= candidate_count) return false;
    const auto & candidate = rune.pnp_debug.candidates[static_cast<std::size_t>(index)];
    if (
      !candidate.valid || !candidate.r_center_world.allFinite() ||
      !candidate.rotation_world.allFinite()) {
      return false;
    }
    draw_reprojection(
      image, solver, candidate.r_center_world, candidate.rotation_world, color);
    return true;
  };

  const int selected_index = rune.pnp_debug.selected_index;
  for (int index = 0; index < candidate_count; ++index) {
    if (index != selected_index) draw_candidate(index, {255, 0, 0});
  }
  if (!draw_candidate(selected_index, {0, 255, 0})) {
    draw_reprojection(image, solver, rune.xyz_in_world, rune.rotation_world, {0, 255, 0});
  }
  const double observed_angle_deg = wrap_angle(rune.physical_angle) * 180.0 / CV_PI;
  tools::draw_text(
    image, fmt::format("green observed angle: {:.1f}deg", observed_angle_deg), {10, 56},
    {0, 255, 0}, 0.7, 2);
  if (!result.valid || result.predict_time_abs <= 0.0 || !std::isfinite(result.predicted_angle)) {
    return;
  }

  const double predicted_angle = result.predicted_angle;
  // 使用和弹道解算完全相同的输入：pose_filter 位姿 + 法向量滤波
  const auto & pose = aimer.pose_snapshot();
  const Eigen::Vector3d & r_center_world =
    pose.valid ? pose.filtered_xyz_world : rune.xyz_in_world;
  const Eigen::Matrix3d & rotation_world =
    pose.valid ? pose.filtered_rotation_world : rune.rotation_world;
  const Eigen::Vector3d n = aimer.get_averaged_normal(rotation_world);
  const Eigen::Matrix3d predicted_rotation =
    tools::rotation_matrix(Eigen::Vector3d(
      std::atan2(n.y(), n.x()),
      -std::atan2(n.z(), std::sqrt(n.x() * n.x() + n.y() * n.y())),
      predicted_angle + CV_PI / 2.0));
  const auto image_points =
    draw_reprojection(image, solver, r_center_world, predicted_rotation, {0, 0, 255});
  const int predicted_leaf_id =
    result.attack_leaf_id >= 0 ? result.attack_leaf_id : observed_leaf_id;
  tools::draw_text(
    image,
    fmt::format(
      "predict attack leaf: {}  dt:{:.1f}ms", predicted_leaf_id,
      std::max(0.0, (result.predict_time_abs - result.observed_time_abs) * 1e3)),
    {10, 84}, {0, 0, 255}, 0.7, 2);
  const double predicted_angle_deg = wrap_angle(predicted_angle) * 180.0 / CV_PI;
  const double prediction_delta_deg =
    wrap_angle(predicted_angle - rune.physical_angle) * 180.0 / CV_PI;
  tools::draw_text(
    image,
    fmt::format(
      "red predicted angle: {:.1f}deg  delta:{:+.1f}deg", predicted_angle_deg,
      prediction_delta_deg),
    {10, 112}, {0, 0, 255}, 0.7, 2);
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

void maybe_show(const std::string & window_name, cv::Mat image)
{
  static std::map<std::string, std::chrono::steady_clock::time_point> last_display;
  const auto now = std::chrono::steady_clock::now();
  if (
    last_display[window_name].time_since_epoch().count() != 0 &&
    now - last_display[window_name] < kDisplayPeriod) {
    return;
  }
  last_display[window_name] = now;
  cv::resize(image, image, {}, 0.5, 0.5);
  cv::imshow(window_name, image);
}
}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }
  const auto r_center_view =
    auto_buff::parse_r_center_debug_view(cli.get<std::string>("r-center-view"));
  if (!r_center_view.has_value()) {
    tools::logger()->error("Unsupported R-center view: {}", cli.get<std::string>("r-center-view"));
    return 1;
  }

  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::Exiter exiter;

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  auto yaml = tools::load(config_path);
  const auto exposure_ms = tools::read<double>(yaml, "exposure_ms");
  const auto exposure_center_offset =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  tools::logger()->info(
    "Image q timestamp offset: -{:.3f} ms (exposure center)", exposure_ms * 0.5);

  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);
  auto_buff::Aimer aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  std::chrono::steady_clock::time_point time_origin;
  bool time_origin_ready = false;

  int frame_count = 0;
  BuffPerfLogState perf_log_state;
  auto fps_report_stamp = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    camera.read(img, t);
    if (!time_origin_ready) {
      time_origin = t;
      time_origin_ready = true;
    }
    auto img_time = t - exposure_center_offset;

    const int submitted_frame = frame_count++;
    const bool submit_ok = detector.submit(img, submitted_frame, img_time);
    perf_log_state.record_submit(submit_ok);

    std::optional<auto_buff::PowerRune> power_runes;
    cv::Mat result_img;
    int result_frame_count = -1;
    double detect_dt_ms = 0.0;
    auto_buff::BuffDetectPerfStats detect_perf_stats;
    std::chrono::steady_clock::time_point result_t;
    auto_buff::RCenterRefineDebug r_center_debug;
    while (detector.fetch(
      power_runes, result_img, result_frame_count, detect_dt_ms, true, &r_center_debug, &result_t,
      &detect_perf_stats)) {
      perf_log_state.record_detect(detect_dt_ms, detect_perf_stats, !power_runes.has_value());

      auto detection_img = result_img.clone();
      Eigen::Quaterniond q = cboard.imu_at(result_t);
      solver.set_R_gimbal2world(q);
      solver.solve(power_runes);
      const double observed_time_abs = tools::delta_time(result_t, time_origin);
      const double now_time_abs = tools::delta_time(std::chrono::steady_clock::now(), time_origin);
      if (power_runes) power_runes->last_observed_time = observed_time_abs;

      io::Command command = {false, false, 0, 0};
      if (power_runes) {
        if (power_runes->pnp_valid) perf_log_state.record_fit_input();
        command = aimer.aim(*power_runes, observed_time_abs, now_time_abs, cboard.bullet_speed);
      } else {
        aimer.notify_observation_missed();
      }

      cboard.send(command);

      nlohmann::json data;
      data["detect_dt_ms"] = detect_dt_ms;
      data["filtered_pose_valid"] = 0;
      data["used_filtered_pose"] = 0;
      if (power_runes.has_value()) {
        auto & p = power_runes.value();
        data["buff_R_yaw"] = p.ypd_in_world[0];
        data["buff_R_pitch"] = p.ypd_in_world[1];
        data["buff_R_dis"] = p.ypd_in_world[2];
        data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
        data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
        data["buff_roll"] = p.ypr_in_world[2] * 57.3;
        const auto & pose_snapshot = aimer.pose_snapshot();
        if (pose_snapshot.valid) {
          data["filtered_pose_valid"] = 1;
          data["filtered_buff_R_yaw"] = pose_snapshot.filtered_ypd_world[0] * 57.3;
          data["filtered_buff_R_pitch"] = pose_snapshot.filtered_ypd_world[1] * 57.3;
          data["filtered_buff_R_dist"] = pose_snapshot.filtered_ypd_world[2];
          data["filtered_buff_yaw"] = pose_snapshot.filtered_buff_yaw * 57.3;
        }

        if (p.pnp_valid) {
          draw_result_overlay(result_img, p, solver, aimer);
          const auto snapshot = aimer.prediction_snapshot();
          perf_log_state.record_predictor(snapshot);
          data["observed_angle"] = p.physical_angle * 57.3;
          data["predict_samples"] = snapshot.sample_count;
          data["predict_cost"] = snapshot.average_cost;
          data["predict_backend"] =
            snapshot.small_predictor == auto_buff::SmallRunePredictor::TARGET_EKF ? "TARGET_EKF"
                                                                                  : "CERES";
          data["a"] = snapshot.energy_tri.a;
          data["w"] = snapshot.energy_tri.w;
          data["c"] = snapshot.energy_tri.c;
          data["t0"] = snapshot.energy_tri.t0;
          if (snapshot.has_ceres_solve_time)
            data["ceres_solve_ms"] = snapshot.last_ceres_solve_time_ms;

          const auto & ballistic_result = aimer.last_ballistic_result();
          data["used_filtered_pose"] = ballistic_result.used_filtered_pose ? 1 : 0;
          data["aim_source"] = static_cast<int>(ballistic_result.aim_source);
          if (
            ballistic_result.valid && ballistic_result.predict_time_abs > 0.0 &&
            std::isfinite(ballistic_result.predicted_angle)) {
            const double predicted_angle = ballistic_result.predicted_angle;
            data["angle"] = predicted_angle * 57.3;
            data["delta_angle"] = (predicted_angle - p.physical_angle) * 57.3;
            data["spd"] = aimer.prediction_speed(ballistic_result.predict_time_abs) * 57.3;
            data["observed_leaf_id"] = ballistic_result.observed_leaf_id;
            data["attack_leaf_id"] = ballistic_result.attack_leaf_id;
            data["selection_cost"] = ballistic_result.selection_cost;
            if (ballistic_result.observed_leaf_id >= 0 && ballistic_result.attack_leaf_id >= 0) {
              const double theta0_obs =
                zero_angle(p.physical_angle, ballistic_result.observed_leaf_id);
              const double theta0_fit =
                zero_angle(predicted_angle, ballistic_result.attack_leaf_id);
              data["theta0_obs"] = theta0_obs * 57.3;
              data["theta0_fit"] = theta0_fit * 57.3;
              data["theta0_residual"] = wrap_angle(theta0_fit - theta0_obs) * 57.3;
            }
          }
        }
      }

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      data["gimbal_yaw"] = ypr[0] * 57.3;
      data["gimbal_pitch"] = ypr[1] * 57.3;
      if (command.control) {
        data["cmd_yaw"] = command.yaw * 57.3;
        data["cmd_pitch"] = command.pitch * 57.3;
        data["shoot"] = command.shoot ? 1 : 0;
      }
      plotter.plot(data);

      auto_buff::show_r_center_debug_views(
        *r_center_view, detection_img, r_center_debug, "buff refine");
      if (kShowDetectorView) {
        draw_buff_detection(detection_img, power_runes, result_frame_count, r_center_debug);
        maybe_show("buff detection", detection_img);
      }
      if (kShowResultView) maybe_show("result", result_img);
    }

    const auto fps_now = std::chrono::steady_clock::now();
    const auto report_dt = tools::delta_time(fps_now, fps_report_stamp);
    if (kEnablePerfLog && report_dt >= kPerfLogPeriodSec) {
      perf_log_state.log_and_reset(detector, report_dt);
      fps_report_stamp = fps_now;
    }

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  const double final_report_dt =
    tools::delta_time(std::chrono::steady_clock::now(), fps_report_stamp);
  if (kEnablePerfLog && final_report_dt > 1e-6) {
    perf_log_state.log_and_reset(detector, final_report_dt);
  }
  return 0;
}
