#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
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
constexpr bool kShowDetectorView = true;   // 是否显示"buff detection" 窗口
constexpr bool kShowResultView = true;     // 是否显示 "result" 窗口（pnp）
constexpr bool kEnablePerfLog = true;      // 是否输出性能日志（仅当前调试入口生效）
constexpr double kPerfLogPeriodSec = 1.0;  // 性能日志输出周期，单位：秒
constexpr auto kDisplayPeriod =
  std::chrono::milliseconds(16);  // OpenCV 图像窗口显示限频。16ms 大概是 60 FPS
constexpr double kLeafAngleStep = 2.0 * CV_PI / 5.0;

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{record          | false | Record raw camera frames without debug overlays }"
  "{record-debug    | false | Record result frames with detection and prediction overlays }"
  "{id-debug        | false | Save PnP and leaf-ID anomaly snapshots with raw and result overlay "
  "video }"
  "{r-center-view  | off | R-center view: off, roi, binary, contours or all }"
  "{@config-path   | | yaml配置文件路径 }";

double wrap_angle(double value) { return std::atan2(std::sin(value), std::cos(value)); }

double zero_angle(double angle, int leaf_id)
{
  if (leaf_id < 0) return angle;
  return angle - static_cast<double>(leaf_id) * kLeafAngleStep;
}

std::string debug_record_prefix()
{
  const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  return fmt::format("records/buff_debug_{}", stamp);
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

constexpr double kIdDiagnosticHistorySec = 2.0;
constexpr double kIdDiagnosticDumpGapSec = 1.0;
constexpr double kIdDiagnosticAngleRateThreshold = 2.5;
constexpr double kIdDiagnosticReprojectionThreshold = 2.0;

struct IdDiagnosticRecord
{
  std::string event;
  int frame_id = -1;
  double observed_time_abs = 0.0;
  double now_time_abs = 0.0;
  double result_age_ms = 0.0;
  double detect_dt_ms = 0.0;
  auto_buff::BuffDetectPerfStats detect_perf_stats;
  std::size_t async_stale_drop_count = 0;
  std::size_t async_out_of_order_drop_count = 0;
  Eigen::Quaterniond result_gimbal_q = Eigen::Quaterniond::Identity();
  io::GimbalState gimbal_state{};
  auto_aim::Plan plan{};
  auto_buff::BuffBallisticResult ballistic;
  auto_buff::BuffPoseSnapshot pose;
  auto_buff::BuffPoseFilterDebug pose_debug;
  auto_buff::CommandGuardDebug command_guard;
  bool has_rune = false;
  bool pnp_valid = false;
  bool r_center_refined = false;
  auto_buff::RCenterRefineDebug r_center_debug;
  auto_buff::BuffTargetSelectionDebug target_selection_debug;
  cv::Point2f r_center;
  cv::Point2f coarse_r_center;
  cv::Point2f refined_r_center;
  std::array<cv::Point2f, 4> target_points{};
  int target_point_count = 0;
  double target_confidence = 0.0;
  int target_label = -1;
  int target_leaf_id = -1;
  int light_num = 0;
  double rune_last_observed_time = 0.0;
  cv::Point2f target_center;
  std::vector<auto_buff::FanBlade> fanblades;
  Eigen::Vector3d rune_xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d rune_ypr_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d rune_ypd_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d blade_xyz_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d blade_ypd_in_world = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_world = Eigen::Matrix3d::Identity();
  Eigen::Vector3d target_center_world = Eigen::Vector3d::Zero();
  double physical_angle = 0.0;
  double reprojection_error = auto_buff::INF;
  double raw_angle_rate = 0.0;
  auto_buff::PnpDebug pnp_debug;
  bool has_id_match = false;
  auto_buff::AngleEstimate::IdMatchDebug id_match;
  bool has_estimator_debug = false;
  auto_buff::AngleEstimate::DebugSnapshot estimator_debug;
  bool has_model_prediction = false;
  int predictor_backend = -1;
  std::array<double, 5> model_expected_angles{};
  std::array<double, 5> model_residuals{};
  double predicted_angle_at_observation = std::numeric_limits<double>::quiet_NaN();
  double predicted_speed_at_observation = std::numeric_limits<double>::quiet_NaN();
  double predicted_speed_at_ballistic_time = std::numeric_limits<double>::quiet_NaN();
  double observed_zero_angle = std::numeric_limits<double>::quiet_NaN();
  double predicted_zero_angle = std::numeric_limits<double>::quiet_NaN();
  double zero_phase_residual = std::numeric_limits<double>::quiet_NaN();
};

struct CameraDiagnosticRecord
{
  int frame_id = -1;
  double captured_time_abs = 0.0;
  bool submit_ok = false;
  int image_width = 0;
  int image_height = 0;
  Eigen::Quaterniond gimbal_q = Eigen::Quaterniond::Identity();
  std::size_t async_stale_drop_count = 0;
  std::size_t async_out_of_order_drop_count = 0;
};

class IdDiagnosisRecorder
{
public:
  IdDiagnosisRecorder(bool enabled, std::filesystem::path config_source_path)
  : enabled_(enabled), config_source_path_(std::move(config_source_path))
  {
    if (enabled_) open_trace_files();
  }

  ~IdDiagnosisRecorder() { flush_trace_files(); }

  void capture_camera(const CameraDiagnosticRecord & record)
  {
    if (!enabled_ || !trace_ready_) return;
    camera_out_ << record.frame_id << ',' << record.captured_time_abs << ',' << record.submit_ok
                << ',' << record.image_width << ',' << record.image_height;
    write_quaternion(camera_out_, record.gimbal_q);
    camera_out_ << ',' << record.async_stale_drop_count << ','
                << record.async_out_of_order_drop_count << '\n';
    mark_trace_row();
  }

  void capture(IdDiagnosticRecord record)
  {
    if (!enabled_) return;

    if (record.pnp_valid && has_previous_angle_) {
      const double dt = record.observed_time_abs - previous_angle_time_;
      if (dt > 1e-6) {
        record.raw_angle_rate = std::abs(wrap_angle(record.physical_angle - previous_angle_)) / dt;
      }
    }
    if (record.pnp_valid) {
      previous_angle_ = record.physical_angle;
      previous_angle_time_ = record.observed_time_abs;
      has_previous_angle_ = true;
    }

    if (record.pnp_valid && record.reprojection_error > kIdDiagnosticReprojectionThreshold) {
      append_event(record.event, "high-reprojection");
    }
    if (record.pnp_valid && record.raw_angle_rate > kIdDiagnosticAngleRateThreshold) {
      append_event(record.event, "angle-jump");
    }
    if (record.has_rune && !record.pnp_valid) append_event(record.event, "pnp-rejected");
    if (record.has_id_match && record.id_match.attempted) {
      if (!record.id_match.accepted) append_event(record.event, "id-rejected");
      if (
        record.id_match.accepted && record.id_match.previous_leaf_id >= 0 &&
        record.id_match.previous_leaf_id != record.id_match.selected_leaf_id) {
        append_event(record.event, "id-switch");
      }
      if (record.id_match.model_overrode_kinematic) {
        append_event(record.event, "model-override");
      }
    }
    if (record.has_model_prediction) {
      const double model_residual =
        *std::min_element(record.model_residuals.begin(), record.model_residuals.end());
      if (model_residual > 0.35) append_event(record.event, "model-mismatch");
    }

    write_full_trace(record);
    write_pipeline_trace(record);
    write_candidate_trace(record);
    write_fanblade_trace(record);
    write_r_center_trace(record);
    mark_trace_row();

    records_.push_back(record);
    while (!records_.empty() && record.observed_time_abs - records_.front().observed_time_abs >
                                  kIdDiagnosticHistorySec) {
      records_.pop_front();
    }

    if (
      record.event.empty() ||
      record.observed_time_abs - last_dump_time_ < kIdDiagnosticDumpGapSec) {
      return;
    }
    dump(record.event);
    last_dump_time_ = record.observed_time_abs;
  }

private:
  static void append_event(std::string & event, const char * reason)
  {
    if (!event.empty()) event += ';';
    event += reason;
  }

  static void write_point(std::ofstream & out, const cv::Point2f & point)
  {
    out << ',' << point.x << ',' << point.y;
  }

  static void write_vector3(std::ofstream & out, const Eigen::Vector3d & value)
  {
    out << ',' << value.x() << ',' << value.y() << ',' << value.z();
  }

  static void write_matrix3(std::ofstream & out, const Eigen::Matrix3d & value)
  {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) out << ',' << value(row, column);
    }
  }

  static void write_quaternion(std::ofstream & out, const Eigen::Quaterniond & value)
  {
    out << ',' << value.w() << ',' << value.x() << ',' << value.y() << ',' << value.z();
  }

  static void write_base_record(std::ofstream & out, const IdDiagnosticRecord & record)
  {
    out << record.event << ',' << record.frame_id << ',' << record.observed_time_abs << ','
        << record.now_time_abs << ',' << record.detect_dt_ms << ',' << record.has_rune << ','
        << record.pnp_valid << ',' << record.r_center_refined;
    write_point(out, record.r_center);
    write_point(out, record.coarse_r_center);
    write_point(out, record.refined_r_center);
    out << ',' << record.target_point_count << ',' << record.target_confidence;
    write_point(out, record.target_center);
    for (const auto & point : record.target_points) write_point(out, point);
    out << ',' << record.physical_angle << ',' << record.reprojection_error << ','
        << record.raw_angle_rate << ',' << record.pnp_debug.candidate_count << ','
        << record.pnp_debug.lowest_reprojection_index << ',' << record.pnp_debug.selected_index
        << ',' << record.pnp_debug.hard_gate_enabled << ','
        << record.pnp_debug.prediction_rotation_enabled << ','
        << record.pnp_debug.has_prediction_reference << ','
        << pnp_selection_reason_text(record.pnp_debug.selection_reason);
    for (const auto & candidate : record.pnp_debug.candidates) {
      out << ',' << candidate.valid << ',' << candidate.reprojection_error << ','
          << candidate.physical_angle << ',' << candidate.depth << ','
          << candidate.rotation_distance_to_prediction;
      write_vector3(out, candidate.r_center_world);
      write_matrix3(out, candidate.rotation_world);
      write_vector3(out, candidate.normal_world);
      out << ',' << candidate.hard_gate_passed << ',' << candidate.depth_gate_passed << ','
          << candidate.position_gate_passed << ',' << candidate.rotation_gate_passed << ','
          << candidate.normal_gate_passed << ',' << candidate.hard_gate_reject_mask;
    }
    out << ',' << record.has_id_match << ',' << record.id_match.attempted << ','
        << record.id_match.accepted << ',' << record.id_match.short_history << ','
        << record.id_match.continuity_rejected << ',' << record.id_match.previous_leaf_id << ','
        << record.id_match.raw_best_leaf_id << ',' << record.id_match.continuity_leaf_id << ','
        << record.id_match.selected_leaf_id << ',' << record.id_match.time << ','
        << record.id_match.observed_angle << ',' << record.id_match.aligned_angle << ','
        << record.id_match.raw_best_residual << ',' << record.id_match.best_residual;
    for (double expected : record.id_match.expected_angles) out << ',' << expected;
    for (double residual : record.id_match.residuals) out << ',' << residual;
    out << ',' << record.has_model_prediction << ',' << record.predictor_backend;
    for (double expected : record.model_expected_angles) out << ',' << expected;
    for (double residual : record.model_residuals) out << ',' << residual;
  }

  static void write_record(std::ofstream & out, const IdDiagnosticRecord & record)
  {
    write_base_record(out, record);
    out << '\n';
  }

  static void write_base_header(std::ofstream & out)
  {
    out
      << "event,frame_id,observed_time_abs,now_time_abs,detect_dt_ms,has_rune,pnp_valid,"
         "r_center_refined,r_center_x,r_center_y,coarse_r_center_x,coarse_r_center_y,"
         "refined_r_center_x,refined_r_center_y,target_point_count,target_confidence,"
         "target_center_x,target_center_y,"
         "target_p0_x,target_p0_y,target_p1_x,target_p1_y,target_p2_x,target_p2_y,"
         "target_p3_x,target_p3_y,physical_angle,reprojection_error,raw_angle_rate,"
         "pnp_candidate_count,pnp_lowest_reprojection_index,pnp_selected_index,"
         "pnp_hard_gate_enabled,"
         "pnp_prediction_rotation_enabled,pnp_has_prediction_reference,pnp_selection_reason,"
         "pnp0_valid,pnp0_reprojection_error,pnp0_physical_angle,pnp0_depth,"
         "pnp0_rotation_distance_to_prediction,"
         "pnp0_r_center_world_x,pnp0_r_center_world_y,pnp0_r_center_world_z,"
         "pnp0_rotation_world_00,pnp0_rotation_world_01,pnp0_rotation_world_02,"
         "pnp0_rotation_world_10,pnp0_rotation_world_11,pnp0_rotation_world_12,"
         "pnp0_rotation_world_20,pnp0_rotation_world_21,pnp0_rotation_world_22,"
         "pnp0_normal_world_x,pnp0_normal_world_y,pnp0_normal_world_z,"
         "pnp0_hard_gate_passed,pnp0_depth_gate_passed,pnp0_position_gate_passed,"
         "pnp0_rotation_gate_passed,pnp0_normal_gate_passed,pnp0_hard_gate_reject_mask,pnp1_valid,"
         "pnp1_reprojection_error,pnp1_physical_angle,pnp1_depth,"
         "pnp1_rotation_distance_to_prediction,"
         "pnp1_r_center_world_x,pnp1_r_center_world_y,pnp1_r_center_world_z,"
         "pnp1_rotation_world_00,pnp1_rotation_world_01,pnp1_rotation_world_02,"
         "pnp1_rotation_world_10,pnp1_rotation_world_11,pnp1_rotation_world_12,"
         "pnp1_rotation_world_20,pnp1_rotation_world_21,pnp1_rotation_world_22,"
         "pnp1_normal_world_x,pnp1_normal_world_y,pnp1_normal_world_z,"
         "pnp1_hard_gate_passed,pnp1_depth_gate_passed,pnp1_position_gate_passed,"
         "pnp1_rotation_gate_passed,pnp1_normal_gate_passed,pnp1_hard_gate_reject_mask,has_id_"
         "match,"
         "id_attempted,id_accepted,id_short_history,id_continuity_rejected,id_previous,"
         "id_raw_best_leaf_id,id_continuity_leaf_id,id_selected,id_time,id_observed_angle,"
         "id_aligned_angle,id_raw_best_residual,id_best_residual,id0_expected,id1_expected,"
         "id2_expected,id3_expected,id4_expected,id0_residual,id1_residual,id2_residual,"
         "id3_residual,id4_residual,has_model_prediction,predictor_backend,model_id0_expected,"
         "model_id1_expected,model_id2_expected,model_id3_expected,model_id4_expected,"
         "model_id0_residual,model_id1_residual,model_id2_residual,model_id3_residual,"
         "model_id4_residual";
  }

  static void write_plan(std::ofstream & out, const auto_aim::Plan & plan)
  {
    out << ',' << plan.control << ',' << plan.fire << ',' << plan.target_yaw << ','
        << plan.target_pitch << ',' << plan.yaw << ',' << plan.yaw_vel << ',' << plan.yaw_acc << ','
        << plan.pitch << ',' << plan.pitch_vel << ',' << plan.pitch_acc;
  }

  static void write_ballistic(std::ofstream & out, const auto_buff::BuffBallisticResult & result)
  {
    out << ',' << result.valid << ',' << result.used_filtered_pose << ',' << result.yaw << ','
        << result.pitch << ',' << result.fly_time << ',' << result.predict_time_abs << ','
        << result.observed_time_abs << ',' << result.now_time_abs << ','
        << result.observation_age_ms << ',' << result.runtime_solve_delay_ms << ','
        << result.tuned_bias_time_ms << ',' << result.delay_time_ms << ',' << result.iterations
        << ',' << result.attack_leaf_id << ',' << result.observed_leaf_id << ','
        << result.selection_cost;
    write_vector3(out, result.target_world);
  }

  static void write_pose(std::ofstream & out, const auto_buff::BuffPoseSnapshot & pose)
  {
    out << ',' << pose.valid;
    write_vector3(out, pose.filtered_ypd_world);
    write_vector3(out, pose.filtered_xyz_world);
    out << ',' << pose.filtered_buff_yaw;
    write_matrix3(out, pose.filtered_rotation_world);
    write_vector3(out, pose.filtered_target_center_world);
    out << ',' << pose.observed_time_abs;
  }

  static void write_estimator_debug(
    std::ofstream & out, const auto_buff::AngleEstimate::DebugSnapshot & debug)
  {
    const auto & summary = debug.summary;
    const auto & energy = summary.energy_tri;
    out << ',' << static_cast<int>(summary.state) << ',' << summary.average_cost << ','
        << summary.last_ceres_solve_time_ms << ',' << summary.last_small_ekf_update_time_ms << ','
        << summary.sample_count << ',' << summary.fitted << ',' << summary.has_ceres_solve_time
        << ',' << static_cast<int>(summary.predictor_backend) << ','
        << static_cast<int>(summary.small_predictor) << ',' << energy.a << ',' << energy.w << ','
        << energy.c << ',' << energy.t0 << ',' << energy.k << ',' << energy.b << ','
        << static_cast<int>(energy.mode) << ',' << energy.direction << ',' << debug.current_leaf_id
        << ',' << debug.lost_count << ',' << debug.start_time << ',' << debug.latest_sample_version
        << ',' << debug.submitted_sample_version << ',' << debug.committed_fit_version << ','
        << debug.id_epoch << ',' << debug.model_epoch << ',' << debug.last_model_sample_time_abs
        << ',' << debug.confirmed_observations_in_epoch << ',' << debug.recent_leaf_data_count;
    for (const auto & leaf : debug.recent_leaf_datas) {
      out << ',' << leaf.angle << ',' << leaf.zero_angle << ',' << leaf.t << ',' << leaf.leaf_id;
    }

    const auto & id = debug.id_match;
    out << ',' << id.history_size << ',' << id.history_previous_leaf_id << ','
        << id.history_last_leaf_id << ',' << id.history_previous_angle << ','
        << id.history_last_angle << ',' << id.history_previous_zero_angle << ','
        << id.history_last_zero_angle << ',' << id.history_dt << ',' << id.extrapolation_dt << ','
        << id.estimated_zero_delta << ',' << id.current_zero_angle << ',' << id.zero_innovation
        << ',' << id.zero_innovation_limit << ',' << id.fused_same_timestamp << ','
        << static_cast<int>(id.model_prior_status) << ',' << id.model_available << ','
        << id.model_used << ',' << id.model_overrode_kinematic << ',' << id.kinematic_leaf_id << ','
        << id.model_leaf_id << ',' << id.model_epoch << ',' << id.confirmed_observations_in_epoch
        << ',' << id.model_age_sec << ',' << id.model_zero_angle;
    for (double expected : id.model_expected_angles) out << ',' << expected;
    for (double residual : id.model_residuals) out << ',' << residual;

    const auto & ekf = debug.small_ekf;
    out << ',' << ekf.initialized_before << ',' << ekf.reset << ',' << ekf.short_dt_fusion << ','
        << ekf.phase_realigned << ',' << ekf.updated << ',' << ekf.phase_shift << ','
        << ekf.input_angle << ',' << ekf.time << ',' << ekf.dt << ',' << ekf.state_angle_before
        << ',' << ekf.state_speed_before << ',' << ekf.raw_innovation << ','
        << ekf.aligned_measurement << ',' << ekf.predicted_angle << ',' << ekf.predicted_speed
        << ',' << ekf.state_angle_after << ',' << ekf.state_speed_after << ',' << ekf.p00 << ','
        << ekf.p01 << ',' << ekf.p10 << ',' << ekf.p11 << ',' << ekf.nis;
  }

  static void write_estimator_debug_header(std::ofstream & out)
  {
    out << ",estimator_state,estimator_average_cost,estimator_last_ceres_solve_time_ms,"
           "estimator_last_small_ekf_update_time_ms,estimator_sample_count,estimator_fitted,"
           "estimator_has_ceres_solve_time,estimator_predictor_backend,"
           "estimator_small_predictor,estimator_energy_a,estimator_energy_w,"
           "estimator_energy_c,estimator_energy_t0,estimator_energy_k,estimator_energy_b,"
           "estimator_energy_mode,estimator_energy_direction,estimator_current_leaf_id,"
           "estimator_lost_count,estimator_start_time,estimator_latest_sample_version,"
           "estimator_submitted_sample_version,estimator_committed_fit_version,"
           "estimator_id_epoch,estimator_model_epoch,estimator_last_model_sample_time_abs,"
           "estimator_confirmed_observations_in_epoch,"
           "estimator_recent_leaf_data_count";
    for (int index = 0; index < 3; ++index) {
      out << ",recent_leaf" << index << "_angle,recent_leaf" << index << "_zero_angle,recent_leaf"
          << index << "_time,recent_leaf" << index << "_id";
    }
    out << ",id_history_size,id_history_previous_leaf_id,id_history_last_leaf_id,"
           "id_history_previous_angle,id_history_last_angle,id_history_previous_zero_angle,"
           "id_history_last_zero_angle,id_history_dt,id_extrapolation_dt,id_estimated_zero_delta,"
           "id_current_zero_angle,id_zero_innovation,id_zero_innovation_limit,"
           "id_fused_same_timestamp,"
           "id_model_prior_status,id_model_available,id_model_used,id_model_overrode_kinematic,"
           "id_kinematic_leaf_id,id_model_leaf_id,id_model_epoch,"
           "id_confirmed_observations_in_epoch,id_model_age_sec,id_model_zero_angle";
    for (int index = 0; index < 5; ++index) out << ",id_model_expected_" << index;
    for (int index = 0; index < 5; ++index) out << ",id_model_residual_" << index;
    out << ","
           "ekf_initialized_before,ekf_reset,ekf_short_dt_fusion,ekf_phase_realigned,ekf_updated,"
           "ekf_phase_shift,ekf_input_angle,ekf_time,ekf_dt,"
           "ekf_state_angle_before,ekf_state_speed_before,ekf_raw_innovation,"
           "ekf_aligned_measurement,ekf_predicted_angle,ekf_predicted_speed,"
           "ekf_state_angle_after,ekf_state_speed_after,ekf_p00,ekf_p01,ekf_p10,ekf_p11,ekf_nis";
  }

  static const char * pnp_selection_reason_text(auto_buff::PnpSelectionReason reason)
  {
    switch (reason) {
      case auto_buff::PnpSelectionReason::LOWEST_REPROJECTION:
        return "lowest-reprojection";
      case auto_buff::PnpSelectionReason::ZERO_CONTINUITY:
        return "zero-continuity";
      case auto_buff::PnpSelectionReason::GATE_REJECTED:
        return "gate-rejected";
      case auto_buff::PnpSelectionReason::FACING_CAMERA:
        return "facing-camera";
      case auto_buff::PnpSelectionReason::NONE:
      default:
        return "none";
    }
  }

  static void write_nan(std::ofstream & out, int count)
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int index = 0; index < count; ++index) out << ',' << nan;
  }

  static void write_state6(std::ofstream & out, const Eigen::Matrix<double, 6, 1> & state)
  {
    for (int index = 0; index < state.size(); ++index) out << ',' << state[index];
  }

  static void write_vector5(std::ofstream & out, const Eigen::Matrix<double, 5, 1> & value)
  {
    for (int index = 0; index < value.size(); ++index) out << ',' << value[index];
  }

  static void write_pnp_candidate_pipeline(
    std::ofstream & out, const auto_buff::PnpCandidateDebug & candidate)
  {
    out << ',' << candidate.valid;
    if (!candidate.valid) {
      write_nan(out, 41);
      return;
    }
    out << ',' << candidate.reprojection_error;
    for (double error : candidate.point_reprojection_errors) out << ',' << error;
    write_vector3(out, candidate.rvec);
    write_vector3(out, candidate.tvec);
    write_vector3(out, candidate.ypr_in_world);
    write_vector3(out, candidate.r_center_world);
    write_matrix3(out, candidate.rotation_world);
    write_vector3(out, candidate.normal_world);
    out << ',' << candidate.physical_angle << ',' << candidate.depth << ','
        << candidate.normal_camera_z << ',' << candidate.normal_facing << ','
        << candidate.rotation_distance_to_prediction << ',' << candidate.hard_gate_passed << ','
        << candidate.depth_gate_passed << ',' << candidate.position_gate_passed << ','
        << candidate.rotation_gate_passed << ',' << candidate.normal_gate_passed << ','
        << candidate.hard_gate_reject_mask;
  }

  static void write_pose_filter_debug(
    std::ofstream & out, const auto_buff::BuffPoseFilterDebug & debug)
  {
    out << ',' << static_cast<int>(debug.update_kind) << ',' << debug.enabled << ','
        << debug.input_valid << ',' << debug.dt;
    write_state6(out, debug.state_before);
    write_state6(out, debug.state_predicted);
    write_state6(out, debug.state_after);
    write_vector5(out, debug.linear_innovation);
    write_vector3(out, debug.blade_innovation);
    write_vector3(out, debug.raw_minus_filtered_r_xyz);
    write_vector3(out, debug.raw_minus_filtered_target_xyz);
    out << ',' << debug.raw_to_filtered_rotation_distance;
  }

  static void write_ceres_fit_debug(
    std::ofstream & out, const auto_buff::AngleEstimate::CeresFitDebug & debug)
  {
    out << ',' << debug.job_submitted << ',' << debug.result_received << ',' << debug.result_applied
        << ',' << debug.result_rejected;
    if (debug.job_submitted) {
      out << ',' << debug.submitted_epoch << ',' << debug.submitted_sample_version << ','
          << debug.submitted_sample_count << ',' << debug.submitted_first_sample_time << ','
          << debug.submitted_last_sample_time;
    } else {
      write_nan(out, 5);
    }
    if (debug.result_received) {
      out << ',' << debug.result_epoch << ',' << debug.result_sample_version << ','
          << debug.result_sample_count << ',' << debug.result_first_sample_time << ','
          << debug.result_last_sample_time << ',' << debug.previous_k << ',' << debug.previous_b
          << ',' << debug.result_k << ',' << debug.result_b << ',' << debug.average_cost << ','
          << debug.solve_time_ms << ',' << debug.total_fit_wall_ms << ',' << debug.residual_count;
    } else {
      write_nan(out, 13);
    }
  }

  static void write_command_guard_debug(
    std::ofstream & out, const auto_buff::CommandGuardDebug & debug)
  {
    out << ',' << debug.applied << ',' << debug.has_fresh_last << ',' << debug.yaw_limited << ','
        << debug.pitch_limited << ',' << debug.raw_dt << ',' << debug.dt << ','
        << debug.reference_yaw << ',' << debug.reference_pitch << ',' << debug.raw_yaw << ','
        << debug.raw_pitch << ',' << debug.yaw_delta << ',' << debug.pitch_delta << ','
        << debug.limited_yaw_delta << ',' << debug.limited_pitch_delta;
  }

  static void write_pipeline_header(std::ofstream & out)
  {
    out << "event,frame_id,observed_time_abs,now_time_abs,pnp_valid,pnp_candidate_count,"
           "pnp_lowest_reprojection_index,pnp_selected_index,pnp_selection_reason,"
           "pnp_hard_gate_enabled,pnp_prediction_rotation_enabled,"
           "pnp_has_prediction_reference,pnp_input_r_x,pnp_input_r_y,pnp_input_p1_x,"
           "pnp_input_p1_y,pnp_input_p2_x,pnp_input_p2_y,pnp_input_p3_x,pnp_input_p3_y,"
           "pnp_input_target_center_x,pnp_input_target_center_y";
    for (int candidate_index = 0; candidate_index < 2; ++candidate_index) {
      out << ",pnp" << candidate_index << "_valid,pnp" << candidate_index
          << "_mean_reprojection_error";
      for (int point_index = 0; point_index < 5; ++point_index) {
        out << ",pnp" << candidate_index << "_point" << point_index << "_reprojection_error";
      }
      out << ",pnp" << candidate_index << "_rvec_x,pnp" << candidate_index << "_rvec_y,pnp"
          << candidate_index << "_rvec_z,pnp" << candidate_index << "_tvec_x,pnp" << candidate_index
          << "_tvec_y,pnp" << candidate_index << "_tvec_z,pnp" << candidate_index
          << "_world_yaw,pnp" << candidate_index << "_world_pitch,pnp" << candidate_index
          << "_world_roll,pnp" << candidate_index << "_r_center_world_x,pnp" << candidate_index
          << "_r_center_world_y,pnp" << candidate_index << "_r_center_world_z";
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          out << ",pnp" << candidate_index << "_rotation_world_" << row << column;
        }
      }
      out << ",pnp" << candidate_index << "_normal_world_x,pnp" << candidate_index
          << "_normal_world_y,pnp" << candidate_index << "_normal_world_z,pnp"
          << "_physical_angle,pnp" << candidate_index << "_depth,pnp" << candidate_index
          << "_normal_camera_z,pnp" << candidate_index << "_normal_facing,pnp" << candidate_index
          << "_rotation_distance_to_prediction,pnp" << candidate_index << "_hard_gate_passed,pnp"
          << candidate_index << "_depth_gate_passed,pnp" << candidate_index
          << "_position_gate_passed,pnp" << candidate_index << "_rotation_gate_passed,pnp"
          << candidate_index << "_normal_gate_passed,pnp" << candidate_index
          << "_hard_gate_reject_mask";
    }
    out << ",pose_update_kind,pose_enabled,pose_input_valid,pose_dt";
    for (const char * stage : {"before", "predicted", "after"}) {
      for (const char * state : {"r_yaw", "r_yaw_rate", "r_pitch", "r_distance", "buff_yaw"}) {
        out << ",pose_state_" << stage << '_' << state;
      }
    }
    for (const char * name : {"r_yaw", "r_pitch", "r_distance", "buff_yaw"}) {
      out << ",pose_linear_innovation_" << name;
    }
    for (const char * name : {"yaw", "pitch", "distance"}) {
      out << ",pose_blade_innovation_" << name;
    }
    for (const char * axis : {"x", "y", "z"}) {
      out << ",pose_raw_minus_filtered_r_" << axis;
    }
    for (const char * axis : {"x", "y", "z"}) {
      out << ",pose_raw_minus_filtered_target_" << axis;
    }
    out << ",pose_raw_to_filtered_rotation_distance,ceres_job_submitted,"
           "ceres_result_received,ceres_result_applied,ceres_result_rejected,"
           "ceres_submitted_epoch,ceres_submitted_sample_version,ceres_submitted_sample_count,"
           "ceres_submitted_first_sample_time,ceres_submitted_last_sample_time,"
           "ceres_result_epoch,ceres_result_sample_version,ceres_result_sample_count,"
           "ceres_result_first_sample_time,ceres_result_last_sample_time,ceres_previous_k,"
           "ceres_previous_b,ceres_result_k,ceres_result_b,ceres_average_cost,"
           "ceres_solve_time_ms,ceres_total_fit_wall_ms,ceres_residual_count,"
           "predict_angle_at_observation,predict_speed_at_observation,ballistic_valid,"
           "ballistic_raw_yaw,ballistic_raw_pitch,ballistic_predicted_angle,"
           "ballistic_predict_time_abs,ballistic_prediction_lead_ms,"
           "predict_speed_at_ballistic_time,observed_zero_angle,predicted_zero_angle,"
           "zero_phase_residual,command_guard_applied,command_guard_has_fresh_last,"
           "command_guard_yaw_limited,command_guard_pitch_limited,command_guard_raw_dt,"
           "command_guard_dt,command_guard_reference_yaw,command_guard_reference_pitch,"
           "command_guard_raw_yaw,command_guard_raw_pitch,command_guard_yaw_delta,"
           "command_guard_pitch_delta,command_guard_limited_yaw_delta,"
           "command_guard_limited_pitch_delta\n";
  }

  void write_pipeline_trace(const IdDiagnosticRecord & record)
  {
    if (!trace_ready_) return;
    pipeline_out_ << record.event << ',' << record.frame_id << ',' << record.observed_time_abs
                  << ',' << record.now_time_abs << ',' << record.pnp_valid << ','
                  << record.pnp_debug.candidate_count << ','
                  << record.pnp_debug.lowest_reprojection_index << ','
                  << record.pnp_debug.selected_index << ','
                  << pnp_selection_reason_text(record.pnp_debug.selection_reason) << ','
                  << record.pnp_debug.hard_gate_enabled << ','
                  << record.pnp_debug.prediction_rotation_enabled << ','
                  << record.pnp_debug.has_prediction_reference;
    if (record.has_rune) {
      write_point(pipeline_out_, record.r_center);
      for (int point_index = 1; point_index < 4; ++point_index) {
        write_point(pipeline_out_, record.target_points[static_cast<std::size_t>(point_index)]);
      }
      write_point(pipeline_out_, record.target_center);
    } else {
      write_nan(pipeline_out_, 10);
    }
    for (const auto & candidate : record.pnp_debug.candidates) {
      write_pnp_candidate_pipeline(pipeline_out_, candidate);
    }
    write_pose_filter_debug(pipeline_out_, record.pose_debug);
    write_ceres_fit_debug(pipeline_out_, record.estimator_debug.ceres_fit);
    pipeline_out_ << ',' << record.predicted_angle_at_observation << ','
                  << record.predicted_speed_at_observation << ',' << record.ballistic.valid;
    if (record.ballistic.valid) {
      const double prediction_lead_ms =
        (record.ballistic.predict_time_abs - record.ballistic.observed_time_abs) * 1e3;
      pipeline_out_ << ',' << record.ballistic.yaw << ',' << record.ballistic.pitch << ','
                    << record.ballistic.predicted_angle << ',' << record.ballistic.predict_time_abs
                    << ',' << prediction_lead_ms << ',' << record.predicted_speed_at_ballistic_time
                    << ',' << record.observed_zero_angle << ',' << record.predicted_zero_angle
                    << ',' << record.zero_phase_residual;
    } else {
      write_nan(pipeline_out_, 9);
    }
    write_command_guard_debug(pipeline_out_, record.command_guard);
    pipeline_out_ << '\n';
  }

  static void write_trace_header(std::ofstream & out)
  {
    write_base_header(out);
    out << ",result_age_ms,preprocess_dt_ms,infer_dt_ms,postprocess_dt_ms,"
           "async_stale_drop_count,async_out_of_order_drop_count,"
           "result_gimbal_q_w,result_gimbal_q_x,result_gimbal_q_y,result_gimbal_q_z,"
           "gimbal_yaw,gimbal_yaw_vel,gimbal_pitch,gimbal_pitch_vel,gimbal_bullet_speed,"
           "gimbal_bullet_count,gimbal_camp,plan_control,plan_fire,plan_target_yaw,"
           "plan_target_pitch,plan_yaw,plan_yaw_vel,plan_yaw_acc,plan_pitch,plan_pitch_vel,"
           "plan_pitch_acc,ballistic_valid,ballistic_used_filtered_pose,ballistic_yaw,"
           "ballistic_pitch,ballistic_fly_time,ballistic_predict_time_abs,"
           "ballistic_observed_time_abs,ballistic_now_time_abs,ballistic_observation_age_ms,"
           "ballistic_runtime_solve_delay_ms,ballistic_tuned_bias_time_ms,ballistic_delay_time_ms,"
           "ballistic_iterations,ballistic_attack_leaf_id,ballistic_observed_leaf_id,"
           "ballistic_selection_cost,ballistic_target_world_x,ballistic_target_world_y,"
           "ballistic_target_world_z,pose_valid,pose_filtered_ypd_x,pose_filtered_ypd_y,"
           "pose_filtered_ypd_z,pose_filtered_xyz_x,pose_filtered_xyz_y,pose_filtered_xyz_z,"
           "pose_filtered_buff_yaw,pose_rotation_00,pose_rotation_01,pose_rotation_02,"
           "pose_rotation_10,pose_rotation_11,pose_rotation_12,pose_rotation_20,"
           "pose_rotation_21,pose_rotation_22,pose_target_center_world_x,"
           "pose_target_center_world_y,pose_target_center_world_z,pose_observed_time_abs,"
           "rune_xyz_world_x,rune_xyz_world_y,rune_xyz_world_z,rune_ypr_world_yaw,"
           "rune_ypr_world_pitch,rune_ypr_world_roll,rune_ypd_world_yaw,"
           "rune_ypd_world_pitch,rune_ypd_world_distance,blade_xyz_world_x,"
           "blade_xyz_world_y,blade_xyz_world_z,blade_ypd_world_yaw,blade_ypd_world_pitch,"
           "blade_ypd_world_distance,rotation_world_00,rotation_world_01,rotation_world_02,"
           "rotation_world_10,rotation_world_11,rotation_world_12,rotation_world_20,"
           "rotation_world_21,rotation_world_22,target_center_world_x,target_center_world_y,"
           "target_center_world_z,target_label,target_leaf_id,light_num,rune_last_observed_time,"
           "selection_raw_object_count,selection_valid_fanblade_count,"
           "selection_has_previous_target,selection_previous_target_center_x,"
           "selection_previous_target_center_y,selection_previous_r_center_x,"
           "selection_previous_r_center_y,selection_previous_target_image_angle,"
           "selection_reason,selection_selected_fanblade_index,selection_selected_source_index,"
           "selection_selected_image_angle,selection_selected_history_angle_residual,"
           "r_center_fallback_reason,r_center_refine_dt_ms,r_center_roi_x,r_center_roi_y,"
           "r_center_roi_width,r_center_roi_height,r_center_selected_contour_index,"
           "r_center_accepted_contour_count,r_center_best_score,r_center_candidate_count,"
           "has_estimator_debug";
    write_estimator_debug_header(out);
    out << '\n';
  }

  static void write_candidate_header(std::ofstream & out)
  {
    out << "event,frame_id,observed_time_abs,raw_object_count,valid_fanblade_count,"
           "has_previous_target,previous_target_center_x,previous_target_center_y,"
           "previous_r_center_x,previous_r_center_y,previous_target_image_angle,selection_reason,"
           "selected_fanblade_index,selected_source_index,selected_image_angle,"
           "selected_history_angle_residual,source_index,fanblade_index,valid_keypoints,label,"
           "confidence,rect_x,rect_y,rect_width,rect_height,keypoint_count,"
           "keypoint0_x,keypoint0_y,keypoint1_x,keypoint1_y,keypoint2_x,keypoint2_y,"
           "keypoint3_x,keypoint3_y,center_x,center_y,image_angle,history_angle_residual,"
           "image_center_distance,selected_as_target\n";
  }

  static void write_fanblade_header(std::ofstream & out)
  {
    out << "event,frame_id,observed_time_abs,slot,type,leaf_id,leaf_angle,leaf_angle_valid,"
           "sort_angle,width,height,label,confidence,center_x,center_y,point_count,"
           "point0_x,point0_y,point1_x,point1_y,point2_x,point2_y,point3_x,point3_y\n";
  }

  static void write_r_center_header(std::ofstream & out)
  {
    out << "event,frame_id,observed_time_abs,refined_valid,fallback_reason,refine_dt_ms,"
           "roi_x,roi_y,roi_width,roi_height,selected_contour_index,accepted_contour_count,"
           "best_score,candidate_count,contour_index,center_in_roi_x,center_in_roi_y,area,"
           "aspect_ratio,child_area_ratio,offset_ratio,score,"
           "reject_mask,selected\n";
  }

  static void write_camera_header(std::ofstream & out)
  {
    out << "frame_id,captured_time_abs,submit_ok,image_width,image_height,gimbal_q_w,"
           "gimbal_q_x,gimbal_q_y,gimbal_q_z,async_stale_drop_count,"
           "async_out_of_order_drop_count\n";
  }

  void open_trace_files()
  {
    std::error_code error;
    std::filesystem::create_directories("records", error);
    if (error) {
      tools::logger()->error("[Buff ID Debug] Cannot create records/: {}", error.message());
      return;
    }

    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    trace_path_ = fmt::format("records/buff_id_trace_{}.csv", stamp);
    pipeline_path_ = fmt::format("records/buff_pipeline_trace_{}.csv", stamp);
    candidate_path_ = fmt::format("records/buff_id_candidates_{}.csv", stamp);
    fanblade_path_ = fmt::format("records/buff_id_fanblades_{}.csv", stamp);
    r_center_path_ = fmt::format("records/buff_id_r_center_{}.csv", stamp);
    camera_path_ = fmt::format("records/buff_id_camera_{}.csv", stamp);
    config_snapshot_path_ = fmt::format("records/buff_replay_config_{}.yaml", stamp);
    manifest_path_ = fmt::format("records/buff_replay_manifest_{}.yaml", stamp);

    trace_out_.open(trace_path_);
    pipeline_out_.open(pipeline_path_);
    candidate_out_.open(candidate_path_);
    fanblade_out_.open(fanblade_path_);
    r_center_out_.open(r_center_path_);
    camera_out_.open(camera_path_);
    if (
      !trace_out_ || !pipeline_out_ || !candidate_out_ || !fanblade_out_ || !r_center_out_ ||
      !camera_out_) {
      tools::logger()->error("[Buff ID Debug] Cannot open one or more continuous trace CSV files.");
      return;
    }

    trace_out_ << std::fixed << std::setprecision(8);
    pipeline_out_ << std::fixed << std::setprecision(8);
    candidate_out_ << std::fixed << std::setprecision(8);
    fanblade_out_ << std::fixed << std::setprecision(8);
    r_center_out_ << std::fixed << std::setprecision(8);
    camera_out_ << std::fixed << std::setprecision(8);
    write_trace_header(trace_out_);
    write_pipeline_header(pipeline_out_);
    write_candidate_header(candidate_out_);
    write_fanblade_header(fanblade_out_);
    write_r_center_header(r_center_out_);
    write_camera_header(camera_out_);

    bool config_snapshot_available = false;
    if (!config_source_path_.empty()) {
      std::filesystem::copy_file(
        config_source_path_, config_snapshot_path_,
        std::filesystem::copy_options::overwrite_existing, error);
      if (error) {
        tools::logger()->warn(
          "[Buff ID Debug] Cannot snapshot config {}: {}", config_source_path_.string(),
          error.message());
        error.clear();
      } else {
        config_snapshot_available = true;
      }
    }
    std::ofstream manifest(manifest_path_);
    if (manifest) {
      manifest << "record_version: 2\n";
      manifest << "replay_scope: post_detection\n";
      manifest << "frame_alignment: frame_id_and_observed_time_abs\n";
      manifest << "trace: " << std::quoted(trace_path_.filename().string()) << '\n';
      manifest << "pipeline_trace: " << std::quoted(pipeline_path_.filename().string()) << '\n';
      manifest << "fanblades: " << std::quoted(fanblade_path_.filename().string()) << '\n';
      manifest << "config_snapshot_available: " << (config_snapshot_available ? "true" : "false")
               << '\n';
      if (config_snapshot_available) {
        manifest << "config_snapshot: " << std::quoted(config_snapshot_path_.filename().string())
                 << '\n';
      }
      manifest << "source_config: " << std::quoted(config_source_path_.string()) << '\n';
    } else {
      tools::logger()->error(
        "[Buff ID Debug] Cannot create replay manifest {}", manifest_path_.string());
    }
    trace_ready_ = true;
    tools::logger()->info(
      "[Buff ID Debug] Continuous CSV enabled: {}, {}, {}, {}, {}, {}; replay manifest={}",
      trace_path_.string(), pipeline_path_.string(), candidate_path_.string(),
      fanblade_path_.string(), r_center_path_.string(), camera_path_.string(),
      manifest_path_.string());
  }

  void flush_trace_files()
  {
    trace_out_.flush();
    pipeline_out_.flush();
    candidate_out_.flush();
    fanblade_out_.flush();
    r_center_out_.flush();
    camera_out_.flush();
  }

  void mark_trace_row()
  {
    if (!trace_ready_) return;
    ++trace_row_count_;
    if (trace_row_count_ % 64 == 0) flush_trace_files();
  }

  void write_full_trace(const IdDiagnosticRecord & record)
  {
    if (!trace_ready_) return;
    write_base_record(trace_out_, record);
    trace_out_ << ',' << record.result_age_ms << ',' << record.detect_perf_stats.preprocess_dt_ms
               << ',' << record.detect_perf_stats.infer_dt_ms << ','
               << record.detect_perf_stats.postprocess_dt_ms << ',' << record.async_stale_drop_count
               << ',' << record.async_out_of_order_drop_count;
    write_quaternion(trace_out_, record.result_gimbal_q);
    trace_out_ << ',' << record.gimbal_state.yaw << ',' << record.gimbal_state.yaw_vel << ','
               << record.gimbal_state.pitch << ',' << record.gimbal_state.pitch_vel << ','
               << record.gimbal_state.bullet_speed << ',' << record.gimbal_state.bullet_count << ','
               << static_cast<int>(record.gimbal_state.camp);
    write_plan(trace_out_, record.plan);
    write_ballistic(trace_out_, record.ballistic);
    write_pose(trace_out_, record.pose);
    write_vector3(trace_out_, record.rune_xyz_in_world);
    write_vector3(trace_out_, record.rune_ypr_in_world);
    write_vector3(trace_out_, record.rune_ypd_in_world);
    write_vector3(trace_out_, record.blade_xyz_in_world);
    write_vector3(trace_out_, record.blade_ypd_in_world);
    write_matrix3(trace_out_, record.rotation_world);
    write_vector3(trace_out_, record.target_center_world);
    trace_out_ << ',' << record.target_label << ',' << record.target_leaf_id << ','
               << record.light_num << ',' << record.rune_last_observed_time;

    const auto & selection = record.target_selection_debug;
    trace_out_ << ',' << selection.raw_object_count << ',' << selection.valid_fanblade_count << ','
               << selection.has_previous_target;
    write_point(trace_out_, selection.previous_target_center);
    write_point(trace_out_, selection.previous_r_center);
    trace_out_ << ',' << selection.previous_target_image_angle << ','
               << static_cast<int>(selection.reason) << ',' << selection.selected_fanblade_index
               << ',' << selection.selected_source_index << ',' << selection.selected_image_angle
               << ',' << selection.selected_history_angle_residual;

    const auto & r_debug = record.r_center_debug;
    trace_out_ << ',' << static_cast<int>(r_debug.fallback_reason) << ',' << r_debug.refine_dt_ms
               << ',' << r_debug.roi_rect.x << ',' << r_debug.roi_rect.y << ','
               << r_debug.roi_rect.width << ',' << r_debug.roi_rect.height << ','
               << r_debug.selected_contour_index << ',' << r_debug.accepted_contour_count << ','
               << r_debug.best_score << ',' << r_debug.candidates.size();

    trace_out_ << ',' << record.has_estimator_debug;
    write_estimator_debug(trace_out_, record.estimator_debug);
    trace_out_ << '\n';
  }

  void write_candidate_trace(const IdDiagnosticRecord & record)
  {
    if (!trace_ready_) return;
    const auto & selection = record.target_selection_debug;
    for (const auto & candidate : selection.candidates) {
      candidate_out_ << record.event << ',' << record.frame_id << ',' << record.observed_time_abs
                     << ',' << selection.raw_object_count << ',' << selection.valid_fanblade_count
                     << ',' << selection.has_previous_target;
      write_point(candidate_out_, selection.previous_target_center);
      write_point(candidate_out_, selection.previous_r_center);
      candidate_out_ << ',' << selection.previous_target_image_angle << ','
                     << static_cast<int>(selection.reason) << ','
                     << selection.selected_fanblade_index << ',' << selection.selected_source_index
                     << ',' << selection.selected_image_angle << ','
                     << selection.selected_history_angle_residual << ',' << candidate.source_index
                     << ',' << candidate.fanblade_index << ',' << candidate.valid_keypoints << ','
                     << candidate.label << ',' << candidate.confidence << ',' << candidate.rect.x
                     << ',' << candidate.rect.y << ',' << candidate.rect.width << ','
                     << candidate.rect.height << ',' << candidate.keypoint_count;
      for (const auto & point : candidate.keypoints) write_point(candidate_out_, point);
      write_point(candidate_out_, candidate.center);
      candidate_out_ << ',' << candidate.image_angle << ',' << candidate.history_angle_residual
                     << ',' << candidate.image_center_distance << ','
                     << candidate.selected_as_target << '\n';
    }
  }

  void write_fanblade_trace(const IdDiagnosticRecord & record)
  {
    if (!trace_ready_) return;
    for (std::size_t slot = 0; slot < record.fanblades.size(); ++slot) {
      const auto & blade = record.fanblades[slot];
      fanblade_out_ << record.event << ',' << record.frame_id << ',' << record.observed_time_abs
                    << ',' << slot << ',' << static_cast<int>(blade.type) << ',' << blade.leaf_id
                    << ',' << blade.leaf_angle << ',' << blade.leaf_angle_valid << ','
                    << blade.angle << ',' << blade.width << ',' << blade.height << ','
                    << blade.label << ',' << blade.confidence;
      write_point(fanblade_out_, blade.center);
      fanblade_out_ << ',' << blade.points.size();
      for (std::size_t i = 0; i < 4; ++i) {
        const cv::Point2f point = i < blade.points.size() ? blade.points[i] : cv::Point2f{};
        write_point(fanblade_out_, point);
      }
      fanblade_out_ << '\n';
    }
  }

  void write_r_center_trace(const IdDiagnosticRecord & record)
  {
    if (!trace_ready_) return;
    const auto & debug = record.r_center_debug;
    const auto write_row = [&](const auto_buff::RCenterCandidateDebug * candidate) {
      r_center_out_ << record.event << ',' << record.frame_id << ',' << record.observed_time_abs
                    << ',' << debug.refined_valid << ',' << static_cast<int>(debug.fallback_reason)
                    << ',' << debug.refine_dt_ms << ',' << debug.roi_rect.x << ','
                    << debug.roi_rect.y << ',' << debug.roi_rect.width << ','
                    << debug.roi_rect.height << ',' << debug.selected_contour_index << ','
                    << debug.accepted_contour_count << ',' << debug.best_score << ','
                    << debug.candidates.size();
      if (!candidate) {
        r_center_out_ << ",-1,0,0,nan,nan,nan,nan,nan,0,0\n";
        return;
      }
      r_center_out_ << ',' << candidate->contour_index;
      write_point(r_center_out_, candidate->center_in_roi);
      r_center_out_ << ',' << candidate->area << ',' << candidate->aspect_ratio << ','
                    << candidate->child_area_ratio << ',' << candidate->offset_ratio << ','
                    << candidate->score << ',' << candidate->reject_mask << ','
                    << (candidate->contour_index == debug.selected_contour_index) << '\n';
    };
    if (debug.candidates.empty()) {
      write_row(nullptr);
      return;
    }
    for (const auto & candidate : debug.candidates) write_row(&candidate);
  }

  void dump(const std::string & event)
  {
    std::error_code error;
    std::filesystem::create_directories("records", error);
    if (error) {
      tools::logger()->error("[Buff ID Debug] Cannot create records/: {}", error.message());
      return;
    }

    const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    const auto path = std::filesystem::path(
      fmt::format("records/buff_id_diagnosis_{}_{}.csv", stamp, dump_count_++));
    std::ofstream out(path);
    if (!out) {
      tools::logger()->error("[Buff ID Debug] Cannot open {}", path.string());
      return;
    }

    out << std::fixed << std::setprecision(8);
    write_base_header(out);
    out << '\n';
    for (const auto & record : records_) write_record(out, record);
    tools::logger()->warn(
      "[Buff ID Debug] event={} samples={} snapshot={}", event, records_.size(), path.string());
  }

  bool enabled_ = false;
  bool trace_ready_ = false;
  bool has_previous_angle_ = false;
  double previous_angle_ = 0.0;
  double previous_angle_time_ = 0.0;
  double last_dump_time_ = -1e100;
  std::size_t trace_row_count_ = 0;
  int dump_count_ = 0;
  std::filesystem::path trace_path_;
  std::filesystem::path pipeline_path_;
  std::filesystem::path candidate_path_;
  std::filesystem::path fanblade_path_;
  std::filesystem::path r_center_path_;
  std::filesystem::path camera_path_;
  std::filesystem::path config_source_path_;
  std::filesystem::path config_snapshot_path_;
  std::filesystem::path manifest_path_;
  std::ofstream trace_out_;
  std::ofstream pipeline_out_;
  std::ofstream candidate_out_;
  std::ofstream fanblade_out_;
  std::ofstream r_center_out_;
  std::ofstream camera_out_;
  std::deque<IdDiagnosticRecord> records_;
};

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
  const Eigen::Matrix3d & rotation_world, const cv::Scalar & color)
{
  auto image_points = solver.reproject_buff(r_center_world, rotation_world);
  if (image_points.size() < 5) return image_points;
  tools::draw_points(
    image, std::vector<cv::Point2f>(image_points.begin(), image_points.begin() + 4), color, 2);
  tools::draw_points(
    image, std::vector<cv::Point2f>(image_points.begin() + 4, image_points.end()), color, 2);
  return image_points;
}

void draw_target_selection_overlay(
  cv::Mat & image, const auto_buff::BuffTargetSelectionDebug & debug)
{
  for (const auto & candidate : debug.candidates) {
    const cv::Scalar color =
      candidate.selected_as_target
        ? cv::Scalar(0, 0, 255)
        : (candidate.valid_keypoints ? cv::Scalar(255, 0, 255) : cv::Scalar(128, 128, 128));
    const cv::Rect rect(
      cvRound(candidate.rect.x), cvRound(candidate.rect.y),
      std::max(0, cvRound(candidate.rect.width)), std::max(0, cvRound(candidate.rect.height)));
    if (rect.width > 0 && rect.height > 0) {
      cv::rectangle(image, rect, color, candidate.selected_as_target ? 2 : 1);
    }

    const int point_count = std::clamp(candidate.keypoint_count, 0, 4);
    if (point_count > 0) {
      tools::draw_points(
        image,
        std::vector<cv::Point2f>(
          candidate.keypoints.begin(), candidate.keypoints.begin() + point_count),
        color, candidate.selected_as_target ? 3 : 1);
    }
    if (candidate.valid_keypoints) tools::draw_point(image, candidate.center, color, 3);
    tools::draw_text(
      image,
      fmt::format(
        "raw:{} {} {:.2f}", candidate.source_index,
        candidate.selected_as_target ? "selected" : "candidate", candidate.confidence),
      cv::Point(rect.x, rect.y - 4), color, 0.45, 1);
  }
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
    draw_reprojection(image, solver, candidate.r_center_world, candidate.rotation_world, color);
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
  if (!result.valid || result.predict_time_abs <= 0.0) return;

  const double predicted_angle = result.predicted_angle;
  if (!std::isfinite(predicted_angle)) return;
  // predicted_angle is the angle of the attack leaf at bullet arrival time.
  // predict(t, id) already includes the per-leaf offset (id * 2π/5), so
  // setting roll = predicted_angle + π/2 makes the model Z-axis point directly
  // at the attack leaf's predicted position — no inter-leaf subtraction needed.
  Eigen::Vector3d predicted_ypr = rune.ypr_in_world;
  predicted_ypr[2] = predicted_angle + CV_PI / 2.0;
  const auto predicted_rotation = tools::rotation_matrix(predicted_ypr);
  const auto image_points =
    draw_reprojection(image, solver, rune.xyz_in_world, predicted_rotation, {0, 0, 255});
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
  const bool record_raw_video = cli.get<bool>("record");
  const bool record_debug_video = cli.get<bool>("record-debug");
  const bool id_debug_enabled = cli.get<bool>("id-debug");

  tools::Plotter plotter;
  std::optional<tools::Recorder> recorder;
  if (record_raw_video || id_debug_enabled) {
    recorder.emplace();
    tools::logger()->info("Raw video recording enabled; output will be saved under records/.");
  }
  if (id_debug_enabled) {
    tools::logger()->info(
      "Buff ID diagnosis enabled; anomaly snapshots and both video streams will be saved under "
      "records/.");
  }
  std::optional<tools::Recorder> debug_recorder;
  if (record_debug_video || id_debug_enabled) {
    tools::RecorderOptions options;
    options.output_path = debug_record_prefix();
    debug_recorder.emplace(options);
    tools::logger()->info(
      "Result overlay recording enabled; output will be saved as {}.avi and {}.txt.",
      options.output_path, options.output_path);
  }
  tools::Exiter exiter;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto yaml = tools::load(config_path);
  const auto exposure_ms = yaml["buff_exposure"].as<double>();
  camera.set_exposure_ms(exposure_ms);

  tools::logger()->info("exposure:{}",exposure_ms);

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
  constexpr std::size_t kMaxFrameQs = 128;
  std::map<int, Eigen::Quaterniond> frame_qs;
  BuffPerfLogState perf_log_state;
  IdDiagnosisRecorder id_diagnosis(id_debug_enabled, config_path);
  auto fps_report_stamp = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    camera.read(img, t);
    auto img_time = t - exposure_center_offset;
    const Eigen::Quaterniond camera_q = gimbal.q(img_time);
    if (recorder) recorder->record(img.clone(), camera_q, img_time);
    if (!time_origin_ready) {
      time_origin = t;
      time_origin_ready = true;
    }

    const int submitted_frame = frame_count++;
    const bool submit_ok = detector.submit(img, submitted_frame, img_time);
    if (submit_ok) {
      frame_qs[submitted_frame] = camera_q;
      while (frame_qs.size() > kMaxFrameQs) {
        frame_qs.erase(frame_qs.begin());
      }
    }
    perf_log_state.record_submit(submit_ok);
    if (id_debug_enabled) {
      CameraDiagnosticRecord diagnostic;
      diagnostic.frame_id = submitted_frame;
      diagnostic.captured_time_abs = tools::delta_time(img_time, time_origin);
      diagnostic.submit_ok = submit_ok;
      diagnostic.image_width = img.cols;
      diagnostic.image_height = img.rows;
      diagnostic.gimbal_q = camera_q;
      diagnostic.async_stale_drop_count = detector.async_stale_drop_count();
      diagnostic.async_out_of_order_drop_count = detector.async_out_of_order_drop_count();
      id_diagnosis.capture_camera(diagnostic);
    }

    std::optional<auto_buff::PowerRune> power_runes;
    cv::Mat result_img;
    int result_frame_count = -1;
    double detect_dt_ms = 0.0;
    auto_buff::BuffDetectPerfStats detect_perf_stats;
    std::chrono::steady_clock::time_point result_t;
    auto_buff::RCenterRefineDebug r_center_debug;
    auto_buff::BuffTargetSelectionDebug target_selection_debug;
    while (detector.fetch(
      power_runes, result_img, result_frame_count, detect_dt_ms, true, &r_center_debug, &result_t,
      &detect_perf_stats, &target_selection_debug)) {
      perf_log_state.record_detect(detect_dt_ms, detect_perf_stats, !power_runes.has_value());

      auto detection_img = result_img.clone();
      const auto frame_q_it = frame_qs.find(result_frame_count);
      if (frame_q_it == frame_qs.end()) {
        tools::logger()->warn("[Buff] Missing gimbal quaternion for frame {}", result_frame_count);
        aimer.notify_observation_missed();
        continue;
      }

      const Eigen::Quaterniond q = frame_q_it->second;
      frame_qs.erase(frame_qs.begin(), std::next(frame_q_it));
      auto gs = gimbal.state();
      solver.set_R_gimbal2world(q);
      const double observed_time_abs = tools::delta_time(result_t, time_origin);
      solver.solve(power_runes, &result_t);
      const double now_time_abs = tools::delta_time(std::chrono::steady_clock::now(), time_origin);
      if (power_runes) power_runes->last_observed_time = observed_time_abs;

      auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
      if (power_runes) {
        if (power_runes->pnp_valid) perf_log_state.record_fit_input();
        plan = aimer.mpc_aim(*power_runes, observed_time_abs, now_time_abs, gs);
      } else {
        aimer.notify_observation_missed();
      }

    // tools::logger()->info("[gimbal send] yaw={:.3f} pitch={:.3f} yaw_vel={:.3f} pitch_vel={:.3f}",
    //   plan.yaw * 57.3, plan.pitch * 57.3,
    //   plan.yaw_vel * 57.3, plan.pitch_vel * 57.3);

      gimbal.send(
        plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
        plan.pitch_acc);

      if (id_debug_enabled) {
        IdDiagnosticRecord diagnostic;
        diagnostic.frame_id = result_frame_count;
        diagnostic.observed_time_abs = observed_time_abs;
        diagnostic.now_time_abs = now_time_abs;
        diagnostic.result_age_ms = std::max(0.0, (now_time_abs - observed_time_abs) * 1e3);
        diagnostic.detect_dt_ms = detect_dt_ms;
        diagnostic.detect_perf_stats = detect_perf_stats;
        diagnostic.async_stale_drop_count = detector.async_stale_drop_count();
        diagnostic.async_out_of_order_drop_count = detector.async_out_of_order_drop_count();
        diagnostic.result_gimbal_q = q;
        diagnostic.gimbal_state = gs;
        diagnostic.plan = plan;
        diagnostic.ballistic = aimer.last_ballistic_result();
        diagnostic.pose = aimer.pose_snapshot();
        diagnostic.command_guard = aimer.command_guard_debug();
        diagnostic.r_center_refined = r_center_debug.refined_valid;
        diagnostic.r_center_debug = r_center_debug;
        diagnostic.target_selection_debug = target_selection_debug;
        diagnostic.coarse_r_center = r_center_debug.coarse_r_center;
        diagnostic.refined_r_center = r_center_debug.refined_r_center;
        diagnostic.has_estimator_debug = true;
        diagnostic.estimator_debug = aimer.prediction_debug_snapshot();
        diagnostic.id_match = diagnostic.estimator_debug.id_match;
        diagnostic.predictor_backend =
          static_cast<int>(diagnostic.estimator_debug.summary.predictor_backend);
        if (power_runes) {
          const auto & rune = *power_runes;
          diagnostic.has_rune = true;
          diagnostic.pnp_valid = rune.pnp_valid;
          diagnostic.r_center = rune.r_center;
          diagnostic.light_num = rune.light_num;
          diagnostic.rune_last_observed_time = rune.last_observed_time;
          diagnostic.physical_angle = rune.physical_angle;
          diagnostic.reprojection_error = rune.reprojection_error;
          diagnostic.pnp_debug = rune.pnp_debug;
          diagnostic.target_leaf_id = rune.target_leaf_id;
          diagnostic.fanblades = rune.fanblades;
          diagnostic.rune_xyz_in_world = rune.xyz_in_world;
          diagnostic.rune_ypr_in_world = rune.ypr_in_world;
          diagnostic.rune_ypd_in_world = rune.ypd_in_world;
          diagnostic.blade_xyz_in_world = rune.blade_xyz_in_world;
          diagnostic.blade_ypd_in_world = rune.blade_ypd_in_world;
          diagnostic.rotation_world = rune.rotation_world;
          diagnostic.target_center_world = rune.target_center_world;
          if (!rune.fanblades.empty()) {
            const auto & target = rune.target();
            diagnostic.target_confidence = target.confidence;
            diagnostic.target_label = target.label;
            diagnostic.target_center = target.center;
            diagnostic.target_point_count = std::min(static_cast<int>(target.points.size()), 4);
            for (int i = 0; i < diagnostic.target_point_count; ++i) {
              diagnostic.target_points[static_cast<std::size_t>(i)] =
                target.points[static_cast<std::size_t>(i)];
            }
          }
          if (rune.pnp_valid) {
            diagnostic.pose_debug = aimer.pose_debug_snapshot();
            diagnostic.id_match = aimer.id_match_debug();
            diagnostic.has_id_match = diagnostic.id_match.attempted;
            diagnostic.has_model_prediction = diagnostic.id_match.model_available;
            if (diagnostic.estimator_debug.summary.fitted) {
              diagnostic.predicted_angle_at_observation = aimer.prediction_angle(observed_time_abs);
              diagnostic.predicted_speed_at_observation = aimer.prediction_speed(observed_time_abs);
            }
            if (
              diagnostic.ballistic.valid && std::isfinite(diagnostic.ballistic.predicted_angle) &&
              diagnostic.ballistic.predict_time_abs > 0.0) {
              diagnostic.predicted_speed_at_ballistic_time =
                aimer.prediction_speed(diagnostic.ballistic.predict_time_abs);
              if (
                diagnostic.ballistic.observed_leaf_id >= 0 &&
                diagnostic.ballistic.attack_leaf_id >= 0) {
                diagnostic.observed_zero_angle =
                  zero_angle(rune.physical_angle, diagnostic.ballistic.observed_leaf_id);
                diagnostic.predicted_zero_angle = zero_angle(
                  diagnostic.ballistic.predicted_angle, diagnostic.ballistic.attack_leaf_id);
                diagnostic.zero_phase_residual =
                  wrap_angle(diagnostic.predicted_zero_angle - diagnostic.observed_zero_angle);
              }
            }
            if (diagnostic.has_model_prediction) {
              diagnostic.model_expected_angles = diagnostic.id_match.model_expected_angles;
              diagnostic.model_residuals = diagnostic.id_match.model_residuals;
            }
          }
        }
        id_diagnosis.capture(diagnostic);
      }

      nlohmann::json data;
      data["detect_dt_ms"] = detect_dt_ms;
      data["filtered_pose_valid"] = 0;
      data["used_filtered_pose"] = 0;
      data["id_model_prior_status"] =
        static_cast<int>(auto_buff::AngleEstimate::IdModelPriorStatus::DISABLED);
      data["id_model_available"] = 0;
      data["id_model_used"] = 0;
      data["id_model_overrode_kinematic"] = 0;
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

          const auto & id_match = aimer.id_match_debug();
          data["id_model_prior_status"] = static_cast<int>(id_match.model_prior_status);
          data["id_model_available"] = id_match.model_available ? 1 : 0;
          data["id_model_used"] = id_match.model_used ? 1 : 0;
          data["id_model_overrode_kinematic"] = id_match.model_overrode_kinematic ? 1 : 0;
          data["id_kinematic_leaf_id"] = id_match.kinematic_leaf_id;
          data["id_model_leaf_id"] = id_match.model_leaf_id;
          data["id_model_age_ms"] = id_match.model_age_sec * 1e3;

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

      data["gimbal_yaw"] = gs.yaw * 57.3;
      data["gimbal_pitch"] = gs.pitch * 57.3;
      data["gimbal_yaw_vel"] = gs.yaw_vel * 57.3;
      data["gimbal_pitch_vel"] = gs.pitch_vel * 57.3;
      if (plan.control) {
        data["plan_yaw"] = plan.yaw * 57.3;
        data["plan_pitch"] = plan.pitch * 57.3;
        data["plan_yaw_vel"] = plan.yaw_vel * 57.3;
        data["plan_pitch_vel"] = plan.pitch_vel * 57.3;
        data["plan_yaw_acc"] = plan.yaw_acc * 57.3;
        data["plan_pitch_acc"] = plan.pitch_acc * 57.3;
        data["shoot"] = plan.fire ? 1 : 0;
      }
      plotter.plot(data);

      auto_buff::show_r_center_debug_views(
        *r_center_view, detection_img, r_center_debug, "buff refine");
      draw_buff_detection(detection_img, power_runes, result_frame_count, r_center_debug);
      if (kShowDetectorView) {
        maybe_show("buff detection", detection_img);
      }
      if (debug_recorder) {
        auto debug_img = result_img.clone();
        draw_buff_detection(debug_img, power_runes, result_frame_count, r_center_debug);
        draw_target_selection_overlay(debug_img, target_selection_debug);
        debug_recorder->record(debug_img, q, result_t);
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
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  return 0;
}
