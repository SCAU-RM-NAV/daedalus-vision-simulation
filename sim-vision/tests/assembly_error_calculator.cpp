// ./build/assembly_error_calculator configs/standard3.yaml
// ./build/assembly_error_calculator configs/hero_remote.yaml --camera=idle
// ./build/assembly_error_calculator configs/hero_remote.yaml --camera=combat

#include <Eigen/Dense>
#include <fmt/core.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <limits>
#include <list>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? | | output command line help }"
  "{all-modes     | false | run detector outside AUTO_AIM too }"
  "{pitch-search-deg | 8.0 | pitch search half range, degree }"
  "{pitch-step-deg | 0.02 | pitch search step, degree }"
  "{min-samples | 30 | min samples before pitch search }"
  "{max-samples | 600 | max recent samples kept for pitch search }"
  "{report-interval | 1.0 | pitch search report interval, second }"
  "{min-confidence | 0.60 | minimum detector confidence accepted }"
  "{max-reprojection-error | 3.0 | maximum mean PnP reprojection error, pixel }"
  "{trim-fraction | 0.10 | fraction of largest height residuals ignored }"
  "{min-depth-span | 0.80 | minimum camera-depth span for a reliable result, meter }"
  "{stable-reports | 3 | consecutive stable reports required before output }"
  "{max-stable-delta-deg | 0.10 | maximum pitch change between stable reports, degree }"
  "{camera | root | camera config: root, combat, or idle }"
  "{@config-path   | | yaml config path }";

namespace
{
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m

const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

Eigen::Matrix3d matrix3_from_yaml(const YAML::Node & yaml, const std::string & key)
{
  auto data = tools::read<std::vector<double>>(yaml, key);
  if (data.size() != 9) {
    tools::logger()->error("[AssemblyError] {} must contain 9 values.", key);
    std::exit(1);
  }

  return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(data.data());
}

Eigen::Vector3d vector3_from_yaml(const YAML::Node & yaml, const std::string & key)
{
  auto data = tools::read<std::vector<double>>(yaml, key);
  if (data.size() != 3) {
    tools::logger()->error("[AssemblyError] {} must contain 3 values.", key);
    std::exit(1);
  }

  return Eigen::Vector3d(data[0], data[1], data[2]);
}

Eigen::Matrix3d ideal_camera2gimbal()
{
  Eigen::Matrix3d R_ideal;
  R_ideal << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  return R_ideal;
}

Eigen::Matrix3d rpy_to_camera2gimbal(double roll, double pitch, double yaw)
{
  const Eigen::Matrix3d R_bias =
    (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
     Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
     Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();

  return R_bias * ideal_camera2gimbal();
}

std::vector<double> read_rpy_vector(const YAML::Node & yaml, const std::string & key)
{
  auto rpy = yaml[key].as<std::vector<double>>();
  if (rpy.size() != 3) {
    tools::logger()->error("[AssemblyError] {} must contain 3 values.", key);
    std::exit(1);
  }
  return rpy;
}

Eigen::Vector3d bias_rpy_from_camera2gimbal(const Eigen::Matrix3d & R_camera2gimbal)
{
  const Eigen::Matrix3d R_bias = R_camera2gimbal * ideal_camera2gimbal().transpose();

  const double pitch = std::asin(std::clamp(-R_bias(2, 0), -1.0, 1.0));
  const double cos_pitch = std::cos(pitch);

  double roll = 0.0;
  double yaw = 0.0;
  if (std::abs(cos_pitch) > 1e-9) {
    roll = std::atan2(R_bias(2, 1), R_bias(2, 2));
    yaw = std::atan2(R_bias(1, 0), R_bias(0, 0));
  } else {
    yaw = std::atan2(-R_bias(0, 1), R_bias(1, 1));
  }

  return {roll, pitch, yaw};
}

struct Camera2GimbalConfig
{
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
};

Camera2GimbalConfig read_camera2gimbal_config(const YAML::Node & yaml)
{
  if (yaml["camera2gimbal_rpy_deg"]) {
    auto rpy = read_rpy_vector(yaml, "camera2gimbal_rpy_deg");
    for (auto & value : rpy) value *= CV_PI / 180.0;
    tools::logger()->info("[AssemblyError] Use camera2gimbal_rpy_deg: roll pitch yaw.");
    return {rpy[0], rpy[1], rpy[2], rpy_to_camera2gimbal(rpy[0], rpy[1], rpy[2])};
  }

  if (yaml["camera2gimbal_rpy"]) {
    auto rpy = read_rpy_vector(yaml, "camera2gimbal_rpy");
    tools::logger()->info("[AssemblyError] Use camera2gimbal_rpy: roll pitch yaw, rad.");
    return {rpy[0], rpy[1], rpy[2], rpy_to_camera2gimbal(rpy[0], rpy[1], rpy[2])};
  }

  tools::logger()->info("[AssemblyError] Use R_camera2gimbal.");
  const Eigen::Matrix3d R = matrix3_from_yaml(yaml, "R_camera2gimbal");
  const Eigen::Vector3d rpy = bias_rpy_from_camera2gimbal(R);
  return {rpy.x(), rpy.y(), rpy.z(), R};
}

struct HeightSample
{
  Eigen::Vector3d xyz_in_camera;
  Eigen::Matrix3d R_gimbal2world = Eigen::Matrix3d::Identity();
  double confidence = 0.0;
  double reprojection_error = 0.0;
  int frame_count = 0;
};

struct PitchSearchResult
{
  bool valid = false;
  double pitch = 0.0;
  double pitch_delta = 0.0;
  double mean_z = 0.0;
  double variance_z = 0.0;
  double stddev_z = 0.0;
  double baseline_stddev_z = 0.0;
  double min_distance = 0.0;
  double max_distance = 0.0;
  double min_depth = 0.0;
  double max_depth = 0.0;
  std::size_t used_samples = 0;
  bool at_search_boundary = false;
  bool reliable = false;
  int stable_reports = 0;
  bool ready_to_apply = false;
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
};

struct ArmorIdentity
{
  auto_aim::Color color = auto_aim::Color::red;
  auto_aim::ArmorName name = auto_aim::ArmorName::not_armor;
  auto_aim::ArmorType type = auto_aim::ArmorType::small;

  bool matches(const auto_aim::Armor & armor) const
  {
    return color == armor.color && name == armor.name && type == armor.type;
  }
};

std::string format_matrix(const Eigen::Matrix3d & R)
{
  return fmt::format(
    "[[{:.9f}, {:.9f}, {:.9f}],\n"
    " [{:.9f}, {:.9f}, {:.9f}],\n"
    " [{:.9f}, {:.9f}, {:.9f}]]",
    R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1),
    R(2, 2));
}

std::string format_yaml_flow(const Eigen::Matrix3d & R)
{
  return fmt::format(
    "[{:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}, {:.9f}]",
    R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1),
    R(2, 2));
}

class PnPHeightCalculator
{
public:
  PnPHeightCalculator(
    const YAML::Node & root_yaml, const YAML::Node & camera_yaml,
    const std::string & camera_name)
  {
    const bool has_camera_intrinsics =
      camera_yaml["camera_matrix"] && camera_yaml["distort_coeffs"];
    const YAML::Node intrinsics_yaml = has_camera_intrinsics ? camera_yaml : root_yaml;
    const bool has_camera_extrinsics =
      camera_yaml["camera2gimbal_rpy_deg"] || camera_yaml["camera2gimbal_rpy"] ||
      camera_yaml["R_camera2gimbal"];
    const YAML::Node rotation_yaml = has_camera_extrinsics ? camera_yaml : root_yaml;
    const YAML::Node translation_yaml =
      camera_yaml["t_camera2gimbal"] ? camera_yaml : root_yaml;

    const auto camera_matrix_data =
      tools::read<std::vector<double>>(intrinsics_yaml, "camera_matrix");
    const auto distort_coeffs_data =
      tools::read<std::vector<double>>(intrinsics_yaml, "distort_coeffs");
    if (camera_matrix_data.size() != 9) {
      tools::logger()->error("[AssemblyError] camera_matrix must contain 9 values.");
      std::exit(1);
    }
    if (distort_coeffs_data.size() != 5) {
      tools::logger()->error("[AssemblyError] distort_coeffs must contain 5 values.");
      std::exit(1);
    }

    const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
    const Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
    cv::eigen2cv(camera_matrix, camera_matrix_);
    cv::eigen2cv(distort_coeffs, distort_coeffs_);

    config_ = read_camera2gimbal_config(rotation_yaml);
    t_camera2gimbal_ = vector3_from_yaml(translation_yaml, "t_camera2gimbal");
    R_gimbal2imubody_ = matrix3_from_yaml(root_yaml, "R_gimbal2imubody");

    if (
      camera_name != "root" &&
      (!has_camera_intrinsics || !has_camera_extrinsics || !camera_yaml["t_camera2gimbal"])) {
      tools::logger()->warn(
        "[AssemblyError] {} camera calibration is incomplete; missing values fall back to the "
        "root calibration. A dual-camera system should calibrate each camera separately.",
        camera_name);
    }

    tools::logger()->info(
      "[AssemblyError] Base camera2gimbal rpy: roll {:.4f}, pitch {:.4f}, yaw {:.4f} deg",
      config_.roll * 180.0 / CV_PI, config_.pitch * 180.0 / CV_PI,
      config_.yaw * 180.0 / CV_PI);
  }

  bool solve(
    auto_aim::Armor & armor, Eigen::Vector3d & xyz_in_camera,
    double & reprojection_error) const
  {
    const auto & object_points =
      (armor.type == auto_aim::ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

    if (armor.points.size() != object_points.size()) return false;

    cv::Vec3d rvec, tvec;
    const bool ok = cv::solvePnP(
      object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
      cv::SOLVEPNP_IPPE);
    if (!ok) return false;

    cv::cv2eigen(tvec, xyz_in_camera);
    if (!xyz_in_camera.allFinite() || xyz_in_camera.z() <= 1e-6) return false;

    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(
      object_points, rvec, tvec, camera_matrix_, distort_coeffs_, reprojected);
    reprojection_error = 0.0;
    for (std::size_t i = 0; i < reprojected.size(); i++) {
      reprojection_error += cv::norm(reprojected[i] - armor.points[i]);
    }
    reprojection_error /= static_cast<double>(reprojected.size());
    if (!std::isfinite(reprojection_error)) return false;

    armor.xyz_in_gimbal = transform_to_gimbal(xyz_in_camera, config_.pitch);
    return true;
  }

  Eigen::Matrix3d camera2gimbal_with_pitch(double pitch) const
  {
    return rpy_to_camera2gimbal(config_.roll, pitch, config_.yaw);
  }

  Eigen::Vector3d transform_to_gimbal(
    const Eigen::Vector3d & xyz_in_camera, double pitch) const
  {
    return camera2gimbal_with_pitch(pitch) * xyz_in_camera + t_camera2gimbal_;
  }

  Eigen::Vector3d transform_to_world(const HeightSample & sample, double pitch) const
  {
    return sample.R_gimbal2world * transform_to_gimbal(sample.xyz_in_camera, pitch);
  }

  std::optional<Eigen::Matrix3d> gimbal2world(const Eigen::Quaterniond & q) const
  {
    if (!q.coeffs().allFinite() || q.norm() < 1e-6) return std::nullopt;
    const Eigen::Quaterniond normalized_q = q.normalized();
    const Eigen::Matrix3d R =
      R_gimbal2imubody_.transpose() * normalized_q.toRotationMatrix() * R_gimbal2imubody_;
    return R;
  }

  const Camera2GimbalConfig & config() const { return config_; }

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Camera2GimbalConfig config_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
};

struct RobustHeightStats
{
  double mean = 0.0;
  double variance = std::numeric_limits<double>::max();
  std::size_t used_samples = 0;
};

RobustHeightStats robust_height_stats(std::vector<double> heights, double trim_fraction)
{
  RobustHeightStats stats;
  if (heights.size() < 2) return stats;

  std::sort(heights.begin(), heights.end());
  const std::size_t middle = heights.size() / 2;
  const double median = heights.size() % 2 == 0
                          ? 0.5 * (heights[middle - 1] + heights[middle])
                          : heights[middle];

  std::vector<std::pair<double, double>> residual_and_height;
  residual_and_height.reserve(heights.size());
  for (const double height : heights) {
    residual_and_height.emplace_back(std::abs(height - median), height);
  }
  std::sort(residual_and_height.begin(), residual_and_height.end());

  const auto requested_trim = static_cast<std::size_t>(
    std::floor(std::clamp(trim_fraction, 0.0, 0.45) * heights.size()));
  const std::size_t keep_count = std::max<std::size_t>(2, heights.size() - requested_trim);

  double sum = 0.0;
  for (std::size_t i = 0; i < keep_count; i++) sum += residual_and_height[i].second;
  stats.mean = sum / static_cast<double>(keep_count);

  double sum_squared_error = 0.0;
  for (std::size_t i = 0; i < keep_count; i++) {
    const double error = residual_and_height[i].second - stats.mean;
    sum_squared_error += error * error;
  }
  stats.variance = sum_squared_error / static_cast<double>(keep_count);
  stats.used_samples = keep_count;
  return stats;
}

PitchSearchResult search_best_pitch(
  const std::deque<HeightSample> & samples, const PnPHeightCalculator & calculator,
  double search_range_rad, double step_rad, double trim_fraction, double min_depth_span)
{
  PitchSearchResult best;
  if (samples.size() < 2) return best;

  step_rad = std::max(step_rad, 1e-6);
  const int steps = std::max(1, static_cast<int>(std::ceil(2.0 * search_range_rad / step_rad)));
  const double start_pitch = calculator.config().pitch - search_range_rad;
  const double actual_step = (2.0 * search_range_rad) / steps;

  best.variance_z = std::numeric_limits<double>::max();
  for (int i = 0; i <= steps; i++) {
    const double pitch = start_pitch + actual_step * i;

    std::vector<double> heights;
    heights.reserve(samples.size());
    for (const auto & sample : samples) {
      heights.push_back(calculator.transform_to_world(sample, pitch).z());
    }

    const auto stats = robust_height_stats(std::move(heights), trim_fraction);
    if (stats.variance < best.variance_z) {
      best.valid = true;
      best.pitch = pitch;
      best.pitch_delta = pitch - calculator.config().pitch;
      best.mean_z = stats.mean;
      best.variance_z = stats.variance;
      best.stddev_z = std::sqrt(stats.variance);
      best.used_samples = stats.used_samples;
      best.at_search_boundary = i == 0 || i == steps;
      best.R = calculator.camera2gimbal_with_pitch(pitch);
    }
  }

  std::vector<double> baseline_heights;
  baseline_heights.reserve(samples.size());
  for (const auto & sample : samples) {
    baseline_heights.push_back(
      calculator.transform_to_world(sample, calculator.config().pitch).z());
  }
  const auto baseline_stats = robust_height_stats(std::move(baseline_heights), trim_fraction);
  best.baseline_stddev_z = std::sqrt(baseline_stats.variance);

  best.min_distance = std::numeric_limits<double>::max();
  best.max_distance = 0.0;
  std::vector<double> depths;
  depths.reserve(samples.size());
  for (const auto & sample : samples) {
    const double distance = sample.xyz_in_camera.norm();
    best.min_distance = std::min(best.min_distance, distance);
    best.max_distance = std::max(best.max_distance, distance);
    depths.push_back(sample.xyz_in_camera.z());
  }
  std::sort(depths.begin(), depths.end());
  const std::size_t depth_margin = std::min(
    static_cast<std::size_t>(std::floor(0.05 * static_cast<double>(depths.size()))),
    (depths.size() - 1) / 2);
  best.min_depth = depths[depth_margin];
  best.max_depth = depths[depths.size() - 1 - depth_margin];
  best.reliable = best.valid && !best.at_search_boundary &&
                  best.max_depth - best.min_depth >= min_depth_span;

  return best;
}

void publish_pitch_search(
  tools::Plotter & plotter, const PitchSearchResult & result, std::size_t sample_count)
{
  if (!result.valid) return;

  nlohmann::json data;
  data["pitch_search_samples"] = sample_count;
  data["best_pitch_deg"] = result.pitch * 180.0 / CV_PI;
  data["pitch_error_delta_deg"] = result.pitch_delta * 180.0 / CV_PI;
  data["best_z_mean_m"] = result.mean_z;
  data["best_z_std_m"] = result.stddev_z;
  data["best_z_variance_m2"] = result.variance_z;
  data["base_z_std_m"] = result.baseline_stddev_z;
  data["pitch_search_used_samples"] = result.used_samples;
  data["pitch_search_reliable"] = result.reliable;
  data["pitch_search_at_boundary"] = result.at_search_boundary;
  data["pitch_search_stable_reports"] = result.stable_reports;
  data["pitch_search_ready_to_apply"] = result.ready_to_apply;
  data["sample_min_distance_m"] = result.min_distance;
  data["sample_max_distance_m"] = result.max_distance;
  data["sample_distance_span_m"] = result.max_distance - result.min_distance;
  data["sample_min_depth_m"] = result.min_depth;
  data["sample_max_depth_m"] = result.max_depth;
  data["sample_depth_span_m"] = result.max_depth - result.min_depth;
  plotter.plot(data);
}

void publish_height(
  tools::Plotter & plotter, const auto_aim::Armor & armor, int frame_count, int solved_count,
  double detect_dt, double reprojection_error)
{
  const auto & p_gimbal = armor.xyz_in_gimbal;
  const auto & p_world = armor.xyz_in_world;
  const double planar_distance = std::hypot(p_world.x(), p_world.y());
  const double distance = p_world.norm();
  const double height_diff = p_world.z();

  nlohmann::json data;
  data["frame"] = frame_count;
  data["armor_num"] = solved_count;
  data["height_diff_m"] = height_diff;
  data["height_diff_cm"] = height_diff * 100.0;
  data["distance_m"] = distance;
  data["planar_distance_m"] = planar_distance;
  data["world_x_m"] = p_world.x();
  data["world_y_m"] = p_world.y();
  data["world_z_m"] = p_world.z();
  data["gimbal_x_m"] = p_gimbal.x();
  data["gimbal_y_m"] = p_gimbal.y();
  data["gimbal_z_m"] = p_gimbal.z();
  data["confidence"] = armor.confidence;
  data["pnp_reprojection_error_px"] = reprojection_error;
  data["detect_ms"] = detect_dt * 1e3;
  plotter.plot(data);

  tools::logger()->info(
    "[AssemblyError] frame {} {} {} conf {:.2f}: world height {:+.3f} m ({:+.1f} cm), "
    "xyz_world [{:+.3f}, {:+.3f}, {:+.3f}] m, distance {:.3f} m, reproj {:.2f}px, "
    "detect {:.1f} ms",
    frame_count, auto_aim::COLORS.at(static_cast<std::size_t>(armor.color)),
    auto_aim::ARMOR_NAMES.at(static_cast<std::size_t>(armor.name)), armor.confidence, height_diff,
    height_diff * 100.0, p_world.x(), p_world.y(), p_world.z(), distance, reprojection_error,
    detect_dt * 1e3);
}

void show_frame(
  const cv::Mat & raw_img, const std::list<auto_aim::Armor> & armors, int frame_count,
  const std::string & camera_name, bool mode_skipped)
{
  if (raw_img.empty()) return;

  auto display = raw_img.clone();
  for (const auto & armor : armors) {
    std::vector<cv::Point> points;
    points.reserve(armor.points.size());
    for (const auto & point : armor.points) {
      points.emplace_back(cvRound(point.x), cvRound(point.y));
    }

    if (points.size() >= 4) {
      cv::polylines(display, points, true, {0, 255, 0}, 2);
    }
    const cv::Point center(cvRound(armor.center.x), cvRound(armor.center.y));
    cv::circle(display, center, 4, {0, 255, 0}, -1);

    const auto label = fmt::format(
      "{:.2f} {} {}", armor.confidence, auto_aim::COLORS.at(static_cast<std::size_t>(armor.color)),
      auto_aim::ARMOR_NAMES.at(static_cast<std::size_t>(armor.name)));
    cv::putText(
      display, label, center + cv::Point(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0},
      2);
  }

  const auto status = mode_skipped ? "waiting AUTO_AIM" : "detecting";
  cv::putText(
    display, fmt::format("{} [{}] frame {}", camera_name, status, frame_count), {16, 32},
    cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);

  cv::resize(display, display, {}, 0.5, 0.5);
  cv::imshow("assembly_error_calculator", display);
}

bool process_detection(
  tools::Plotter & plotter, PnPHeightCalculator & calculator, std::list<auto_aim::Armor> armors,
  const Eigen::Matrix3d & R_gimbal2world, int frame_count, double detect_dt,
  double min_confidence, double max_reprojection_error,
  std::optional<ArmorIdentity> & sample_identity, HeightSample & sample)
{
  int solved_count = 0;
  auto primary = armors.end();
  Eigen::Vector3d primary_xyz_in_camera = Eigen::Vector3d::Zero();
  double primary_reprojection_error = std::numeric_limits<double>::max();
  for (auto armor_it = armors.begin(); armor_it != armors.end(); ++armor_it) {
    if (armor_it->confidence < min_confidence) continue;
    if (sample_identity && !sample_identity->matches(*armor_it)) continue;

    Eigen::Vector3d xyz_in_camera;
    double reprojection_error = 0.0;
    if (!calculator.solve(*armor_it, xyz_in_camera, reprojection_error)) continue;
    if (reprojection_error > max_reprojection_error) continue;

    armor_it->xyz_in_world = R_gimbal2world * armor_it->xyz_in_gimbal;

    solved_count++;
    if (primary == armors.end() || armor_it->confidence > primary->confidence) {
      primary = armor_it;
      primary_xyz_in_camera = xyz_in_camera;
      primary_reprojection_error = reprojection_error;
    }
  }

  if (solved_count == 0) {
    nlohmann::json data;
    data["frame"] = frame_count;
    data["armor_num"] = 0;
    plotter.plot(data);
    return false;
  }

  if (primary != armors.end()) {
    if (!sample_identity) {
      sample_identity = ArmorIdentity{primary->color, primary->name, primary->type};
      tools::logger()->info(
        "[AssemblyError] Locked sample identity: {} {} {}. Restart to calibrate another target.",
        auto_aim::COLORS.at(static_cast<std::size_t>(primary->color)),
        auto_aim::ARMOR_NAMES.at(static_cast<std::size_t>(primary->name)),
        auto_aim::ARMOR_TYPES.at(static_cast<std::size_t>(primary->type)));
    }
    publish_height(
      plotter, *primary, frame_count, solved_count, detect_dt, primary_reprojection_error);
    sample.xyz_in_camera = primary_xyz_in_camera;
    sample.R_gimbal2world = R_gimbal2world;
    sample.confidence = primary->confidence;
    sample.reprojection_error = primary_reprojection_error;
    sample.frame_count = frame_count;
    return true;
  }

  return false;
}

}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const bool all_modes = cli.get<bool>("all-modes");
  const double pitch_search_range_rad =
    std::abs(cli.get<double>("pitch-search-deg")) * CV_PI / 180.0;
  const double pitch_step_rad =
    std::max(1e-6, std::abs(cli.get<double>("pitch-step-deg")) * CV_PI / 180.0);
  const int min_samples = std::max(2, cli.get<int>("min-samples"));
  const int max_samples = std::max(min_samples, cli.get<int>("max-samples"));
  const double report_interval = std::max(0.1, cli.get<double>("report-interval"));
  const double min_confidence = std::clamp(cli.get<double>("min-confidence"), 0.0, 1.0);
  const double max_reprojection_error =
    std::max(0.0, cli.get<double>("max-reprojection-error"));
  const double trim_fraction = std::clamp(cli.get<double>("trim-fraction"), 0.0, 0.45);
  const double min_depth_span = std::max(0.0, cli.get<double>("min-depth-span"));
  const int required_stable_reports = std::max(1, cli.get<int>("stable-reports"));
  const double max_stable_delta_rad =
    std::max(0.0, cli.get<double>("max-stable-delta-deg")) * CV_PI / 180.0;
  const auto camera_name = cli.get<std::string>("camera");

  tools::Exiter exiter;
  tools::Plotter plotter;

  const auto yaml = tools::load(config_path);
  YAML::Node camera_yaml;
  if (camera_name == "root") {
    camera_yaml = yaml;
  } else if (camera_name == "combat") {
    if (!yaml["combat_camera"]) {
      tools::logger()->error("[AssemblyError] combat_camera not found in config.");
      return 1;
    }
    camera_yaml = yaml["combat_camera"];
  } else if (camera_name == "idle") {
    if (!yaml["idle_camera"]) {
      tools::logger()->error("[AssemblyError] idle_camera not found in config.");
      return 1;
    }
    camera_yaml = yaml["idle_camera"];
  } else {
    tools::logger()->error("[AssemblyError] Unknown camera option: {}", camera_name);
    return 1;
  }

  const auto exposure_ms = tools::read<double>(camera_yaml, "exposure_ms");
  const auto exposure_center_offset =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  tools::logger()->info(
    "[AssemblyError] Image q timestamp offset: -{:.3f} ms (exposure center)",
    exposure_ms * 0.5);

  io::Gimbal gimbal(config_path);
  io::Camera camera(camera_yaml);
  auto_aim::YOLO yolo(config_path, true);
  PnPHeightCalculator calculator(yaml, camera_yaml, camera_name);
  tools::logger()->info(
    "[AssemblyError] Mode gate: {}", all_modes ? "all gimbal modes" : "AUTO_AIM only");
  tools::logger()->info(
    "[AssemblyError] Pitch search: range +/-{:.3f} deg, step {:.4f} deg, samples {}..{}",
    pitch_search_range_rad * 180.0 / CV_PI, pitch_step_rad * 180.0 / CV_PI, min_samples,
    max_samples);
  tools::logger()->info(
    "[AssemblyError] Sample gates: confidence >= {:.2f}, reprojection <= {:.2f}px, "
    "trim {:.0f}%, required depth span >= {:.2f}m, stable reports {} within {:.2f}deg",
    min_confidence, max_reprojection_error, trim_fraction * 100.0, min_depth_span,
    required_stable_reports, max_stable_delta_rad * 180.0 / CV_PI);
  tools::logger()->info(
    "[AssemblyError] Keep one physical armor at a constant world height and move it through "
    "near-far positions; do not collect from a rotating or vertically moving target.");

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  int frame_count = 0;
  auto last_mode = io::GimbalMode::IDLE;
  auto fps_report_stamp = std::chrono::steady_clock::now();
  auto pitch_report_stamp = std::chrono::steady_clock::now();
  double detect_time_sum = 0.0;
  int detect_frame_count = 0;
  std::deque<HeightSample> samples;
  std::optional<ArmorIdentity> sample_identity;
  std::optional<double> previous_reliable_pitch;
  int stable_report_count = 0;
  int last_stability_frame = -1;

  while (!exiter.exit()) {
    const auto mode = gimbal.mode();
    if (last_mode != mode) {
      tools::logger()->info("[AssemblyError] Switch to {}", gimbal.str(mode));
      last_mode = mode;
    }

    camera.read(img, t);
    const auto img_time = t - exposure_center_offset;
    if (img.empty()) break;

    if (!all_modes && mode != io::GimbalMode::AUTO_AIM) {
      show_frame(img, {}, frame_count, camera_name, true);
      if (cv::waitKey(1) == 'q') break;
      std::this_thread::sleep_for(10ms);
      continue;
    }

    const auto img_q = gimbal.q(img_time);
    const auto R_gimbal2world = calculator.gimbal2world(img_q);
    if (!R_gimbal2world) {
      tools::logger()->warn("[AssemblyError] Invalid image-time quaternion; sample dropped.");
      continue;
    }

    const auto frame_id = frame_count++;
    const auto detect_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_id);
    const auto detect_end = std::chrono::steady_clock::now();
    const double detect_dt = tools::delta_time(detect_end, detect_start);
    show_frame(img, armors, frame_id, camera_name, false);

    HeightSample sample;
    if (process_detection(
          plotter, calculator, std::move(armors), *R_gimbal2world, frame_id, detect_dt,
          min_confidence, max_reprojection_error, sample_identity, sample)) {
      samples.push_back(sample);
      while (static_cast<int>(samples.size()) > max_samples) samples.pop_front();
    }

    detect_time_sum += detect_dt;
    detect_frame_count++;
    const auto now = std::chrono::steady_clock::now();
    if (
      static_cast<int>(samples.size()) >= min_samples &&
      tools::delta_time(now, pitch_report_stamp) >= report_interval) {
      pitch_report_stamp = now;
      auto result = search_best_pitch(
        samples, calculator, pitch_search_range_rad, pitch_step_rad, trim_fraction,
        min_depth_span);
      if (result.valid) {
        const bool has_new_sample = samples.back().frame_count != last_stability_frame;
        if (result.reliable && has_new_sample) {
          if (
            previous_reliable_pitch &&
            std::abs(result.pitch - *previous_reliable_pitch) <= max_stable_delta_rad) {
            stable_report_count++;
          } else {
            stable_report_count = 1;
          }
          previous_reliable_pitch = result.pitch;
          last_stability_frame = samples.back().frame_count;
        } else if (!result.reliable) {
          previous_reliable_pitch.reset();
          stable_report_count = 0;
          last_stability_frame = -1;
        }
        result.stable_reports = stable_report_count;
        result.ready_to_apply =
          result.reliable && stable_report_count >= required_stable_reports;

        publish_pitch_search(plotter, result, samples.size());
        tools::logger()->info(
          "[AssemblyError] best pitch from {} samples: base {:.4f} deg, best {:.4f} deg, "
          "delta {:+.4f} deg, world-z std {:.4f}->{:.4f} m, depth {:.2f}-{:.2f} m",
          samples.size(), calculator.config().pitch * 180.0 / CV_PI,
          result.pitch * 180.0 / CV_PI, result.pitch_delta * 180.0 / CV_PI,
          result.baseline_stddev_z, result.stddev_z, result.min_depth, result.max_depth);

        if (result.ready_to_apply) {
          tools::logger()->info(
            "[AssemblyError] reliable best R_camera2gimbal =\n{}", format_matrix(result.R));
          tools::logger()->info(
            "[AssemblyError] yaml R_camera2gimbal: {}", format_yaml_flow(result.R));
        } else if (result.reliable) {
          tools::logger()->warn(
            "[AssemblyError] Candidate is geometrically valid but only stable for {}/{} reports; "
            "wait before applying it.",
            stable_report_count, required_stable_reports);
        } else if (result.at_search_boundary) {
          tools::logger()->warn(
            "[AssemblyError] Result reached the pitch search boundary. DO NOT apply it; check "
            "coordinate conventions, target identity, and the starting extrinsic.");
        } else {
          tools::logger()->warn(
            "[AssemblyError] Camera-depth span is only {:.2f} m (required {:.2f} m). DO NOT apply "
            "this result; move the same target through more near-far positions.",
            result.max_depth - result.min_depth, min_depth_span);
        }
      }
    }

    const auto report_dt = tools::delta_time(now, fps_report_stamp);
    if (report_dt >= 1.0 && detect_time_sum > 0.0) {
      tools::logger()->info(
        "[AssemblyError] detect avg: {:.2f} fps, loop: {:.2f} fps",
        detect_frame_count / detect_time_sum, detect_frame_count / report_dt);
      fps_report_stamp = now;
      detect_time_sum = 0.0;
      detect_frame_count = 0;
    }

    if (cv::waitKey(1) == 'q') break;
  }

  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  return 0;
}
