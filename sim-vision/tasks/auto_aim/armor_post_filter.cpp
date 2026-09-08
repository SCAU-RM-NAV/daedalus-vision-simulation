#include "armor_post_filter.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace auto_aim
{
namespace
{
constexpr double kDegToRad = CV_PI / 180.0;

template <typename T>
T read_or(const YAML::Node & yaml, const char * key, const T & fallback)
{
  return yaml[key] ? yaml[key].as<T>() : fallback;
}

double normalize_angle(double angle)
{
  while (angle > CV_PI) angle -= 2.0 * CV_PI;
  while (angle < -CV_PI) angle += 2.0 * CV_PI;
  return angle;
}

bool finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

bool point_in_image(const cv::Point & point, const cv::Mat & img)
{
  return point.x >= 0 && point.y >= 0 && point.x < img.cols && point.y < img.rows;
}

}  // namespace

ArmorPostFilter::ArmorPostFilter(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);

  enabled_ = read_or<bool>(yaml, "fast_yolo_filter", true);
  geometry_enabled_ = read_or<bool>(yaml, "fast_yolo_geometry_filter", enabled_);
  color_enabled_ = read_or<bool>(yaml, "fast_yolo_color_filter", enabled_);

  min_armor_ratio_ = read_or<double>(yaml, "min_armor_ratio", 1.0);
  max_armor_ratio_ = read_or<double>(yaml, "max_armor_ratio", 5.0);
  max_rectangular_error_ =
    read_or<double>(yaml, "max_rectangular_error", 25.0) * kDegToRad;
  min_light_length_ = read_or<double>(yaml, "min_lightbar_length", 8.0);

  min_light_ratio_ = read_or<double>(yaml, "fast_yolo_min_light_ratio", 0.45);
  max_center_angle_ = read_or<double>(yaml, "max_angle_error", 45.0) * kDegToRad;
  max_light_angle_diff_ =
    read_or<double>(yaml, "fast_yolo_max_light_angle_diff", 18.0) * kDegToRad;

  color_dominance_ = read_or<double>(yaml, "fast_yolo_color_dominance", 1.05);
  color_min_ratio_ = read_or<double>(yaml, "fast_yolo_color_min_ratio", 0.35);
  color_sample_stride_ = std::max(1, read_or<int>(yaml, "fast_yolo_color_stride", 2));
  color_min_samples_ = std::max(1, read_or<int>(yaml, "fast_yolo_color_min_samples", 4));
}

bool ArmorPostFilter::keep(const Armor & armor, const cv::Mat & bgr_img) const
{
  if (!enabled_) return true;
  return check_geometry(armor) && check_color(armor, bgr_img);
}

bool ArmorPostFilter::check_geometry(const Armor & armor) const
{
  if (!enabled_ || !geometry_enabled_) return true;
  if (armor.points.size() != 4) return false;

  for (const auto & point : armor.points) {
    if (!finite_point(point)) return false;
  }

  const auto left = armor.points[3] - armor.points[0];
  const auto right = armor.points[2] - armor.points[1];
  const double left_length = cv::norm(left);
  const double right_length = cv::norm(right);
  if (left_length < min_light_length_ || right_length < min_light_length_) return false;

  const double shorter = std::min(left_length, right_length);
  const double longer = std::max(left_length, right_length);
  const double light_ratio = shorter / std::max(longer, std::numeric_limits<double>::epsilon());
  if (light_ratio < min_light_ratio_) return false;

  if (
    !std::isfinite(armor.ratio) || armor.ratio < min_armor_ratio_ ||
    armor.ratio > max_armor_ratio_) {
    return false;
  }

  if (
    !std::isfinite(armor.rectangular_error) ||
    armor.rectangular_error > max_rectangular_error_) {
    return false;
  }

  const auto left_center = (armor.points[0] + armor.points[3]) * 0.5f;
  const auto right_center = (armor.points[1] + armor.points[2]) * 0.5f;
  const auto center_delta = right_center - left_center;
  const double center_angle = std::atan2(std::abs(center_delta.y), std::abs(center_delta.x));
  if (!std::isfinite(center_angle) || center_angle > max_center_angle_) return false;

  const double left_angle = std::atan2(left.y, left.x);
  const double right_angle = std::atan2(right.y, right.x);
  const double light_angle_diff = std::abs(normalize_angle(left_angle - right_angle));
  return light_angle_diff < max_light_angle_diff_;
}

bool ArmorPostFilter::check_color(const Armor & armor, const cv::Mat & bgr_img) const
{
  if (!enabled_ || !color_enabled_) return true;
  if (armor.color != Color::red && armor.color != Color::blue) return true;
  if (armor.points.size() != 4 || bgr_img.empty() || bgr_img.type() != CV_8UC3) return true;

  const bool left_ok = line_matches_color(armor.points[0], armor.points[3], bgr_img, armor.color);
  const bool right_ok = line_matches_color(armor.points[1], armor.points[2], bgr_img, armor.color);

  return left_ok || right_ok;
}

bool ArmorPostFilter::line_matches_color(
  const cv::Point2f & start, const cv::Point2f & end, const cv::Mat & bgr_img,
  Color expected_color) const
{
  if (!finite_point(start) || !finite_point(end)) return false;

  const auto delta = end - start;
  const int steps = static_cast<int>(std::ceil(std::max(std::abs(delta.x), std::abs(delta.y))));
  if (steps <= 0) return false;

  int samples = 0;
  int dominant_samples = 0;
  double red_sum = 0.0;
  double blue_sum = 0.0;

  for (int step = 0; step <= steps; step += color_sample_stride_) {
    const float t = static_cast<float>(step) / static_cast<float>(steps);
    const cv::Point point(cvRound(start.x + delta.x * t), cvRound(start.y + delta.y * t));
    if (!point_in_image(point, bgr_img)) continue;

    const auto pixel = bgr_img.at<cv::Vec3b>(point);
    const double blue_value = pixel[0];
    const double red_value = pixel[2];
    blue_sum += blue_value;
    red_sum += red_value;
    samples += 1;

    if (expected_color == Color::red) {
      dominant_samples += red_value >= blue_value * color_dominance_ ? 1 : 0;
    } else if (expected_color == Color::blue) {
      dominant_samples += blue_value >= red_value * color_dominance_ ? 1 : 0;
    }
  }

  if (samples < color_min_samples_) return false;

  const bool sum_ok = expected_color == Color::red ? red_sum >= blue_sum * color_dominance_
                                                   : blue_sum >= red_sum * color_dominance_;
  const bool ratio_ok =
    static_cast<double>(dominant_samples) >= static_cast<double>(samples) * color_min_ratio_;
  return sum_ok && ratio_ok;
}

}  // namespace auto_aim
