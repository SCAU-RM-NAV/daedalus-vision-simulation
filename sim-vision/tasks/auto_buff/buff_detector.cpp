#include "buff_detector.hpp"

#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "tools/logger.hpp"

namespace auto_buff
{
std::optional<RCenterDebugView> parse_r_center_debug_view(const std::string & value)
{
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (normalized == "off") return RCenterDebugView::Off;
  if (normalized == "roi") return RCenterDebugView::Roi;
  if (normalized == "binary") return RCenterDebugView::Binary;
  if (normalized == "contours") return RCenterDebugView::Contours;
  if (normalized == "all") return RCenterDebugView::All;
  return std::nullopt;
}

const char * r_center_fallback_reason_name(RCenterFallbackReason reason)
{
  switch (reason) {
    case RCenterFallbackReason::None:
      return "none";
    case RCenterFallbackReason::EmptyImage:
      return "empty_image";
    case RCenterFallbackReason::InvalidBladeRadius:
      return "invalid_blade_radius";
    case RCenterFallbackReason::EmptyRoi:
      return "empty_roi";
    case RCenterFallbackReason::NoContours:
      return "no_contours";
    case RCenterFallbackReason::NoAcceptedCandidate:
      return "no_accepted_candidate";
  }
  return "unknown";
}

std::string r_center_reject_mask_text(std::uint32_t reject_mask)
{
  if (reject_mask == R_CENTER_REJECT_NONE) return "accepted";

  struct RejectName
  {
    std::uint32_t mask;
    const char * name;
  };
  constexpr RejectName reject_names[] = {
    {R_CENTER_REJECT_SMALL_AREA, "area"},
    {R_CENTER_REJECT_NESTED, "nested"},
    {R_CENTER_REJECT_INVALID_PERIMETER, "perimeter"},
    {R_CENTER_REJECT_CHILD_AREA, "children"},
  };

  std::ostringstream text;
  bool first = true;
  for (const auto & reject_name : reject_names) {
    if ((reject_mask & reject_name.mask) == 0) continue;
    if (!first) text << ',';
    text << reject_name.name;
    first = false;
  }
  return text.str();
}

RCenterDebugImages make_r_center_debug_images(
  const cv::Mat & image, const RCenterRefineDebug & debug)
{
  RCenterDebugImages views;
  if (!image.empty() && debug.roi_rect.area() > 0) {
    const auto clipped = debug.roi_rect & cv::Rect(0, 0, image.cols, image.rows);
    if (clipped.area() > 0) views.roi = image(clipped).clone();
  }
  if (debug.binary_roi.empty()) return views;

  views.binary = debug.binary_roi.clone();
  cv::cvtColor(debug.binary_roi, views.contours, cv::COLOR_GRAY2BGR);
  for (std::size_t index = 0; index < debug.contours.size(); ++index) {
    const bool selected = static_cast<int>(index) == debug.selected_contour_index;
    const bool accepted = index < debug.candidates.size() &&
                          debug.candidates[index].reject_mask == R_CENTER_REJECT_NONE;
    const cv::Scalar color = selected   ? cv::Scalar(0, 0, 255)
                             : accepted ? cv::Scalar(0, 255, 0)
                                        : cv::Scalar(128, 128, 128);
    cv::drawContours(
      views.contours, debug.contours, static_cast<int>(index), color, selected ? 2 : 1);

    if (index >= debug.candidates.size()) continue;
    const auto & candidate = debug.candidates[index];
    const auto anchor = candidate.center_in_roi + cv::Point2f(3.0f, -3.0f);
    std::string label = cv::format(
      "#%zu s%.2f", index, candidate.score);
    if (candidate.reject_mask != R_CENTER_REJECT_NONE) {
      label += " " + r_center_reject_mask_text(candidate.reject_mask);
    }
    cv::putText(
      views.contours, label, anchor, cv::FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv::LINE_AA);
  }

  if (debug.selected_contour_index < 0) {
    const std::string label =
      std::string("fallback: ") + r_center_fallback_reason_name(debug.fallback_reason);
    cv::putText(
      views.contours, label, {4, 14}, cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 165, 255), 1,
      cv::LINE_AA);
  }
  return views;
}

void show_r_center_debug_views(
  RCenterDebugView view, const cv::Mat & image, const RCenterRefineDebug & debug,
  const std::string & window_prefix)
{
  if (view == RCenterDebugView::Off) return;
  const auto views = make_r_center_debug_images(image, debug);
  if ((view == RCenterDebugView::Roi || view == RCenterDebugView::All) && !views.roi.empty()) {
    cv::imshow(window_prefix + " roi", views.roi);
  }
  if (
    (view == RCenterDebugView::Binary || view == RCenterDebugView::All) && !views.binary.empty()) {
    cv::imshow(window_prefix + " binary", views.binary);
  }
  if (
    (view == RCenterDebugView::Contours || view == RCenterDebugView::All) &&
    !views.contours.empty()) {
    cv::imshow(window_prefix + " contours", views.contours);
  }
}

namespace
{
double image_angle(const cv::Point2f & point, const cv::Point2f & r_center)
{
  const auto v = point - r_center;
  return std::atan2(-v.y, v.x);
}

double abs_angle_diff(double lhs, double rhs)
{
  return std::abs(std::remainder(lhs - rhs, CV_2PI));
}

template <typename T>
T require_detector_scalar(
  const YAML::Node & detector, const std::string & key, const std::string & path)
{
  const YAML::Node node = detector[key];
  if (!node || !node.IsScalar()) {
    throw std::invalid_argument("[Buff_Detector config] '" + path + "' must be a scalar.");
  }

  try {
    return node.as<T>();
  } catch (const YAML::Exception &) {
    throw std::invalid_argument("[Buff_Detector config] '" + path + "' has an invalid value.");
  }
}

void validate_r_center_refine_config(const RCenterRefineConfig & config)
{
  if (config.threshold < 0 || config.threshold > 255) {
    throw std::invalid_argument(
      "[Buff_Detector config] 'buff_detector.r_center_refine_threshold' must be in [0, 255].");
  }
  if (!std::isfinite(config.radius_scale) || config.radius_scale <= 0.0) {
    throw std::invalid_argument(
      "[Buff_Detector config] 'buff_detector.r_center_refine_radius_scale' must be positive.");
  }
  if (config.kernel_size <= 0) {
    throw std::invalid_argument(
      "[Buff_Detector config] 'buff_detector.r_center_refine_kernel_size' must be positive.");
  }
  if (
    !std::isfinite(config.max_child_area_ratio) || config.max_child_area_ratio < 0.0 ||
    config.max_child_area_ratio > 1.0) {
    throw std::invalid_argument(
      "[Buff_Detector config] 'buff_detector.r_center_refine_max_child_area_ratio' must be in [0, 1].");
  }
}

std::vector<FanBlade> objects_to_fanblades(
  const std::vector<YOLO11_BUFF::Object> & results,
  BuffTargetSelectionDebug * target_selection_debug)
{
  std::vector<FanBlade> fanblades;
  fanblades.reserve(results.size());

  if (target_selection_debug) {
    *target_selection_debug = {};
    target_selection_debug->raw_object_count = static_cast<int>(results.size());
    target_selection_debug->candidates.reserve(results.size());
  }

  for (std::size_t source_index = 0; source_index < results.size(); ++source_index) {
    const auto & result = results[source_index];
    BuffCandidateDebug * candidate_debug = nullptr;
    if (target_selection_debug) {
      target_selection_debug->candidates.emplace_back();
      candidate_debug = &target_selection_debug->candidates.back();
      candidate_debug->source_index = static_cast<int>(source_index);
      candidate_debug->label = result.label;
      candidate_debug->confidence = result.prob;
      candidate_debug->rect = result.rect;
      candidate_debug->keypoint_count = static_cast<int>(result.kpt.size());
      for (std::size_t i = 0; i < std::min<std::size_t>(result.kpt.size(), 4); ++i) {
        candidate_debug->keypoints[i] = result.kpt[i];
      }
    }
    if (result.kpt.size() != 4) {
      tools::logger()->warn("[Buff_Detector] Invalid keypoint count: {}", result.kpt.size());
      continue;
    }

    fanblades.emplace_back(result.kpt, (result.kpt[1] + result.kpt[3]) * 0.5f, _light);
    fanblades.back().confidence = result.prob;
    fanblades.back().label = result.label;
    if (candidate_debug) {
      candidate_debug->valid_keypoints = true;
      candidate_debug->fanblade_index = static_cast<int>(fanblades.size()) - 1;
      candidate_debug->center = fanblades.back().center;
    }
  }

  if (target_selection_debug) {
    target_selection_debug->valid_fanblade_count = static_cast<int>(fanblades.size());
  }

  return fanblades;
}

std::optional<int> dominant_label(const std::vector<FanBlade> & fanblades)
{
  constexpr int kBuffClassNum = 2;
  std::array<double, kBuffClassNum> confidence_sum{};
  bool found = false;
  for (const auto & fanblade : fanblades) {
    if (fanblade.label < 0 || fanblade.label >= kBuffClassNum) continue;
    confidence_sum[static_cast<std::size_t>(fanblade.label)] += fanblade.confidence;
    found = true;
  }

  if (!found) return std::nullopt;
  return confidence_sum[0] >= confidence_sum[1] ? 0 : 1;
}

// 颜色过滤（R标重定位用）
cv::Mat build_r_center_color_mask(const cv::Mat & bgr_roi, int expected_label, int threshold_value)
{
  constexpr int kColorMargin = 20;

  std::vector<cv::Mat> channels;
  cv::split(bgr_roi, channels);  // BGR

  cv::Mat dominant_channel = expected_label == 0 ? channels[2] : channels[0];
  cv::Mat other_channel = expected_label == 0 ? channels[0] : channels[2];

  cv::Mat dominant_mask;
  cv::threshold(dominant_channel, dominant_mask, threshold_value, 255, cv::THRESH_BINARY);

  cv::Mat color_diff;
  cv::subtract(dominant_channel, other_channel, color_diff);

  cv::Mat green_suppressed;
  cv::subtract(dominant_channel, channels[1], green_suppressed);

  cv::Mat color_mask;
  cv::threshold(color_diff, color_mask, kColorMargin, 255, cv::THRESH_BINARY);

  cv::Mat green_mask;
  cv::threshold(green_suppressed, green_mask, kColorMargin, 255, cv::THRESH_BINARY);

  cv::bitwise_and(dominant_mask, color_mask, color_mask);
  cv::bitwise_and(color_mask, green_mask, color_mask);
  return color_mask;
}

cv::Point2f average_r_center(const std::vector<FanBlade> & fanblades)
{
  cv::Point2f sum(0.0f, 0.0f);
  for (const auto & fanblade : fanblades) sum += fanblade.points[0];
  return sum * (1.0f / static_cast<float>(fanblades.size()));
}

// 计算R标中心到扇叶中心的像素距离
double average_r_center_radius(
  const std::vector<FanBlade> & fanblades, const cv::Point2f & r_center)
{
  double radius_sum = 0.0;
  int radius_count = 0;
  for (const auto & fanblade : fanblades) {
    if (fanblade.points.size() <= 2) continue;
    radius_sum += cv::norm(fanblade.points[2] - r_center);
    radius_count++;
  }
  if (radius_count == 0) return 0.0;
  return radius_sum / radius_count;
}

// 计算中位数
double median(std::vector<double> values)
{
  if (values.empty()) return 0.0;
  const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
  std::nth_element(values.begin(), middle, values.end());
  if (values.size() % 2 != 0) return *middle;
  const auto lower = std::max_element(values.begin(), middle);
  return (*lower + *middle) * 0.5;
}

// 递归计算一个轮廓的所有子轮廓的总面积
double descendant_contour_area(
  const std::vector<std::vector<cv::Point>> & contours, const std::vector<cv::Vec4i> & hierarchy,
  int contour_index)
{
  if (contour_index < 0 || contour_index >= static_cast<int>(hierarchy.size())) return 0.0;

  double area_sum = 0.0;
  int child_index = hierarchy[static_cast<std::size_t>(contour_index)][2];
  while (child_index >= 0 && child_index < static_cast<int>(contours.size())) {
    area_sum += std::abs(cv::contourArea(contours[static_cast<std::size_t>(child_index)]));
    area_sum += descendant_contour_area(contours, hierarchy, child_index);
    child_index = hierarchy[static_cast<std::size_t>(child_index)][0];
  }
  return area_sum;
}

// 计算半径中位数绝对偏差比
double radius_mad_ratio(
  const std::vector<FanBlade> & fanblades, const cv::Point2f & candidate_center)
{
  if (fanblades.size() < 2) return 0.0;

  std::vector<double> radii;
  radii.reserve(fanblades.size());
  for (const auto & fanblade : fanblades)
    radii.push_back(cv::norm(fanblade.center - candidate_center));

  const double median_radius = median(radii);
  if (median_radius <= 1e-6) return std::numeric_limits<double>::infinity();

  std::vector<double> deviations;
  deviations.reserve(radii.size());
  for (const double radius : radii) deviations.push_back(std::abs(radius - median_radius));
  return median(std::move(deviations)) / median_radius;
}
}  // namespace

cv::Point2f refine_r_center(
  const std::vector<FanBlade> & fanblades, const cv::Mat & bgr_img,
  const cv::Point2f & coarse_r_center, const RCenterRefineConfig & config,
  RCenterRefineDebug * debug)
{
  const auto start_time = std::chrono::steady_clock::now();
  const auto finish_debug =
    [&](bool refined_valid, RCenterFallbackReason fallback_reason, int selected_index = -1) {
      const auto end_time = std::chrono::steady_clock::now();
      const auto refine_dt_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
      if (debug) {
        debug->refine_dt_ms = refine_dt_ms;
        debug->refined_valid = refined_valid;
        debug->selected_contour_index = selected_index;
        debug->fallback_reason = fallback_reason;
      }
      // tools::logger()->debug(
      //   "[Buff_Detector] R center refine: {:.3f} ms, valid: {}, selected: {}", refine_dt_ms,
      //   refined_valid, selected_index);
    };

  if (debug) {
    *debug = {};
    debug->coarse_r_center = coarse_r_center;
    debug->refined_r_center = coarse_r_center;
  }

  if (bgr_img.empty()) {
    finish_debug(false, RCenterFallbackReason::EmptyImage);
    return coarse_r_center;
  }

  // 定位范围
  const double blade_radius = average_r_center_radius(fanblades, coarse_r_center);
  if (blade_radius <= 1e-6) {
    finish_debug(false, RCenterFallbackReason::InvalidBladeRadius);
    return coarse_r_center;
  }

  const double roi_radius = std::max(8.0, blade_radius * config.radius_scale);
  cv::Rect roi_rect(
    static_cast<int>(std::floor(coarse_r_center.x - roi_radius)),
    static_cast<int>(std::floor(coarse_r_center.y - roi_radius)),
    static_cast<int>(std::ceil(roi_radius * 2.0)), static_cast<int>(std::ceil(roi_radius * 2.0)));
  roi_rect &= cv::Rect(0, 0, bgr_img.cols, bgr_img.rows);
  if (roi_rect.empty()) {
    finish_debug(false, RCenterFallbackReason::EmptyRoi);
    return coarse_r_center;
  }

  // 二值化
  cv::Mat binary_roi;
  const auto expected_label = dominant_label(fanblades);
  if (expected_label.has_value()) {
    binary_roi = build_r_center_color_mask(bgr_img(roi_rect), *expected_label, config.threshold);
  } else {
    cv::Mat gray_roi;
    cv::cvtColor(bgr_img(roi_rect), gray_roi, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_roi, binary_roi, config.threshold, 255, cv::THRESH_BINARY);
  }
  if (config.kernel_size > 1) {
    const int safe_kernel_size = std::max(1, config.kernel_size);
    const auto kernel =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(safe_kernel_size, safe_kernel_size));
    cv::dilate(binary_roi, binary_roi, kernel, cv::Point(-1, -1), 1);  // ROI进行膨胀操作
  }

  // 找轮廓
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Mat contour_input = binary_roi.clone();
  cv::findContours(
    contour_input, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

  int selected_index = -1;
  double best_score = std::numeric_limits<double>::max();
  int accepted_contour_count = 0;
  std::vector<RCenterCandidateDebug> candidates(contours.size());

  // 遍历候选轮廓并评分
  for (std::size_t i = 0; i < contours.size(); ++i) {
    auto & candidate = candidates[i];
    candidate.contour_index = static_cast<int>(i);
    const double area = cv::contourArea(contours[i]);
    candidate.area = area;
    // 面积过滤
    if (area < 100.0) candidate.reject_mask |= R_CENTER_REJECT_SMALL_AREA;
    // 计算最小外接矩形、长宽比、中心
    const auto rotated_rect = cv::minAreaRect(contours[i]);
    const float min_side =
      std::max(1.0f, std::min(rotated_rect.size.width, rotated_rect.size.height));
    const float max_side = std::max(rotated_rect.size.width, rotated_rect.size.height);
    candidate.aspect_ratio = max_side / min_side;
    candidate.center_in_roi = rotated_rect.center;
    const cv::Point2f rect_center_global =
      rotated_rect.center +
      cv::Point2f(static_cast<float>(roi_rect.x), static_cast<float>(roi_rect.y));

    // 计算偏移比（相对原始R标中心）
    candidate.offset_ratio = cv::norm(rect_center_global - coarse_r_center) / roi_radius;

    // 层级过滤：仅接受顶层轮廓 + 子轮廓面积占比过滤
    if (config.hierarchy_filter && i < hierarchy.size()) {
      if (hierarchy[i][3] >= 0) candidate.reject_mask |= R_CENTER_REJECT_NESTED;
      candidate.child_area_ratio =
        descendant_contour_area(contours, hierarchy, static_cast<int>(i)) /
        std::max(1.0, std::abs(area));
      if (candidate.child_area_ratio > config.max_child_area_ratio) {
        candidate.reject_mask |= R_CENTER_REJECT_CHILD_AREA;
      }
    }

    // 简化评分
    candidate.score = candidate.aspect_ratio + candidate.offset_ratio + candidate.child_area_ratio;

    if (candidate.reject_mask == R_CENTER_REJECT_NONE) {
      accepted_contour_count++;
    }
    // 选出最佳候选
    if (candidate.reject_mask == R_CENTER_REJECT_NONE && candidate.score < best_score) {
      best_score = candidate.score;
      selected_index = static_cast<int>(i);
    }
  }

  cv::Point2f refined_r_center = coarse_r_center;
  bool refined_valid = false;
  if (selected_index >= 0) {
    const auto selected_rect = cv::minAreaRect(contours[static_cast<std::size_t>(selected_index)]);
    refined_r_center = selected_rect.center +
                       cv::Point2f(static_cast<float>(roi_rect.x), static_cast<float>(roi_rect.y));
    refined_valid = true;
  }

  if (debug) {
    debug->refined_r_center = refined_r_center;
    debug->roi_rect = roi_rect;
    debug->binary_roi = binary_roi.clone();
    debug->contours = contours;
    debug->hierarchy = hierarchy;
    debug->candidates = candidates;
    debug->accepted_contour_count = accepted_contour_count;
    debug->best_score = best_score;
  }

  const auto fallback_reason = contours.empty() ? RCenterFallbackReason::NoContours
                                                : RCenterFallbackReason::NoAcceptedCandidate;
  finish_debug(
    refined_valid, refined_valid ? RCenterFallbackReason::None : fallback_reason, selected_index);
  return refined_valid ? refined_r_center : coarse_r_center;
}

namespace
{

// 选择观测扇叶（但不一定是击打扇叶，击打扇叶在aim中选出）
std::size_t select_target_index(
  const std::vector<FanBlade> & fanblades, const cv::Point2f & r_center,
  const cv::Size & image_size, const std::optional<PowerRune> & last_powerrune,
  BuffTargetSelectionDebug * target_selection_debug)
{
  const bool has_previous_target =
    last_powerrune && !last_powerrune->fanblades.empty() &&
    last_powerrune->target().type == _target;
  const double last_angle = has_previous_target
                              ? image_angle(last_powerrune->target().center, last_powerrune->r_center)
                              : std::numeric_limits<double>::quiet_NaN();
  const cv::Point2f image_center(image_size.width * 0.5f, image_size.height * 0.5f);

  const auto candidate_debug_for = [&](std::size_t fanblade_index) -> BuffCandidateDebug * {
    if (!target_selection_debug) return nullptr;
    for (auto & candidate : target_selection_debug->candidates) {
      if (candidate.fanblade_index == static_cast<int>(fanblade_index)) return &candidate;
    }
    return nullptr;
  };
  const auto record_candidate_geometry = [&]() {
    if (!target_selection_debug) return;
    target_selection_debug->has_previous_target = has_previous_target;
    if (has_previous_target) {
      target_selection_debug->previous_target_center = last_powerrune->target().center;
      target_selection_debug->previous_r_center = last_powerrune->r_center;
      target_selection_debug->previous_target_image_angle = last_angle;
    }
    for (std::size_t i = 0; i < fanblades.size(); ++i) {
      auto * candidate = candidate_debug_for(i);
      if (!candidate) continue;
      candidate->center = fanblades[i].center;
      candidate->image_angle = image_angle(fanblades[i].center, r_center);
      candidate->image_center_distance = cv::norm(fanblades[i].center - image_center);
      if (has_previous_target) {
        candidate->history_angle_residual =
          abs_angle_diff(candidate->image_angle, last_angle);
      }
    }
  };
  const auto record_selection = [&](std::size_t selected, TargetSelectionReason reason) {
    if (!target_selection_debug) return;
    target_selection_debug->reason = reason;
    target_selection_debug->selected_fanblade_index = static_cast<int>(selected);
    target_selection_debug->selected_image_angle = image_angle(fanblades[selected].center, r_center);
    if (has_previous_target) {
      target_selection_debug->selected_history_angle_residual =
        abs_angle_diff(target_selection_debug->selected_image_angle, last_angle);
    }
    if (auto * candidate = candidate_debug_for(selected)) {
      candidate->selected_as_target = true;
      target_selection_debug->selected_source_index = candidate->source_index;
    }
  };

  record_candidate_geometry();
  if (fanblades.size() == 1) {
    record_selection(0, TargetSelectionReason::SingleCandidate);
    return 0;
  }

  std::size_t best_index = 0;
  // 如果上一帧有目标，利用角度连续性
  if (has_previous_target) {
    double best_diff = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < fanblades.size(); ++i) {
      const double diff = abs_angle_diff(image_angle(fanblades[i].center, r_center), last_angle);
      if (
        diff < best_diff - 1e-6 || (std::abs(diff - best_diff) <= 1e-6 &&
                                    fanblades[i].confidence > fanblades[best_index].confidence)) {
        best_diff = diff;
        best_index = i;
      }
    }

    record_selection(best_index, TargetSelectionReason::HistoryAngle);
    return best_index;
  }

  // 如果没有上一帧，则选择离图像中心最近的
  double best_distance = std::numeric_limits<double>::max();
  for (std::size_t i = 0; i < fanblades.size(); ++i) {
    const double distance = cv::norm(fanblades[i].center - image_center);
    if (
      distance < best_distance - 1e-6 ||
      (std::abs(distance - best_distance) <= 1e-6 &&
       fanblades[i].confidence > fanblades[best_index].confidence)) {
      best_distance = distance;
      best_index = i;
    }
  }

  record_selection(best_index, TargetSelectionReason::ImageCenter);
  return best_index;
}
}  // namespace

Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0), MODE_(config)
{
  const auto yaml = YAML::LoadFile(config);
  const auto detector = yaml["buff_detector"];
  if (!detector || !detector.IsMap()) {
    throw std::invalid_argument("[Buff_Detector config] 'buff_detector' must be a mapping.");
  }

  refine_r_center_ =
    require_detector_scalar<bool>(detector, "r_center_refine", "buff_detector.r_center_refine");
  r_center_refine_config_.threshold = require_detector_scalar<int>(
    detector, "r_center_refine_threshold", "buff_detector.r_center_refine_threshold");
  r_center_refine_config_.radius_scale = require_detector_scalar<double>(
    detector, "r_center_refine_radius_scale", "buff_detector.r_center_refine_radius_scale");
  r_center_refine_config_.kernel_size = require_detector_scalar<int>(
    detector, "r_center_refine_kernel_size", "buff_detector.r_center_refine_kernel_size");
  r_center_refine_config_.hierarchy_filter = require_detector_scalar<bool>(
    detector, "r_center_refine_hierarchy_filter",
    "buff_detector.r_center_refine_hierarchy_filter");
  r_center_refine_config_.max_child_area_ratio = require_detector_scalar<double>(
    detector, "r_center_refine_max_child_area_ratio",
    "buff_detector.r_center_refine_max_child_area_ratio");
  validate_r_center_refine_config(r_center_refine_config_);
}

void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt;
    return;
  }
  status_ = TEM_LOSE;
}

void Buff_Detector::set_expected_color_label(int label)
{
  if (label != 0 && label != 1) throw std::invalid_argument("buff color label must be 0 or 1");
  if (expected_color_label_ == label) return;
  expected_color_label_ = label;
  status_ = LOSE;
  lose_ = 0;
  last_powerrune_ = std::nullopt;
  tools::logger()->info(
    "[Buff_Detector] Detection color set to {}", label == 0 ? "red" : "blue");
}

// 主函数
std::optional<PowerRune> Buff_Detector::build_powerrune(
  const std::vector<YOLO11_BUFF::Object> & results, const cv::Mat & bgr_img,
  const std::optional<cv::Point2f> & debug_vector, RCenterRefineDebug * r_center_debug,
  BuffTargetSelectionDebug * target_selection_debug)
{
  // 获取YOLO结果，构造FanBlade对象
  auto filtered_results = results;
  if (expected_color_label_.has_value()) {
    filtered_results.erase(
      std::remove_if(
        filtered_results.begin(), filtered_results.end(), [&](const auto & object) {
          return object.label != *expected_color_label_;
        }),
      filtered_results.end());
  }
  auto fanblades = objects_to_fanblades(filtered_results, target_selection_debug);
  if (fanblades.empty()) return std::nullopt;

  // 粗定位R标中心，并可选择重定位
  const auto coarse_r_center = average_r_center(fanblades);
  auto r_center = coarse_r_center;
  if (refine_r_center_ || r_center_debug) {
    const auto refined_r_center =
      refine_r_center(fanblades, bgr_img, coarse_r_center, r_center_refine_config_, r_center_debug);
    if (refine_r_center_) r_center = refined_r_center;
  }

  // 更新所有扇叶的R标中心
  for (auto & fanblade : fanblades) {
    fanblade.points[0] = r_center;
    fanblade.center = (fanblade.points[1] + fanblade.points[3]) * 0.5f;
  }

  // debug
  if (debug_vector) {
    std::vector<FanBlade> filtered;
    filtered.reserve(1);
    for (const auto & fanblade : fanblades) {
      if (cv::norm((fanblade.center - r_center) - *debug_vector) < 10 || fanblades.size() == 1) {
        filtered.emplace_back(fanblade);
        break;
      }
    }
    fanblades = std::move(filtered);
    if (fanblades.empty()) return std::nullopt;
    if (target_selection_debug) target_selection_debug->reason = TargetSelectionReason::DebugVector;
  }

  // 选择观测扇叶
  const auto target_index =
    select_target_index(fanblades, r_center, bgr_img.size(), last_powerrune_, target_selection_debug);
  fanblades[target_index].type = _target;
  std::iter_swap(fanblades.begin(), fanblades.begin() + static_cast<std::ptrdiff_t>(target_index));

  // 构造PowerRune
  PowerRune powerrune(fanblades, r_center, last_powerrune_);
  if (powerrune.is_unsolve()) return std::nullopt;
  return powerrune;
}

std::optional<PowerRune> Buff_Detector::detect_24(
  cv::Mat & bgr_img, RCenterRefineDebug * r_center_debug)
{
  return detect(bgr_img, r_center_debug);
}

std::optional<PowerRune> Buff_Detector::detect(
  cv::Mat & bgr_img, RCenterRefineDebug * r_center_debug)
{
  const auto results = MODE_.get_multicandidateboxes(bgr_img);
  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  auto powerrune = build_powerrune(results, bgr_img, std::nullopt, r_center_debug);
  if (!powerrune) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  last_powerrune_ = powerrune;
  return powerrune;
}

std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f v)
{
  const auto results = MODE_.get_multicandidateboxes(bgr_img);
  if (results.empty()) return std::nullopt;

  auto powerrune = build_powerrune(results, bgr_img, v);
  if (!powerrune) return std::nullopt;
  return powerrune;
}

bool Buff_Detector::submit(
  const cv::Mat & bgr_img, int frame_count, const std::chrono::steady_clock::time_point & timestamp)
{
  return MODE_.submit(bgr_img, frame_count, timestamp);
}

bool Buff_Detector::submit(const cv::Mat & bgr_img, int frame_count)
{
  return MODE_.submit(bgr_img, frame_count);
}

bool Buff_Detector::fetch(
  std::optional<PowerRune> & powerrune, cv::Mat & image, int & frame_count, double & detect_dt_ms,
  bool debug, RCenterRefineDebug * r_center_debug,
  std::chrono::steady_clock::time_point * timestamp, BuffDetectPerfStats * perf_stats,
  BuffTargetSelectionDebug * target_selection_debug)
{
  YOLO11_BUFF::AsyncResult result;
  if (!MODE_.fetch(result)) return false;

  image = std::move(result.image);
  frame_count = result.frame_count;
  detect_dt_ms = result.detect_dt_ms;
  if (timestamp) *timestamp = result.timestamp;
  if (perf_stats) {
    perf_stats->preprocess_dt_ms = result.preprocess_dt_ms;
    perf_stats->infer_dt_ms = result.infer_dt_ms;
    perf_stats->postprocess_dt_ms = result.postprocess_dt_ms;
  }
  if (target_selection_debug) *target_selection_debug = {};

  if (result.objects.empty()) {
    if (debug && r_center_debug) *r_center_debug = {};
    handle_lose();
    powerrune = std::nullopt;
    return true;
  }

  powerrune =
    build_powerrune(
      result.objects, image, std::nullopt, debug ? r_center_debug : nullptr,
      target_selection_debug);
  if (!powerrune) {
    handle_lose();
    return true;
  }

  status_ = TRACK;
  lose_ = 0;
  last_powerrune_ = powerrune;
  return true;
}

// 清空底层异步结果，用于模式切换时丢旧帧
std::size_t Buff_Detector::discard_pending_results()
{
  std::size_t dropped = 0;
  YOLO11_BUFF::AsyncResult result;
  while (MODE_.fetch(result)) dropped++;
  return dropped;
}

std::size_t Buff_Detector::async_stale_drop_count() const { return MODE_.async_stale_drop_count(); }

std::size_t Buff_Detector::async_out_of_order_drop_count() const
{
  return MODE_.async_out_of_order_drop_count();
}

}  // namespace auto_buff
