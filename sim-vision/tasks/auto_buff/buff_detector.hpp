#ifndef AUTO_BUFF_DETECTOR_HPP
#define AUTO_BUFF_DETECTOR_HPP

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <array>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"
const int LOSE_MAX = 20;  // 丢失的阈值
namespace auto_buff
{
// 拒绝掩码（一个int32可以同时标记多个拒绝原因）
enum RCenterRejectMask : std::uint32_t
{
  R_CENTER_REJECT_NONE = 0,
  R_CENTER_REJECT_SMALL_AREA = 1U << 0, // 面积太小
  R_CENTER_REJECT_NESTED = 1U << 1, // 嵌套轮廓（有父轮廓）
  R_CENTER_REJECT_INVALID_PERIMETER = 1U << 2, // 周长无效
  R_CENTER_REJECT_CHILD_AREA = 1U << 3, // 子轮廓面积占比太大
};

enum class RCenterFallbackReason
{
  None,
  EmptyImage,
  InvalidBladeRadius,
  EmptyRoi,
  NoContours,
  NoAcceptedCandidate,
};

enum class RCenterDebugView
{
  Off,
  Roi,
  Binary,
  Contours,
  All,
};

struct RCenterRefineConfig
{
  int threshold = 100;
  double radius_scale = 0.3;
  int kernel_size = 3;
  bool hierarchy_filter = true;
  double max_child_area_ratio = 0.50;
};

struct RCenterCandidateDebug
{
  int contour_index = -1;
  cv::Point2f center_in_roi;
  double area = 0.0;
  double aspect_ratio = 0.0;
  double child_area_ratio = 0.0;
  double offset_ratio = 0.0;
  double score = std::numeric_limits<double>::infinity();
  std::uint32_t reject_mask = R_CENTER_REJECT_NONE;
};

struct RCenterRefineDebug
{
  cv::Point2f coarse_r_center;
  cv::Point2f refined_r_center;
  cv::Rect roi_rect;
  cv::Mat binary_roi;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<RCenterCandidateDebug> candidates;
  int selected_contour_index = -1;
  int accepted_contour_count = 0;
  double best_score = std::numeric_limits<double>::infinity();
  double refine_dt_ms = 0.0;
  bool refined_valid = false;
  RCenterFallbackReason fallback_reason = RCenterFallbackReason::None;
};

struct RCenterDebugImages
{
  cv::Mat roi;
  cv::Mat binary;
  cv::Mat contours;
};

std::optional<RCenterDebugView> parse_r_center_debug_view(const std::string & value);
const char * r_center_fallback_reason_name(RCenterFallbackReason reason);
std::string r_center_reject_mask_text(std::uint32_t reject_mask);
RCenterDebugImages make_r_center_debug_images(
  const cv::Mat & image, const RCenterRefineDebug & debug);
void show_r_center_debug_views(
  RCenterDebugView view, const cv::Mat & image, const RCenterRefineDebug & debug,
  const std::string & window_prefix = "buff refine");

cv::Point2f refine_r_center(
  const std::vector<FanBlade> & fanblades, const cv::Mat & bgr_img,
  const cv::Point2f & coarse_r_center, const RCenterRefineConfig & config,
  RCenterRefineDebug * debug = nullptr);

struct BuffDetectPerfStats
{
  double preprocess_dt_ms = 0.0;
  double infer_dt_ms = 0.0;
  double postprocess_dt_ms = 0.0;
};

enum class TargetSelectionReason
{
  None,
  SingleCandidate,
  HistoryAngle,
  ImageCenter,
  DebugVector,
};

struct BuffCandidateDebug
{
  int source_index = -1;
  int fanblade_index = -1;
  bool valid_keypoints = false;
  int label = -1;
  double confidence = 0.0;
  cv::Rect_<float> rect;
  int keypoint_count = 0;
  std::array<cv::Point2f, 4> keypoints{};
  cv::Point2f center;
  double image_angle = std::numeric_limits<double>::quiet_NaN();
  double history_angle_residual = std::numeric_limits<double>::quiet_NaN();
  double image_center_distance = std::numeric_limits<double>::quiet_NaN();
  bool selected_as_target = false;
};

struct BuffTargetSelectionDebug
{
  int raw_object_count = 0;
  int valid_fanblade_count = 0;
  bool has_previous_target = false;
  cv::Point2f previous_target_center;
  cv::Point2f previous_r_center;
  double previous_target_image_angle = std::numeric_limits<double>::quiet_NaN();
  TargetSelectionReason reason = TargetSelectionReason::None;
  int selected_fanblade_index = -1;
  int selected_source_index = -1;
  double selected_image_angle = std::numeric_limits<double>::quiet_NaN();
  double selected_history_angle_residual = std::numeric_limits<double>::quiet_NaN();
  std::vector<BuffCandidateDebug> candidates;
};

class Buff_Detector
{
public:
  Buff_Detector(const std::string & config);

  // detect是同步推理用，submit/fetch是异步推理用

  std::optional<PowerRune> detect_24(
    cv::Mat & bgr_img, RCenterRefineDebug * r_center_debug = nullptr);

  std::optional<PowerRune> detect(cv::Mat & bgr_img, RCenterRefineDebug * r_center_debug = nullptr);

  // YOLO buff labels are 0=red and 1=blue. Simulation uses this to select
  // the opponent color announced through Talos camp feedback.
  void set_expected_color_label(int label);

  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

  bool submit(
    const cv::Mat & bgr_img, int frame_count,
    const std::chrono::steady_clock::time_point & timestamp);

  bool submit(const cv::Mat & bgr_img, int frame_count);

  bool fetch(
    std::optional<PowerRune> & powerrune, cv::Mat & image, int & frame_count, double & detect_dt_ms,
    bool debug = false, RCenterRefineDebug * r_center_debug = nullptr,
    std::chrono::steady_clock::time_point * timestamp = nullptr,
    BuffDetectPerfStats * perf_stats = nullptr,
    BuffTargetSelectionDebug * target_selection_debug = nullptr);

  std::size_t discard_pending_results();

  std::size_t async_stale_drop_count() const;
  std::size_t async_out_of_order_drop_count() const;

private:
  std::optional<PowerRune> build_powerrune(
    const std::vector<YOLO11_BUFF::Object> & results, const cv::Mat & bgr_img,
    const std::optional<cv::Point2f> & debug_vector = std::nullopt,
    RCenterRefineDebug * r_center_debug = nullptr,
    BuffTargetSelectionDebug * target_selection_debug = nullptr);

  void handle_lose();

  YOLO11_BUFF MODE_;
  Track_status status_;
  int lose_;  // 丢失的次数
  double lastlen_;
  bool refine_r_center_ = false;
  RCenterRefineConfig r_center_refine_config_;
  std::optional<PowerRune> last_powerrune_ = std::nullopt;
  std::optional<int> expected_color_label_ = std::nullopt;
};
}  // namespace auto_buff
#endif  // DETECTOR_HPP
