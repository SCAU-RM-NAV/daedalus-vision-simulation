#ifndef AUTO_AIM__YOLO11_HPP
#define AUTO_AIM__YOLO11_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <queue>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor_post_filter.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{
class YOLO11 : public YOLOBase
{
public:
  YOLO11(const std::string & config_path, bool debug);
  ~YOLO11() override;

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

  bool supports_async() const override { return true; }

  bool submit(
    const cv::Mat & img, int frame_count,
    const std::chrono::steady_clock::time_point & timestamp) override;

  bool submit(
    const cv::Mat & img, const std::chrono::steady_clock::time_point & timestamp) override;

  bool fetch(YOLOAsyncResult & result) override;
  bool try_fetch(YOLOAsyncResult & result) override;

private:
  class ThreadPool;
  struct RequestContext;

  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;

  static constexpr int class_num_ = static_cast<int>(armor_properties.size());
  static constexpr int keypoint_num_ = 4;
  static constexpr int keypoint_stride_ = 3;
  static constexpr int output_feature_num_ =
    4 + class_num_ + keypoint_num_ * keypoint_stride_;
  const float keypoint_score_threshold_ = 0.5;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;
  ArmorPostFilter post_filter_;

  std::unique_ptr<ThreadPool> preprocess_pool_;
  std::unique_ptr<ThreadPool> postprocess_pool_;
  std::vector<std::unique_ptr<RequestContext>> request_contexts_;
  std::queue<int> free_request_ids_;
  std::mutex request_mutex_;

  std::deque<YOLOAsyncResult> result_queue_;
  std::mutex result_mutex_;
  std::size_t result_queue_limit_ = 2;
  std::size_t num_requests_ = 2;

  std::atomic<bool> stopping_{false};
  std::mutex parse_mutex_;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  bool prepare_input(const cv::Mat & raw_img, cv::Mat & input, double & scale) const;

  std::list<Armor> parse(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count,
    bool draw_debug = true);

  void handle_completion(RequestContext * ctx, std::exception_ptr ex_ptr);
  void release_request(int id);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLO11_HPP
