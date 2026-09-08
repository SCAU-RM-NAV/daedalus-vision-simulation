#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#ifdef SP_VISION_WITH_OPENVINO
#include <openvino/openvino.hpp>
#endif
#ifdef SP_VISION_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif
#include <string>
#include <vector>

#include "tasks/auto_aim/armor_post_filter.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
#include "tools/onnxruntime_cuda.hpp"
#endif

namespace auto_aim
{
class YOLOV5 : public YOLOBase
{
public:
  YOLOV5(const std::string & config_path, bool debug);
  ~YOLOV5() override;

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  enum class InferenceBackend { OpenVINO, OnnxRuntime, OnnxRuntimeCuda, OpenCVDnnCuda };

  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 13;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

#ifdef SP_VISION_WITH_OPENVINO
  ov::Core core_;
  ov::CompiledModel compiled_model_;
#endif
  cv::dnn::Net dnn_net_;

  InferenceBackend backend_ = InferenceBackend::OpenVINO;
#ifdef SP_VISION_HAS_ONNXRUNTIME_CUDA
  std::unique_ptr<inference::OnnxRuntimeCudaSession> onnxruntime_cuda_;
#endif
#ifdef SP_VISION_WITH_ONNXRUNTIME
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::SessionOptions ort_session_options_;
  std::string ort_input_name_;
  std::string ort_output_name_;
#endif

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_;
  ArmorPostFilter post_filter_;
  friend class MultiThreadDetector;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLOV5_HPP
