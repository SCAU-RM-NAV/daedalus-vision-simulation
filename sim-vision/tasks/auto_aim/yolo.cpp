#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/yolov5.hpp"
#ifdef SP_VISION_WITH_OPENVINO
#include "yolos/yolo11.hpp"
#include "yolos/yolov8.hpp"
#endif

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  if (yolo_name == "yolov8") {
#ifdef SP_VISION_WITH_OPENVINO
    yolo_ = std::make_unique<YOLOV8>(config_path, debug);
#else
    throw std::runtime_error("yolov8 requires an OpenVINO build.");
#endif
  }

  else if (yolo_name == "yolo11") {
#ifdef SP_VISION_WITH_OPENVINO
    yolo_ = std::make_unique<YOLO11>(config_path, debug);
#else
    throw std::runtime_error("yolo11 requires an OpenVINO build.");
#endif
  }

  else if (yolo_name == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug);
  }

  else {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

bool YOLO::supports_async() const
{
  return yolo_ && yolo_->supports_async();
}

bool YOLO::submit(
  const cv::Mat & img, int frame_count,
  const std::chrono::steady_clock::time_point & timestamp)
{
  return yolo_ && yolo_->submit(img, frame_count, timestamp);
}

bool YOLO::submit(
  const cv::Mat & img, const std::chrono::steady_clock::time_point & timestamp)
{
  return submit(img, -1, timestamp);
}

bool YOLO::fetch(YOLOAsyncResult & result)
{
  return yolo_ && yolo_->fetch(result);
}

bool YOLO::try_fetch(YOLOAsyncResult & result)
{
  return fetch(result);
}

}  // namespace auto_aim
