#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <chrono>
#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "armor.hpp"

namespace auto_aim
{
struct YOLOAsyncResult
{
  cv::Mat img;
  std::list<Armor> armors;

  // 新接口使用 timestamp；兼容旧调用处使用 stamp。两者保持相同值。
  std::chrono::steady_clock::time_point timestamp;
  std::chrono::steady_clock::time_point stamp;

  // 从 submit 到结果进入队列的耗时，单位 ms。
  double detect_dt = 0.0;

  int frame_count = -1;
};

class YOLOBase
{
public:
  virtual ~YOLOBase() = default;

  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;

  // Optional async pipeline. YOLOV5/YOLOV8 can ignore it; YOLO11 overrides it.
  virtual bool supports_async() const { return false; }

  virtual bool submit(
    const cv::Mat & img, int frame_count,
    const std::chrono::steady_clock::time_point & timestamp)
  {
    (void)img;
    (void)frame_count;
    (void)timestamp;
    return false;
  }

  // 兼容旧代码：yolo.submit(img, t)。
  virtual bool submit(
    const cv::Mat & img, const std::chrono::steady_clock::time_point & timestamp)
  {
    return submit(img, -1, timestamp);
  }

  virtual bool fetch(YOLOAsyncResult & result)
  {
    (void)result;
    return false;
  }

  // 兼容旧代码：yolo.try_fetch(result)。
  virtual bool try_fetch(YOLOAsyncResult & result) { return fetch(result); }
};

class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = true);

  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  bool supports_async() const;

  bool submit(
    const cv::Mat & img, int frame_count,
    const std::chrono::steady_clock::time_point & timestamp);

  // 兼容旧代码：yolo.submit(img, t)。
  bool submit(
    const cv::Mat & img, const std::chrono::steady_clock::time_point & timestamp);

  bool fetch(YOLOAsyncResult & result);

  // 兼容旧代码：yolo.try_fetch(result)。
  bool try_fetch(YOLOAsyncResult & result);

private:
  std::unique_ptr<YOLOBase> yolo_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
