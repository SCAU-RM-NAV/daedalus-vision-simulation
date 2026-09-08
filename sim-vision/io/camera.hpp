#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

namespace io
{
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;
  virtual void set_active(bool active);
  virtual void set_exposure_ms(double exposure_ms);
};

class Camera
{
public:
  explicit Camera(const std::string & config_path, bool active = true);
  explicit Camera(const YAML::Node & yaml, bool active = true);
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
  void set_active(bool active);
  void set_exposure_ms(double exposure_ms);

private:
  std::unique_ptr<CameraBase> camera_;
  bool rotate_180_ = false;
  void init(const YAML::Node & yaml, bool active);
};

}  // namespace io

#endif  // IO__CAMERA_HPP
