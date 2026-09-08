#include "camera.hpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace io
{
void CameraBase::set_active(bool) {}

void CameraBase::set_exposure_ms(double) {}

Camera::Camera(const std::string & config_path, bool active)
{
  auto yaml = tools::load(config_path);
  init(yaml, active);
}

Camera::Camera(const YAML::Node & yaml, bool active)
{
  init(yaml, active);
}

void Camera::init(const YAML::Node & yaml, bool active)
{
  auto camera_name = tools::read<std::string>(yaml, "camera_name");
  auto exposure_ms = tools::read<double>(yaml, "exposure_ms");
  rotate_180_ = yaml["rotate_180"] ? yaml["rotate_180"].as<bool>() : false;
  tools::logger()->info(
    "{} camera image rotation: {}", camera_name, rotate_180_ ? "180 degrees" : "disabled");

  if (camera_name == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
    camera_->set_active(active);
  }

  else if (camera_name == "hikrobot") {
    auto gain = tools::read<double>(yaml, "gain");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    auto camera_user_id = yaml["camera_user_id"] ? yaml["camera_user_id"].as<std::string>() : "";
    auto camera_index = yaml["camera_index"] ? yaml["camera_index"].as<int>() : 0;
    camera_ =
      std::make_unique<HikRobot>(exposure_ms, gain, vid_pid, camera_user_id, camera_index, active);
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
  if (rotate_180_ && !img.empty()) {
    cv::rotate(img, img, cv::ROTATE_180);
  }
}

void Camera::set_active(bool active)
{
  camera_->set_active(active);
}

void Camera::set_exposure_ms(double exposure_ms)
{
  camera_->set_exposure_ms(exposure_ms);
}

}  // namespace io
