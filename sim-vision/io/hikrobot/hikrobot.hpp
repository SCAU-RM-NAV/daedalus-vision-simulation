#ifndef IO__HIKROBOT_HPP
#define IO__HIKROBOT_HPP

#include <atomic>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "MvCameraControl.h"
#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
class HikRobot : public CameraBase
{
public:
  HikRobot(
    double exposure_ms, double gain, const std::string & vid_pid,
    const std::string & camera_user_id = "", int camera_index = 0, bool active = true);
  ~HikRobot() override;
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
  void set_active(bool active) override;
  void set_exposure_ms(double exposure_ms) override;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_us_;
  double gain_;
  std::string camera_user_id_;
  int camera_index_;

  std::thread daemon_thread_;
  std::atomic<bool> daemon_quit_;
  std::atomic<bool> enabled_;
  std::mutex camera_mutex_;

  void * handle_ = nullptr;
  std::thread capture_thread_;
  std::atomic<bool> capturing_ = false;
  std::atomic<bool> capture_quit_ = false;
  tools::ThreadSafeQueue<CameraData> queue_;

  int vid_, pid_;

  void capture_start();
  void capture_stop();

  void set_float_value(const std::string & name, double value);
  void set_enum_value(const std::string & name, unsigned int value);

  void set_vid_pid(const std::string & vid_pid);
  void reset_usb() const;
  std::string get_device_user_id(const MV_CC_DEVICE_INFO * info) const;
  int find_device_index(const MV_CC_DEVICE_INFO_LIST & device_list) const;
};

}  // namespace io

#endif  // IO__HIKROBOT_HPP
