#include "hikrobot.hpp"

#include <libusb-1.0/libusb.h>

#include <cstring>
#include <unordered_map>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{
HikRobot::HikRobot(
  double exposure_ms, double gain, const std::string & vid_pid,
  const std::string & camera_user_id, int camera_index, bool active)
: exposure_us_(exposure_ms * 1e3),
  gain_(gain),
  camera_user_id_(camera_user_id),
  camera_index_(camera_index),
  daemon_quit_(false),
  enabled_(active),
  queue_(1),
  vid_(-1),
  pid_(-1)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("Unable to init libusb!");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's daemon thread started.");

    if (enabled_) capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      if (!enabled_) {
        capture_stop();
        continue;
      }

      if (capturing_) continue;

      capture_stop();
      reset_usb();
      capture_start();
    }

    capture_stop();

    tools::logger()->info("HikRobot's daemon thread stopped.");
  }};
}

HikRobot::~HikRobot()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("HikRobot destructed.");
}

void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);

  img = data.img;
  timestamp = data.timestamp;
}

void HikRobot::set_active(bool active)
{
  enabled_ = active;
  queue_.clear();

  if (active) {
    capture_start();
  } else {
    capture_stop();
  }
}

void HikRobot::set_exposure_ms(double exposure_ms)
{
  std::lock_guard<std::mutex> lock(camera_mutex_);

  exposure_us_ = exposure_ms * 1e3;
  if (handle_ != nullptr) set_float_value("ExposureTime", exposure_us_);
}

void HikRobot::capture_start()
{
  std::lock_guard<std::mutex> lock(camera_mutex_);

  if (!enabled_ || capturing_ || handle_ != nullptr) return;

  capturing_ = false;
  capture_quit_ = false;

  unsigned int ret;

  MV_CC_DEVICE_INFO_LIST device_list;
  // The sentry camera is a USB3 HikRobot camera.  The older Nano-verified
  // implementation enumerated USB only; requesting GigE devices as well can
  // make the ARM MVS runtime fail before it reaches the USB camera.
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_EnumDevices failed: {:#x}", ret);
    return;
  }

  if (device_list.nDeviceNum == 0) {
    tools::logger()->warn("Not found camera!");
    return;
  }

  int selected_index = find_device_index(device_list);
  if (selected_index < 0) {
    if (camera_user_id_.empty()) {
      tools::logger()->warn("Not found camera index {}!", camera_index_);
    } else {
      tools::logger()->warn("Not found camera with user id \"{}\"!", camera_user_id_);
    }
    return;
  }

  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[selected_index]);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CreateHandle failed: {:#x}", ret);
    handle_ = nullptr;
    return;
  }

  ret = MV_CC_OpenDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_OpenDevice failed: {:#x}", ret);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return;
  }

  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 180);

  ret = MV_CC_StartGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StartGrabbing failed: {:#x}", ret);
    MV_CC_CloseDevice(handle_);
    MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
    return;
  }

  capturing_ = true;
  capture_thread_ = std::thread{[this] {
    tools::logger()->info("HikRobot's capture thread started.");

    MV_FRAME_OUT raw;
    MV_CC_PIXEL_CONVERT_PARAM cvt_param;

    while (!capture_quit_) {
      std::this_thread::sleep_for(1ms);

      unsigned int ret;
      unsigned int nMsec = 100;

      ret = MV_CC_GetImageBuffer(handle_, &raw, nMsec);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_GetImageBuffer failed: {:#x}", ret);
        break;
      }

      auto timestamp = std::chrono::steady_clock::now();
      cv::Mat img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);

      cvt_param.nWidth = raw.stFrameInfo.nWidth;
      cvt_param.nHeight = raw.stFrameInfo.nHeight;

      cvt_param.pSrcData = raw.pBufAddr;
      cvt_param.nSrcDataLen = raw.stFrameInfo.nFrameLen;
      cvt_param.enSrcPixelType = raw.stFrameInfo.enPixelType;

      cvt_param.pDstBuffer = img.data;
      cvt_param.nDstBufferSize = img.total() * img.elemSize();
      cvt_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed;

      // ret = MV_CC_ConvertPixelType(handle_, &cvt_param);
      const auto & frame_info = raw.stFrameInfo;
      auto pixel_type = frame_info.enPixelType;
      cv::Mat dst_image;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      cv::cvtColor(img, dst_image, type_map.at(pixel_type));
      img = dst_image;

      queue_.push({img, timestamp});

      ret = MV_CC_FreeImageBuffer(handle_, &raw);
      if (ret != MV_OK) {
        tools::logger()->warn("MV_CC_FreeImageBuffer failed: {:#x}", ret);
        break;
      }
    }

    capturing_ = false;
    tools::logger()->info("HikRobot's capture thread stopped.");
  }};
}

void HikRobot::capture_stop()
{
  std::lock_guard<std::mutex> lock(camera_mutex_);

  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  capturing_ = false;

  if (handle_ == nullptr) return;

  unsigned int ret;

  ret = MV_CC_StopGrabbing(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", ret);
  }

  ret = MV_CC_CloseDevice(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", ret);
  }

  ret = MV_CC_DestroyHandle(handle_);
  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", ret);
  }

  handle_ = nullptr;
}

void HikRobot::set_float_value(const std::string & name, double value)
{
  unsigned int ret;

  ret = MV_CC_SetFloatValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetFloatValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  unsigned int ret;

  ret = MV_CC_SetEnumValue(handle_, name.c_str(), value);

  if (ret != MV_OK) {
    tools::logger()->warn("MV_CC_SetEnumValue(\"{}\", {}) failed: {:#x}", name, value, ret);
    return;
  }
}

void HikRobot::set_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (const std::exception &) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
  }
}

std::string HikRobot::get_device_user_id(const MV_CC_DEVICE_INFO * info) const
{
  if (info == nullptr) return "";

  if (info->nTLayerType == MV_USB_DEVICE || info->nTLayerType == MV_VIR_USB_DEVICE) {
    const auto * name =
      reinterpret_cast<const char *>(info->SpecialInfo.stUsb3VInfo.chUserDefinedName);
    return std::string(name, strnlen(name, sizeof(info->SpecialInfo.stUsb3VInfo.chUserDefinedName)));
  }

  if (
    info->nTLayerType == MV_GIGE_DEVICE || info->nTLayerType == MV_VIR_GIGE_DEVICE ||
    info->nTLayerType == MV_GENTL_GIGE_DEVICE) {
    const auto * name =
      reinterpret_cast<const char *>(info->SpecialInfo.stGigEInfo.chUserDefinedName);
    return std::string(name, strnlen(name, sizeof(info->SpecialInfo.stGigEInfo.chUserDefinedName)));
  }

  return "";
}

int HikRobot::find_device_index(const MV_CC_DEVICE_INFO_LIST & device_list) const
{
  for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
    const auto * info = device_list.pDeviceInfo[i];
    const auto user_id = get_device_user_id(info);
    tools::logger()->info("HikRobot device[{}] user_id = \"{}\"", i, user_id);

    if (!camera_user_id_.empty() && user_id == camera_user_id_) return static_cast<int>(i);
  }

  if (!camera_user_id_.empty()) return -1;
  if (camera_index_ < 0 || static_cast<unsigned int>(camera_index_) >= device_list.nDeviceNum) {
    return -1;
  }

  return camera_index_;
}

void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;

  // https://github.com/ralight/usb-reset/blob/master/usb-reset.c
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) {
    tools::logger()->warn("Unable to open usb!");
    return;
  }

  if (libusb_reset_device(handle))
    tools::logger()->warn("Unable to reset usb!");
  else
    tools::logger()->info("Reset usb successfully :)");

  libusb_close(handle);
}

}  // namespace io
