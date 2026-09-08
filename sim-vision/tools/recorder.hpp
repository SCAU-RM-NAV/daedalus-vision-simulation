#ifndef TOOLS__RECORDER_HPP
#define TOOLS__RECORDER_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "tools/thread_safe_queue.hpp"
namespace tools
{
enum class RecorderOutputMode
{
  legacy_pair,
  session_dir
};

struct RecorderOptions
{
  double fps = 30.0;
  RecorderOutputMode output_mode = RecorderOutputMode::legacy_pair;
  std::string output_path;
  std::string config_path;
  // 0 keeps timestamp-only naming. A positive value uses month_day_slot names by default,
  // cycles slot through 1 ... rotation_slot_count, and removes files in the selected slot.
  std::size_t rotation_slot_count = 0;
  // 0 disables the check. A positive value prevents recording when the disk has less free space.
  std::uintmax_t min_available_space_bytes = 0;
};

struct RecordSampleMetadata
{
  int frame_index = -1;
  double bullet_speed = 0.0;
  std::string vision_mode = "unknown";
};

class Recorder
{
public:
  Recorder(double fps = 30);
  explicit Recorder(const RecorderOptions & options);
  ~Recorder();
  void record(
    const cv::Mat & img, const Eigen::Quaterniond & q,
    const std::chrono::steady_clock::time_point & timestamp);
  void record(
    const cv::Mat & img, const Eigen::Quaterniond & q,
    const std::chrono::steady_clock::time_point & timestamp,
    const RecordSampleMetadata & metadata);

private:
  struct FrameData
  {
    cv::Mat img;
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    std::chrono::steady_clock::time_point timestamp;
    RecordSampleMetadata metadata;
  };

  bool init_;
  std::atomic<bool> stop_thread_;
  RecorderOptions options_;
  double fps_;
  std::string text_path_;
  std::string video_path_;
  std::string imu_csv_path_;
  std::string metadata_path_;
  std::ofstream text_writer_;
  std::ofstream imu_writer_;
  cv::VideoWriter video_writer_;
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_time_;
  int next_frame_index_ = 0;
  std::size_t rotation_slot_ = 0;
  bool recording_disabled_ = false;
  tools::ThreadSafeQueue<FrameData> queue_;
  std::thread saving_thread_;  // 负责保存帧数据的线程

  void configure_paths();
  void clear_rotation_slot();
  bool has_sufficient_available_space() const;
  void stop_recording();
  void init(const cv::Mat & img);
  void write_metadata_file() const;
  void save_to_file();
};

}  // namespace tools

#endif  // TOOLS__RECORDER_HPP
