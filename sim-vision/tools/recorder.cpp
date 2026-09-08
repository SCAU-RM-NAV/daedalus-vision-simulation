#include "recorder.hpp"

#include <fmt/chrono.h>

#include <algorithm>
#include <charconv>
#include <ctime>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "math_tools.hpp"
#include "tools/logger.hpp"

namespace tools
{
namespace
{
std::string timestamp_string()
{
  return fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
}

std::string date_string()
{
  const auto local_time = fmt::localtime(std::time(nullptr));
  return fmt::format("{}_{}", local_time.tm_mon + 1, local_time.tm_mday);
}

std::string escape_yaml_double_quotes(const std::string & value)
{
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '"') escaped.push_back('\\');
    escaped.push_back(c);
  }
  return escaped;
}

std::optional<std::size_t> parse_number(std::string_view value)
{
  if (value.empty()) return std::nullopt;

  std::size_t number = 0;
  const auto * first = value.data();
  const auto * last = value.data() + value.size();
  const auto [parsed_end, error] = std::from_chars(first, last, number);
  if (error != std::errc{} || parsed_end != last) return std::nullopt;
  return number;
}

bool is_date_name(const std::string & value)
{
  const auto separator = value.find('_');
  if (
    separator == std::string::npos || separator == 0 || separator + 1 == value.size() ||
    value.find('_', separator + 1) != std::string::npos) {
    return false;
  }

  const auto month = parse_number(std::string_view{value}.substr(0, separator));
  const auto day = parse_number(std::string_view{value}.substr(separator + 1));
  return month && day && *month >= 1 && *month <= 12 && *day >= 1 && *day <= 31;
}

bool is_legacy_timestamp_name(const std::string & value)
{
  if (value.size() < 19) return false;
  for (std::size_t i = 0; i < 19; ++i) {
    if (i == 4 || i == 7 || i == 13 || i == 16) {
      if (value[i] != '-') return false;
    } else if (i == 10) {
      if (value[i] != '_') return false;
    } else if (value[i] < '0' || value[i] > '9') {
      return false;
    }
  }

  if (value.size() == 19) return true;
  if (value[19] != '.' || value.size() == 20) return false;
  for (std::size_t i = 20; i < value.size(); ++i) {
    if (value[i] < '0' || value[i] > '9') return false;
  }
  return true;
}

std::optional<std::size_t> rotation_slot_from_path(
  const std::filesystem::path & path, std::size_t rotation_slot_count)
{
  const auto stem = path.stem().string();
  const auto separator = stem.rfind('_');
  if (
    separator == std::string::npos || separator + 1 == stem.size() ||
    (!is_date_name(stem.substr(0, separator)) &&
     !is_legacy_timestamp_name(stem.substr(0, separator)))) {
    return std::nullopt;
  }

  const auto slot = parse_number(std::string_view{stem}.substr(separator + 1));
  if (!slot || *slot == 0 || *slot > rotation_slot_count) {
    return std::nullopt;
  }
  return slot;
}

std::size_t next_rotation_slot(const std::filesystem::path & directory, std::size_t rotation_slot_count)
{
  std::error_code error;
  std::size_t newest_slot = 0;
  std::filesystem::file_time_type newest_time;
  bool has_recording = false;

  for (const auto & entry : std::filesystem::directory_iterator(directory, error)) {
    if (error) break;
    if (!entry.is_regular_file(error)) {
      error.clear();
      continue;
    }
    if (entry.path().extension() != ".avi" && entry.path().extension() != ".txt") continue;

    const auto slot = rotation_slot_from_path(entry.path(), rotation_slot_count);
    if (!slot) continue;

    const auto write_time = entry.last_write_time(error);
    if (error) {
      error.clear();
      continue;
    }
    if (!has_recording || write_time > newest_time) {
      newest_time = write_time;
      newest_slot = *slot;
      has_recording = true;
    }
  }

  return has_recording ? newest_slot % rotation_slot_count + 1 : 1;
}
}  // namespace

Recorder::Recorder(double fps) : Recorder(RecorderOptions{fps, RecorderOutputMode::legacy_pair})
{
}

Recorder::Recorder(const RecorderOptions & options)
: init_(false),
  stop_thread_(false),
  options_(options),
  fps_(options.fps > 0.0 ? options.fps : 30.0),
  queue_(1)
{
  start_time_ = std::chrono::steady_clock::now();
  last_time_ = start_time_;

  configure_paths();
}

Recorder::~Recorder()
{
  stop_recording();
}

void Recorder::configure_paths()
{
  const std::filesystem::path records_root{"records"};

  if (options_.output_mode == RecorderOutputMode::legacy_pair) {
    const std::string name = options_.rotation_slot_count > 0 ? date_string() : timestamp_string();
    std::filesystem::path prefix =
      options_.output_path.empty() ? (records_root / name) : std::filesystem::path(options_.output_path);
    if (prefix.has_parent_path()) std::filesystem::create_directories(prefix.parent_path());

    if (options_.rotation_slot_count > 0) {
      const auto directory = prefix.has_parent_path() ? prefix.parent_path() : std::filesystem::path{"."};
      rotation_slot_ = next_rotation_slot(directory, options_.rotation_slot_count);
      prefix = prefix.parent_path() /
               fmt::format("{}_{}", prefix.filename().string(), rotation_slot_);
    }

    text_path_ = fmt::format("{}.txt", prefix.string());
    video_path_ = fmt::format("{}.avi", prefix.string());
    return;
  }

  const std::string name = timestamp_string();
  std::filesystem::path session_dir =
    options_.output_path.empty() ? (records_root / name) : std::filesystem::path(options_.output_path);
  std::filesystem::create_directories(session_dir);
  video_path_ = (session_dir / "video.avi").string();
  imu_csv_path_ = (session_dir / "imu.csv").string();
  metadata_path_ = (session_dir / "metadata.yaml").string();
}

void Recorder::clear_rotation_slot()
{
  if (rotation_slot_ == 0) return;

  const auto directory = std::filesystem::path(video_path_).parent_path();
  std::error_code error;
  std::vector<std::filesystem::path> files_to_remove;
  for (const auto & entry : std::filesystem::directory_iterator(directory, error)) {
    if (error) {
      tools::logger()->warn("Cannot inspect recorder directory {}: {}", directory.string(), error.message());
      return;
    }
    if (!entry.is_regular_file(error)) {
      error.clear();
      continue;
    }
    if (entry.path().extension() != ".avi" && entry.path().extension() != ".txt") continue;

    const auto slot = rotation_slot_from_path(entry.path(), options_.rotation_slot_count);
    if (!slot || *slot != rotation_slot_) continue;

    files_to_remove.push_back(entry.path());
  }

  for (const auto & path : files_to_remove) {
    std::filesystem::remove(path, error);
    if (error) {
      tools::logger()->warn("Cannot remove old recording {}: {}", path.string(), error.message());
      error.clear();
    }
  }
}

bool Recorder::has_sufficient_available_space() const
{
  if (options_.min_available_space_bytes == 0) return true;

  const auto directory = std::filesystem::path(video_path_).parent_path();
  std::error_code error;
  const auto disk_space = std::filesystem::space(directory, error);
  if (error) {
    tools::logger()->warn(
      "Cannot check available recorder disk space at {}: {}; recording is disabled.",
      directory.string(), error.message());
    return false;
  }

  if (disk_space.available >= options_.min_available_space_bytes) return true;

  constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
  tools::logger()->warn(
    "Recorder disk space is too low ({:.2f} GiB available; {:.2f} GiB required); recording is disabled.",
    static_cast<double>(disk_space.available) / kBytesPerGiB,
    static_cast<double>(options_.min_available_space_bytes) / kBytesPerGiB);
  return false;
}

void Recorder::stop_recording()
{
  stop_thread_ = true;
  queue_.clear();
  queue_.push({});
  if (saving_thread_.joinable()) saving_thread_.join();

  if (!init_) return;
  text_writer_.close();
  imu_writer_.close();
  video_writer_.release();
  init_ = false;
}

void Recorder::write_metadata_file() const
{
  if (options_.output_mode != RecorderOutputMode::session_dir) return;

  std::ofstream metadata_writer(metadata_path_);
  metadata_writer << "config_path: \"" << escape_yaml_double_quotes(options_.config_path) << "\"\n";
  metadata_writer << "fps: " << fps_ << "\n";
  metadata_writer << "record_version: 2\n";
}

void Recorder::save_to_file()
{
  while (true) {
    FrameData frame;
    queue_.pop(frame);  // 从队列中取出帧数据
    if (stop_thread_) break;
    if (frame.img.empty()) {
      tools::logger()->debug("Recorder received empty img. Skip this frame.");
      continue;
    }
    // 写入视频文件
    video_writer_.write(frame.img);

    Eigen::Vector4d xyzw = frame.q.coeffs();
    auto since_begin = tools::delta_time(frame.timestamp, start_time_);
    if (options_.output_mode == RecorderOutputMode::legacy_pair) {
      text_writer_ << fmt::format(
        "{} {} {} {} {}\n", since_begin, xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
      continue;
    }

    const int frame_index =
      frame.metadata.frame_index >= 0 ? frame.metadata.frame_index : next_frame_index_;
    next_frame_index_ = std::max(next_frame_index_, frame_index + 1);
    imu_writer_ << fmt::format(
      "{},{},{},{},{},{},{},{}\n", frame_index, since_begin, xyzw[3], xyzw[0], xyzw[1], xyzw[2],
      frame.metadata.bullet_speed, frame.metadata.vision_mode);
  }
}

void Recorder::record(
  const cv::Mat & img, const Eigen::Quaterniond & q,
  const std::chrono::steady_clock::time_point & timestamp)
{
  record(img, q, timestamp, {});
}

void Recorder::record(
  const cv::Mat & img, const Eigen::Quaterniond & q,
  const std::chrono::steady_clock::time_point & timestamp,
  const RecordSampleMetadata & metadata)
{
  if (img.empty() || recording_disabled_) return;
  if (!has_sufficient_available_space()) {
    recording_disabled_ = true;
    stop_recording();
    return;
  }
  if (!init_) init(img);

  auto since_last = tools::delta_time(timestamp, last_time_);
  if (since_last < 1.0 / fps_) return;

  last_time_ = timestamp;
  queue_.push({img, q, timestamp, metadata});
}

void Recorder::init(const cv::Mat & img)
{
  clear_rotation_slot();

  if (options_.output_mode == RecorderOutputMode::legacy_pair) {
    text_writer_.open(text_path_);
  } else {
    imu_writer_.open(imu_csv_path_);
    imu_writer_ << "frame_index,t_capture_sec,q_w,q_x,q_y,q_z,bullet_speed,vision_mode\n";
    write_metadata_file();
  }
  auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  video_writer_ = cv::VideoWriter(video_path_, fourcc, fps_, img.size());
  saving_thread_ = std::thread(&Recorder::save_to_file, this);  // 启动保存线程
  init_ = true;
}

}  // namespace tools
