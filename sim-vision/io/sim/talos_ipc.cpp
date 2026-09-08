#include "talos_ipc.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

namespace io::sim
{
namespace
{
constexpr std::uint64_t kMaximumFrameAgeNs = 500'000'000ULL;

bool finite(float value) { return std::isfinite(value); }

bool finite_array(const float * values, std::size_t count)
{
  return std::all_of(values, values + count, [](float value) { return finite(value); });
}

std::uint8_t load_u8(const std::uint8_t * address)
{
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_u8(std::uint8_t * address, std::uint8_t value)
{
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

bool compare_exchange_u8(std::uint8_t * address, std::uint8_t & expected, std::uint8_t desired)
{
  return __atomic_compare_exchange_n(
    address, &expected, desired, true, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

std::uint64_t load_u64(const std::uint64_t * address)
{
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

void store_u64(std::uint64_t * address, std::uint64_t value)
{
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

bool consume_frame_slot(FrameTripleBuffer & buffer, std::uint8_t & index)
{
  std::uint8_t expected = load_u8(&buffer.state);
  if ((expected & kTripleNew) == 0) return false;

  index = expected & kTripleIndexMask;
  const std::uint8_t desired = buffer.read_index;
  if (!compare_exchange_u8(&buffer.state, expected, desired)) return false;

  buffer.read_index = index;
  return true;
}

bool consume_command_slot(CommandTripleBuffer & buffer, std::uint8_t & index)
{
  std::uint8_t expected = load_u8(&buffer.state);
  if ((expected & kTripleNew) == 0) return false;

  index = expected & kTripleIndexMask;
  const std::uint8_t desired = buffer.read_index;
  if (!compare_exchange_u8(&buffer.state, expected, desired)) return false;

  buffer.read_index = index;
  return true;
}

void publish_command_slot(CommandTripleBuffer & buffer, const VisionCommand & command)
{
  const std::uint8_t write_index = buffer.write_index;
  buffer.slots.at(write_index) = command;
  const std::uint8_t old = __atomic_exchange_n(
    &buffer.state, static_cast<std::uint8_t>(write_index | kTripleNew), __ATOMIC_ACQ_REL);
  buffer.write_index = old & kTripleIndexMask;
}

void publish_frame_slot(FrameTripleBuffer & buffer, const FrameData & frame)
{
  const std::uint8_t write_index = buffer.write_index;
  buffer.slots.at(write_index) = frame;
  const std::uint8_t old = __atomic_exchange_n(
    &buffer.state, static_cast<std::uint8_t>(write_index | kTripleNew), __ATOMIC_ACQ_REL);
  buffer.write_index = old & kTripleIndexMask;
}

std::uint64_t system_clock_now_ns()
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

bool resize_file(int fd, std::size_t size)
{
  return fd >= 0 && ftruncate(fd, static_cast<off_t>(size)) == 0;
}

}  // namespace

std::optional<std::chrono::steady_clock::time_point> UnixSteadyClockMapper::map(
  std::uint64_t unix_timestamp_ns)
{
  if (!unix_origin_ns_.has_value()) {
    const auto unix_now_ns = system_clock_now_ns();
    const auto steady_now = std::chrono::steady_clock::now();
    unix_origin_ns_ = unix_timestamp_ns;
    last_unix_ns_ = unix_timestamp_ns;
    if (unix_timestamp_ns >= unix_now_ns) {
      steady_origin_ = steady_now + std::chrono::nanoseconds(unix_timestamp_ns - unix_now_ns);
    } else {
      steady_origin_ = steady_now - std::chrono::nanoseconds(unix_now_ns - unix_timestamp_ns);
    }
    return steady_origin_;
  }

  if (unix_timestamp_ns < last_unix_ns_) return std::nullopt;
  last_unix_ns_ = unix_timestamp_ns;
  const auto elapsed = std::chrono::nanoseconds(unix_timestamp_ns - *unix_origin_ns_);
  return *steady_origin_ + elapsed;
}

bool validate_frame(const FrameData & frame)
{
  const auto seq = frame.image.frame_seq;
  const auto timestamp = frame.image.timestamp_ns;
  if (seq == 0 || timestamp == 0 || frame.image.width != kImageWidth ||
      frame.image.height != kImageHeight || frame.image.buffer_id >= 3 || frame.image.format != 0)
    return false;
  if (frame.camera.frame_seq != seq || frame.camera.timestamp_ns != timestamp ||
      frame.camera.width != kImageWidth || frame.camera.height != kImageHeight ||
      frame.gimbal_world.frame_seq != seq || frame.gimbal_world.timestamp_ns != timestamp ||
      frame.feedback.frame_seq != seq || frame.feedback.timestamp_ns != timestamp ||
      frame.truth.frame_seq != seq || frame.truth.timestamp_ns != timestamp ||
      frame.truth.target_count > kGroundTruthMaxTargets || frame.truth.rune_count > kGroundTruthMaxRunes)
    return false;
  if (!finite_array(frame.camera.intrinsics.data(), frame.camera.intrinsics.size()) ||
      !finite_array(frame.camera.distortion.data(), frame.camera.distortion.size()) ||
      !finite_array(
        frame.camera.R_camera2gimbal_row_major.data(),
        frame.camera.R_camera2gimbal_row_major.size()) ||
      !finite_array(frame.camera.t_camera2gimbal_m.data(), frame.camera.t_camera2gimbal_m.size()) ||
      !finite_array(frame.gimbal_world.position_m.data(), frame.gimbal_world.position_m.size()) ||
      !finite_array(
        frame.gimbal_world.quaternion_wxyz.data(), frame.gimbal_world.quaternion_wxyz.size()))
    return false;
  return frame.camera.intrinsics[0] > 0.0F && frame.camera.intrinsics[1] > 0.0F;
}

bool validate_command(const VisionCommand & command)
{
  if (command.frame_seq == 0 || command.command_seq == 0 || command.timestamp_ns == 0 ||
      command.control > 1 || command.fire > 1)
    return false;
  const float values[] = {
    command.yaw_rad,
    command.pitch_rad,
    command.yaw_velocity_radps,
    command.pitch_velocity_radps,
    command.yaw_acceleration_radps2,
    command.pitch_acceleration_radps2,
    command.distance_m};
  return finite_array(values, std::size(values));
}

bool validate_hit_event(const HitEvent & event)
{
  return event.hit_timestamp_ns != 0 && event.correct <= 1 &&
         (event.hit_type == HitType::armor || event.hit_type == HitType::rune);
}

VisionCommand localize_world_command(
  VisionCommand world_command, const FrameData & frame, bool physical_fire_enabled)
{
  (void)frame;
  world_command.fire = static_cast<std::uint8_t>(
    world_command.control != 0 && world_command.fire != 0 && physical_fire_enabled);
  return world_command;
}

TalosPublisher::TalosPublisher(std::string meta_path, std::string image_pool_path)
{
  meta_fd_ = open(meta_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  if (meta_fd_ < 0 || !resize_file(meta_fd_, sizeof(ShmMetaRegion))) {
    error_ = "cannot create Talos metadata mmap: " + meta_path;
    close();
    return;
  }
  image_fd_ = open(image_pool_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  if (image_fd_ < 0 || !resize_file(image_fd_, kImagePoolSize)) {
    error_ = "cannot create Talos image mmap: " + image_pool_path;
    close();
    return;
  }

  meta_ = static_cast<ShmMetaRegion *>(
    mmap(nullptr, sizeof(ShmMetaRegion), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd_, 0));
  image_pool_ = static_cast<std::uint8_t *>(
    mmap(nullptr, kImagePoolSize, PROT_READ | PROT_WRITE, MAP_SHARED, image_fd_, 0));
  if (meta_ == MAP_FAILED || image_pool_ == MAP_FAILED) {
    meta_ = nullptr;
    image_pool_ = nullptr;
    error_ = "cannot mmap Talos v3 files for publishing";
    close();
    return;
  }

  std::memset(meta_, 0, sizeof(*meta_));
  std::memset(image_pool_, 0, kImagePoolSize);
  const auto now_ns = system_clock_now_ns();
  meta_->header = {
    kShmMagic,
    kShmVersion,
    static_cast<std::uint32_t>(sizeof(ShmMetaRegion)),
    kImageWidth,
    kImageHeight,
    kImageChannels,
    now_ns,
    now_ns,
    {}};
  meta_->frame.state = 2;
  meta_->frame.write_index = 0;
  meta_->frame.read_index = 2;
  meta_->command.state = 2;
  meta_->command.write_index = 0;
  meta_->command.read_index = 2;
}

TalosPublisher::~TalosPublisher() { close(); }

void TalosPublisher::close()
{
  if (meta_ != nullptr) {
    munmap(meta_, sizeof(ShmMetaRegion));
    meta_ = nullptr;
  }
  if (image_pool_ != nullptr) {
    munmap(image_pool_, kImagePoolSize);
    image_pool_ = nullptr;
  }
  if (meta_fd_ >= 0) {
    ::close(meta_fd_);
    meta_fd_ = -1;
  }
  if (image_fd_ >= 0) {
    ::close(image_fd_);
    image_fd_ = -1;
  }
}

bool TalosPublisher::connected() const { return meta_ != nullptr && image_pool_ != nullptr; }

const std::string & TalosPublisher::error() const { return error_; }

bool TalosPublisher::publish_frame(FrameData frame, const cv::Mat & image_rgb)
{
  if (!connected() || image_rgb.empty() || image_rgb.type() != CV_8UC3 ||
      image_rgb.cols != static_cast<int>(kImageWidth) ||
      image_rgb.rows != static_cast<int>(kImageHeight) ||
      frame.image.frame_seq <= last_frame_seq_)
    return false;

  const std::uint8_t write_index = meta_->frame.write_index;
  frame.image.buffer_id = write_index;
  if (!validate_frame(frame)) return false;

  auto * destination = image_pool_ + static_cast<std::size_t>(write_index) * kImageSize;
  if (image_rgb.isContinuous()) {
    std::memcpy(destination, image_rgb.data, kImageSize);
  } else {
    for (int row = 0; row < image_rgb.rows; ++row) {
      std::memcpy(
        destination + static_cast<std::size_t>(row) * kImageWidth * kImageChannels,
        image_rgb.ptr(row), kImageWidth * kImageChannels);
    }
  }
  publish_frame_slot(meta_->frame, frame);
  store_u64(&meta_->header.heartbeat_ns, frame.image.timestamp_ns);
  last_frame_seq_ = frame.image.frame_seq;
  return true;
}

std::optional<VisionCommand> TalosPublisher::try_read_command()
{
  if (!connected()) return std::nullopt;

  std::uint8_t slot_index = 0;
  if (!consume_command_slot(meta_->command, slot_index)) {
    return std::nullopt;
  }
  const auto command = meta_->command.slots.at(slot_index);
  if (!validate_command(command) || command.command_seq <= last_command_seq_) return std::nullopt;
  last_command_seq_ = command.command_seq;
  return command;
}

bool TalosPublisher::publish_hit_event(HitEvent event)
{
  if (!connected() || !validate_hit_event(event)) return false;

  const auto event_seq = load_u64(&meta_->hit_events.write_seq) + 1;
  event.event_seq = event_seq;
  meta_->hit_events.events.at(event_seq % kHitEventCapacity) = event;
  const auto acknowledged = load_u64(&meta_->hit_events.consumer_ack_seq);
  if (event_seq > acknowledged + kHitEventCapacity) {
    __atomic_fetch_add(&meta_->hit_events.overflow_count, 1ULL, __ATOMIC_ACQ_REL);
  }
  store_u64(&meta_->hit_events.write_seq, event_seq);
  return true;
}

TalosClient::TalosClient(std::string meta_path, std::string image_pool_path)
{
  meta_fd_ = open(meta_path.c_str(), O_RDWR);
  if (meta_fd_ < 0) {
    error_ = "cannot open Talos metadata mmap: " + meta_path;
    return;
  }
  image_fd_ = open(image_pool_path.c_str(), O_RDONLY);
  if (image_fd_ < 0) {
    error_ = "cannot open Talos image mmap: " + image_pool_path;
    close();
    return;
  }

  struct stat meta_stat {};
  struct stat image_stat {};
  if (fstat(meta_fd_, &meta_stat) != 0 || fstat(image_fd_, &image_stat) != 0 ||
      meta_stat.st_size < static_cast<off_t>(sizeof(ShmMetaRegion)) ||
      image_stat.st_size < static_cast<off_t>(kImagePoolSize)) {
    error_ = "Talos mmap file size does not match v3 layout";
    close();
    return;
  }

  meta_map_bytes_ = static_cast<std::size_t>(meta_stat.st_size);
  image_map_bytes_ = static_cast<std::size_t>(image_stat.st_size);
  meta_ = static_cast<ShmMetaRegion *>(
    mmap(nullptr, sizeof(ShmMetaRegion), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd_, 0));
  image_pool_ = static_cast<const std::uint8_t *>(
    mmap(nullptr, kImagePoolSize, PROT_READ, MAP_SHARED, image_fd_, 0));
  if (meta_ == MAP_FAILED || image_pool_ == MAP_FAILED) {
    meta_ = nullptr;
    image_pool_ = nullptr;
    error_ = "cannot mmap Talos v3 files";
    close();
    return;
  }
  if (!validate_header()) {
    error_ = "Talos mmap header is not protocol v3";
    close();
  }
}

TalosClient::~TalosClient() { close(); }

void TalosClient::close()
{
  if (meta_ != nullptr) {
    munmap(meta_, sizeof(ShmMetaRegion));
    meta_ = nullptr;
  }
  if (image_pool_ != nullptr) {
    munmap(const_cast<std::uint8_t *>(image_pool_), kImagePoolSize);
    image_pool_ = nullptr;
  }
  if (meta_fd_ >= 0) {
    ::close(meta_fd_);
    meta_fd_ = -1;
  }
  if (image_fd_ >= 0) {
    ::close(image_fd_);
    image_fd_ = -1;
  }
}

bool TalosClient::validate_header() const
{
  return meta_ != nullptr && meta_->header.magic == kShmMagic &&
         meta_->header.version == kShmVersion && meta_->header.meta_bytes == sizeof(ShmMetaRegion) &&
         meta_->header.image_width == kImageWidth && meta_->header.image_height == kImageHeight &&
         meta_->header.image_channels == kImageChannels;
}

bool TalosClient::connected() const { return meta_ != nullptr && image_pool_ != nullptr; }

const std::string & TalosClient::error() const { return error_; }

std::optional<FramePacket> TalosClient::try_read_frame()
{
  if (!connected()) return std::nullopt;

  std::uint8_t slot_index = 0;
  if (!consume_frame_slot(meta_->frame, slot_index)) return std::nullopt;

  FramePacket packet;
  packet.data = meta_->frame.slots.at(slot_index);
  if (!validate_frame(packet.data) || packet.data.image.frame_seq <= last_frame_seq_) {
    store_u64(&meta_->frame.consumer_ack_frame_seq, packet.data.image.frame_seq);
    return std::nullopt;
  }
  const auto now_ns = system_clock_now_ns();
  if (
    packet.data.image.timestamp_ns < now_ns &&
    now_ns - packet.data.image.timestamp_ns > kMaximumFrameAgeNs) {
    last_frame_seq_ = packet.data.image.frame_seq;
    store_u64(&meta_->frame.consumer_ack_frame_seq, last_frame_seq_);
    return std::nullopt;
  }
  const auto timestamp = clock_mapper_.map(packet.data.image.timestamp_ns);
  if (!timestamp.has_value()) {
    store_u64(&meta_->frame.consumer_ack_frame_seq, packet.data.image.frame_seq);
    return std::nullopt;
  }

  const auto * rgb = image_pool_ + static_cast<std::size_t>(packet.data.image.buffer_id) * kImageSize;
  cv::Mat shared_rgb(
    static_cast<int>(kImageHeight), static_cast<int>(kImageWidth), CV_8UC3,
    const_cast<std::uint8_t *>(rgb));
  cv::cvtColor(shared_rgb, packet.image_bgr, cv::COLOR_RGB2BGR);
  packet.timestamp = *timestamp;

  last_frame_seq_ = packet.data.image.frame_seq;
  // The publisher observes this only after the BGR buffer is private to this process.
  store_u64(&meta_->frame.consumer_ack_frame_seq, last_frame_seq_);
  return packet;
}

bool TalosClient::send_command(VisionCommand command)
{
  if (!connected() || !validate_command(command) || command.command_seq <= last_command_seq_) {
    return false;
  }
  publish_command_slot(meta_->command, command);
  last_command_seq_ = command.command_seq;
  return true;
}

std::vector<HitEvent> TalosClient::read_hit_events()
{
  std::vector<HitEvent> result;
  if (!connected()) return result;

  const auto published = load_u64(&meta_->hit_events.write_seq);
  if (published <= last_hit_event_seq_) return result;
  const auto first = std::max(
    last_hit_event_seq_ + 1,
    published >= kHitEventCapacity ? published - kHitEventCapacity + 1 : std::uint64_t{1});
  result.reserve(static_cast<std::size_t>(published - first + 1));
  for (auto seq = first; seq <= published; ++seq) {
    const auto event = meta_->hit_events.events.at(seq % kHitEventCapacity);
    if (event.event_seq == seq) result.push_back(event);
  }
  last_hit_event_seq_ = published;
  store_u64(&meta_->hit_events.consumer_ack_seq, published);
  return result;
}

std::uint64_t TalosClient::hit_event_overflow_count() const
{
  return connected() ? load_u64(&meta_->hit_events.overflow_count) : 0;
}
}  // namespace io::sim
