#ifndef IO__SIM__TALOS_IPC_HPP
#define IO__SIM__TALOS_IPC_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace io::sim
{
inline constexpr std::uint32_t kShmMagic = 0x54414C05U;
inline constexpr std::uint32_t kShmVersion = 3U;
inline constexpr std::uint32_t kImageWidth = 1440U;
inline constexpr std::uint32_t kImageHeight = 1080U;
inline constexpr std::uint32_t kImageChannels = 3U;
inline constexpr std::size_t kImageSize =
  static_cast<std::size_t>(kImageWidth) * kImageHeight * kImageChannels;
inline constexpr std::size_t kImagePoolSize = kImageSize * 3U;
inline constexpr std::size_t kGroundTruthMaxTargets = 16U;
inline constexpr std::size_t kGroundTruthMaxRunes = 4U;
inline constexpr std::size_t kHitEventCapacity = 256U;
inline constexpr std::uint8_t kTripleNew = 0x80U;
inline constexpr std::uint8_t kTripleIndexMask = 0x03U;

inline constexpr const char * kDefaultMetaPath = "/tmp/talos_ipc_meta";
inline constexpr const char * kDefaultImagePoolPath = "/tmp/talos_ipc_image_pool";

// Values are part of the fixed v3 binary protocol. Only append new target kinds.
enum class RobotType : std::uint8_t { infantry = 0, hero = 1, outpost = 2, base = 3 };
enum class AlgorithmMode : std::uint8_t { idle = 0, auto_aim = 1, small_buff = 2, big_buff = 3 };
enum class HitType : std::uint8_t { armor = 0, rune = 1 };

struct alignas(64) ShmHeader
{
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t meta_bytes;
  std::uint32_t image_width;
  std::uint32_t image_height;
  std::uint32_t image_channels;
  std::uint64_t created_ns;
  std::uint64_t heartbeat_ns;
  std::array<std::uint8_t, 24> reserved{};
};
static_assert(sizeof(ShmHeader) == 64);

struct alignas(32) ImageMeta
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::uint8_t buffer_id;
  std::uint8_t format;
  std::array<std::uint8_t, 6> reserved{};
};
static_assert(sizeof(ImageMeta) == 32);

struct alignas(64) Pose
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::array<float, 3> position_m{};
  std::array<float, 4> quaternion_wxyz{};
  std::array<std::uint8_t, 20> reserved{};
};
static_assert(sizeof(Pose) == 64);

inline std::array<float, 3> world_position_relative_to_gimbal(
  const std::array<float, 3> & world_position_m, const Pose & gimbal_world)
{
  return {
    world_position_m[0] - gimbal_world.position_m[0],
    world_position_m[1] - gimbal_world.position_m[1],
    world_position_m[2] - gimbal_world.position_m[2]};
}

struct alignas(64) CameraCalibration
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t width;
  std::uint32_t height;
  std::array<float, 4> intrinsics{};  // fx, fy, cx, cy
  std::array<float, 8> distortion{};
  std::array<float, 9> R_camera2gimbal_row_major{};
  std::array<float, 3> t_camera2gimbal_m{};
  std::array<std::uint8_t, 136> reserved{};
};
static_assert(sizeof(CameraCalibration) == 256);

struct alignas(64) GimbalFeedback
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  float yaw_rad;
  float pitch_rad;
  float yaw_velocity_radps;
  float pitch_velocity_radps;
  float bullet_speed_mps;
  std::uint32_t projectile_count;
  std::uint8_t camp;
  RobotType robot_type;
  AlgorithmMode mode;
  std::uint8_t simulation_subscription_enabled;
  std::uint32_t reserved0;
  std::uint64_t last_command_seq;
  std::array<std::uint8_t, 8> reserved{};
};
static_assert(sizeof(GimbalFeedback) == 64);

struct alignas(64) GroundTruthTarget
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_id;
  std::uint8_t team;
  std::uint8_t armor_label;
  std::uint8_t robot_type;
  std::uint8_t is_outpost;
  std::array<float, 3> position_m{};
  std::array<float, 4> quaternion_wxyz{};
  std::array<float, 3> velocity_mps{};
  float yaw_rate_radps;
  std::array<std::uint8_t, 8> reserved{};
};
static_assert(sizeof(GroundTruthTarget) == 128);

struct alignas(64) GroundTruthRune
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t rune_id;
  std::uint8_t team;
  AlgorithmMode mode;
  std::uint8_t mechanism_state;
  std::int8_t direction;
  std::array<float, 3> center_m{};
  std::array<float, 4> quaternion_wxyz{};
  float radius_m;
  float angle_rad;
  float angular_velocity_radps;
  float sine_amplitude;
  float sine_omega;
  float sine_phase;
  float sine_offset;
  float relative_time_s;
  std::int32_t active_blade_id;
  std::array<std::uint8_t, 5> target_activations{};
  std::array<std::uint8_t, 35> reserved{};
};
static_assert(sizeof(GroundTruthRune) == 128);

struct alignas(64) GroundTruthBatch
{
  std::uint64_t frame_seq;
  std::uint64_t timestamp_ns;
  std::uint32_t target_count;
  std::uint32_t rune_count;
  std::array<GroundTruthTarget, kGroundTruthMaxTargets> targets{};
  std::array<GroundTruthRune, kGroundTruthMaxRunes> runes{};
  std::array<std::uint8_t, 48> reserved{};
};
static_assert(sizeof(GroundTruthBatch) == 2688);

struct alignas(64) FrameData
{
  ImageMeta image;
  CameraCalibration camera;
  Pose gimbal_world;
  GimbalFeedback feedback;
  GroundTruthBatch truth;
};
static_assert(sizeof(FrameData) == 3136);

struct alignas(64) FrameTripleBuffer
{
  std::uint8_t state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::array<std::uint8_t, 5> reserved0{};
  std::uint64_t consumer_ack_frame_seq;
  std::array<std::uint8_t, 48> reserved1{};
  std::array<FrameData, 3> slots{};
};
static_assert(sizeof(FrameTripleBuffer) == 9472);

struct alignas(64) VisionCommand
{
  std::uint64_t frame_seq;
  std::uint64_t command_seq;
  std::uint64_t timestamp_ns;
  float yaw_rad;
  float pitch_rad;
  float yaw_velocity_radps;
  float pitch_velocity_radps;
  float yaw_acceleration_radps2;
  float pitch_acceleration_radps2;
  float distance_m;
  std::uint8_t control;
  std::uint8_t fire;
  std::array<std::uint8_t, 10> reserved{};
};
static_assert(sizeof(VisionCommand) == 64);

struct alignas(64) CommandTripleBuffer
{
  std::uint8_t state;
  std::uint8_t write_index;
  std::uint8_t read_index;
  std::array<std::uint8_t, 61> reserved{};
  std::array<VisionCommand, 3> slots{};
};
static_assert(sizeof(CommandTripleBuffer) == 256);

struct alignas(64) HitEvent
{
  std::uint64_t event_seq;
  std::uint64_t command_seq;
  std::uint64_t hit_timestamp_ns;
  std::uint32_t target_id;
  std::int32_t blade_id;
  HitType hit_type;
  std::uint8_t correct;
  std::uint8_t outcome;
  std::uint8_t reserved0;
  std::array<std::uint8_t, 24> reserved{};
};
static_assert(sizeof(HitEvent) == 64);

struct alignas(64) HitEventRing
{
  std::uint64_t write_seq;
  std::uint64_t consumer_ack_seq;
  std::uint64_t overflow_count;
  std::array<std::uint8_t, 40> reserved{};
  std::array<HitEvent, kHitEventCapacity> events{};
};
static_assert(sizeof(HitEventRing) == 16448);

struct alignas(64) ShmMetaRegion
{
  ShmHeader header;
  FrameTripleBuffer frame;
  CommandTripleBuffer command;
  HitEventRing hit_events;
};
static_assert(sizeof(ShmMetaRegion) == 26240);
static_assert(offsetof(ShmMetaRegion, frame) == 64);
static_assert(offsetof(ShmMetaRegion, command) == 9536);
static_assert(offsetof(ShmMetaRegion, hit_events) == 9792);

struct FramePacket
{
  FrameData data{};
  cv::Mat image_bgr;
  std::chrono::steady_clock::time_point timestamp;
};

class UnixSteadyClockMapper
{
public:
  std::optional<std::chrono::steady_clock::time_point> map(std::uint64_t unix_timestamp_ns);

private:
  std::optional<std::uint64_t> unix_origin_ns_;
  std::optional<std::chrono::steady_clock::time_point> steady_origin_;
  std::uint64_t last_unix_ns_ = 0;
};

bool validate_frame(const FrameData & frame);
bool validate_command(const VisionCommand & command);
bool validate_hit_event(const HitEvent & event);

// Planner commands are world-frame yaw/pitch. Daedalus performs the world-to-local
// kinematic conversion while it has access to the actual chassis/gimbal hierarchy.
// This boundary only applies simulation fire gating and preserves the fixed v3 layout.
VisionCommand localize_world_command(
  VisionCommand world_command, const FrameData & frame, bool physical_fire_enabled);

class TalosPublisher
{
public:
  explicit TalosPublisher(
    std::string meta_path = kDefaultMetaPath, std::string image_pool_path = kDefaultImagePoolPath);
  ~TalosPublisher();

  TalosPublisher(const TalosPublisher &) = delete;
  TalosPublisher & operator=(const TalosPublisher &) = delete;

  bool connected() const;
  const std::string & error() const;
  bool publish_frame(FrameData frame, const cv::Mat & image_rgb);
  std::optional<VisionCommand> try_read_command();
  bool publish_hit_event(HitEvent event);

private:
  int meta_fd_ = -1;
  int image_fd_ = -1;
  ShmMetaRegion * meta_ = nullptr;
  std::uint8_t * image_pool_ = nullptr;
  std::string error_;
  std::uint64_t last_frame_seq_ = 0;
  std::uint64_t last_command_seq_ = 0;

  void close();
};

class TalosClient
{
public:
  explicit TalosClient(
    std::string meta_path = kDefaultMetaPath, std::string image_pool_path = kDefaultImagePoolPath);
  ~TalosClient();

  TalosClient(const TalosClient &) = delete;
  TalosClient & operator=(const TalosClient &) = delete;

  bool connected() const;
  const std::string & error() const;
  std::optional<FramePacket> try_read_frame();
  bool send_command(VisionCommand command);
  std::vector<HitEvent> read_hit_events();
  std::uint64_t hit_event_overflow_count() const;

private:
  int meta_fd_ = -1;
  int image_fd_ = -1;
  ShmMetaRegion * meta_ = nullptr;
  const std::uint8_t * image_pool_ = nullptr;
  std::size_t meta_map_bytes_ = 0;
  std::size_t image_map_bytes_ = 0;
  std::string error_;
  std::uint64_t last_frame_seq_ = 0;
  std::uint64_t last_command_seq_ = 0;
  std::uint64_t last_hit_event_seq_ = 0;
  UnixSteadyClockMapper clock_mapper_;

  void close();
  bool validate_header() const;
};
}  // namespace io::sim

#endif  // IO__SIM__TALOS_IPC_HPP
