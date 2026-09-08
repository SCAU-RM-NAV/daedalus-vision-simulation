#include <cassert>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "io/sim/talos_ipc.hpp"

int main()
{
  static_assert(sizeof(io::sim::ShmHeader) == 64);
  static_assert(sizeof(io::sim::FrameData) == 3136);
  static_assert(sizeof(io::sim::VisionCommand) == 64);
  static_assert(sizeof(io::sim::GimbalFeedback) == 64);
  static_assert(sizeof(io::sim::GroundTruthTarget) == 128);
  static_assert(static_cast<std::uint8_t>(io::sim::RobotType::outpost) == 2);
  static_assert(static_cast<std::uint8_t>(io::sim::RobotType::base) == 3);

  const auto unix_now_ns = [] {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  }();

  io::sim::FrameData frame{};
  frame.image.frame_seq = 42;
  frame.image.timestamp_ns = unix_now_ns;
  frame.image.width = io::sim::kImageWidth;
  frame.image.height = io::sim::kImageHeight;
  frame.camera.frame_seq = frame.image.frame_seq;
  frame.camera.timestamp_ns = frame.image.timestamp_ns;
  frame.gimbal_world.frame_seq = frame.image.frame_seq;
  frame.gimbal_world.timestamp_ns = frame.image.timestamp_ns;
  frame.feedback.frame_seq = frame.image.frame_seq;
  frame.feedback.timestamp_ns = frame.image.timestamp_ns;
  frame.truth.frame_seq = frame.image.frame_seq;
  frame.truth.timestamp_ns = frame.image.timestamp_ns;
  assert(io::sim::validate_frame(frame));

  frame.gimbal_world.position_m = {1.0F, -2.0F, 0.5F};
  const auto relative = io::sim::world_position_relative_to_gimbal(
    {4.0F, 3.0F, 2.5F}, frame.gimbal_world);
  assert((relative == std::array<float, 3>{3.0F, 5.0F, 2.0F}));

  frame.feedback.timestamp_ns += 1;
  assert(!io::sim::validate_frame(frame));

  io::sim::UnixSteadyClockMapper mapper;
  const auto first = mapper.map(unix_now_ns - 1'000'000'000ULL);
  const auto second = mapper.map(unix_now_ns - 975'000'000ULL);
  if (!first.has_value() || !second.has_value() ||
      *second - *first != std::chrono::milliseconds(25)) {
    std::cerr << "UnixSteadyClockMapper did not preserve timestamp deltas\n";
    return 1;
  }
  const auto mapped_age = std::chrono::steady_clock::now() - *first;
  if (mapped_age < std::chrono::milliseconds(800) || mapped_age > std::chrono::milliseconds(1200)) {
    std::cerr << "UnixSteadyClockMapper did not preserve capture age\n";
    return 1;
  }
  if (mapper.map(unix_now_ns - 1'001'000'000ULL).has_value()) {
    std::cerr << "UnixSteadyClockMapper accepted a non-monotonic timestamp\n";
    return 1;
  }

  io::sim::VisionCommand command{};
  command.frame_seq = 42;
  command.command_seq = 7;
  command.control = 1;
  command.fire = 1;
  command.yaw_rad = 0.2F;
  command.pitch_rad = -0.1F;
  assert(io::sim::validate_command(command));
  command.command_seq = 0;
  assert(!io::sim::validate_command(command));

  io::sim::FrameData localize_frame{};
  localize_frame.feedback.yaw_rad = 0.2F;
  localize_frame.feedback.pitch_rad = -0.1F;
  localize_frame.gimbal_world.quaternion_wxyz = {1.0F, 0.0F, 0.0F, 0.0F};
  command.command_seq = 8;
  command.yaw_rad = 1.1F;
  command.pitch_rad = 0.3F;
  command.fire = 1;
  const auto localized = io::sim::localize_world_command(command, localize_frame, false);
  assert(std::abs(localized.yaw_rad - 1.1F) < 1e-5F);
  assert(std::abs(localized.pitch_rad - 0.3F) < 1e-5F);
  assert(localized.fire == 0);
  assert(io::sim::localize_world_command(command, localize_frame, true).fire == 1);

  const auto prefix = "/tmp/talos_ipc_test_" + std::to_string(getpid());
  const auto meta_path = prefix + "_meta";
  const auto image_path = prefix + "_image";
  {
    std::ofstream meta(meta_path, std::ios::binary | std::ios::trunc);
    meta.seekp(static_cast<std::streamoff>(sizeof(io::sim::ShmMetaRegion) - 1));
    meta.put('\0');
    std::ofstream image(image_path, std::ios::binary | std::ios::trunc);
    image.seekp(static_cast<std::streamoff>(io::sim::kImagePoolSize - 1));
    image.put('\0');
  }

  const auto meta_fd = open(meta_path.c_str(), O_RDWR);
  const auto image_fd = open(image_path.c_str(), O_RDWR);
  assert(meta_fd >= 0 && image_fd >= 0);
  auto * meta = static_cast<io::sim::ShmMetaRegion *>(
    mmap(nullptr, sizeof(io::sim::ShmMetaRegion), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd, 0));
  auto * image = static_cast<std::uint8_t *>(
    mmap(nullptr, io::sim::kImagePoolSize, PROT_READ | PROT_WRITE, MAP_SHARED, image_fd, 0));
  assert(meta != MAP_FAILED && image != MAP_FAILED);
  std::memset(meta, 0, sizeof(*meta));
  meta->header = {
    io::sim::kShmMagic,
    io::sim::kShmVersion,
    static_cast<std::uint32_t>(sizeof(io::sim::ShmMetaRegion)),
    io::sim::kImageWidth,
    io::sim::kImageHeight,
    io::sim::kImageChannels,
    1,
    1,
    {}};
  meta->frame.state = 1;
  meta->frame.write_index = 0;
  meta->frame.read_index = 2;
  meta->command.state = 1;
  meta->command.write_index = 0;
  meta->command.read_index = 2;
  frame.feedback.timestamp_ns = frame.image.timestamp_ns;
  frame.feedback.frame_seq = frame.image.frame_seq;
  frame.camera.intrinsics = {1000.0F, 1000.0F, 720.0F, 540.0F};
  frame.camera.R_camera2gimbal_row_major = {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
  frame.gimbal_world.quaternion_wxyz = {1.0F, 0.0F, 0.0F, 0.0F};
  meta->frame.slots[0] = frame;
  image[0] = 10;
  image[1] = 20;
  image[2] = 30;
  __atomic_store_n(&meta->frame.state, static_cast<std::uint8_t>(io::sim::kTripleNew), __ATOMIC_RELEASE);

  io::sim::TalosClient client(meta_path, image_path);
  assert(client.connected());
  auto packet = client.try_read_frame();
  assert(packet.has_value());
  assert(packet->image_bgr.at<cv::Vec3b>(0, 0) == cv::Vec3b(30, 20, 10));
  assert(meta->frame.consumer_ack_frame_seq == 42);

  auto command_to_send = command;
  command_to_send.command_seq = 9;
  command_to_send.timestamp_ns = frame.image.timestamp_ns;
  assert(client.send_command(command_to_send));
  assert((meta->command.state & io::sim::kTripleNew) != 0);
  assert(meta->command.slots[0].command_seq == command_to_send.command_seq);

  meta->hit_events.events[1] = {1, 9, frame.image.timestamp_ns, 3, -1, io::sim::HitType::armor, 1, 1, {}};
  __atomic_store_n(&meta->hit_events.write_seq, 1ULL, __ATOMIC_RELEASE);
  const auto events = client.read_hit_events();
  assert(events.size() == 1);
  assert(events.front().command_seq == 9);
  assert(events.front().correct == 1);

  munmap(meta, sizeof(io::sim::ShmMetaRegion));
  munmap(image, io::sim::kImagePoolSize);
  close(meta_fd);
  close(image_fd);
  std::filesystem::remove(meta_path);
  std::filesystem::remove(image_path);

  const auto publisher_meta_path = prefix + "_publisher_meta";
  const auto publisher_image_path = prefix + "_publisher_image";
  {
    io::sim::TalosPublisher publisher(publisher_meta_path, publisher_image_path);
    assert(publisher.connected());

    io::sim::FrameData published_frame{};
    published_frame.image.frame_seq = 1;
    published_frame.image.timestamp_ns = unix_now_ns - 2'000'000'000ULL;
    published_frame.image.width = io::sim::kImageWidth;
    published_frame.image.height = io::sim::kImageHeight;
    published_frame.image.format = 0;
    published_frame.camera.frame_seq = published_frame.image.frame_seq;
    published_frame.camera.timestamp_ns = published_frame.image.timestamp_ns;
    published_frame.camera.width = io::sim::kImageWidth;
    published_frame.camera.height = io::sim::kImageHeight;
    published_frame.camera.intrinsics = {1000.0F, 1000.0F, 720.0F, 540.0F};
    published_frame.camera.R_camera2gimbal_row_major = {
      1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    published_frame.gimbal_world.frame_seq = published_frame.image.frame_seq;
    published_frame.gimbal_world.timestamp_ns = published_frame.image.timestamp_ns;
    published_frame.gimbal_world.quaternion_wxyz = {1.0F, 0.0F, 0.0F, 0.0F};
    published_frame.feedback.frame_seq = published_frame.image.frame_seq;
    published_frame.feedback.timestamp_ns = published_frame.image.timestamp_ns;
    published_frame.feedback.robot_type = io::sim::RobotType::outpost;
    published_frame.truth.frame_seq = published_frame.image.frame_seq;
    published_frame.truth.timestamp_ns = published_frame.image.timestamp_ns;
    published_frame.truth.target_count = 1;
    published_frame.truth.targets[0].frame_seq = published_frame.image.frame_seq;
    published_frame.truth.targets[0].timestamp_ns = published_frame.image.timestamp_ns;
    published_frame.truth.targets[0].robot_type =
      static_cast<std::uint8_t>(io::sim::RobotType::outpost);

    cv::Mat rgb(
      static_cast<int>(io::sim::kImageHeight), static_cast<int>(io::sim::kImageWidth), CV_8UC3,
      cv::Scalar(11, 22, 33));
    assert(publisher.publish_frame(published_frame, rgb));
    cv::Mat bad_rgb(1, 1, CV_8UC1, cv::Scalar(0));
    assert(!publisher.publish_frame(published_frame, bad_rgb));

    io::sim::TalosClient publisher_client(publisher_meta_path, publisher_image_path);
    assert(publisher_client.connected());
    if (publisher_client.try_read_frame().has_value()) {
      std::cerr << "TalosClient accepted a stale frame\n";
      return 1;
    }

    const auto fresh_timestamp_ns = unix_now_ns + 1;
    published_frame.image.frame_seq = 2;
    published_frame.image.timestamp_ns = fresh_timestamp_ns;
    published_frame.camera.frame_seq = published_frame.image.frame_seq;
    published_frame.camera.timestamp_ns = fresh_timestamp_ns;
    published_frame.gimbal_world.frame_seq = published_frame.image.frame_seq;
    published_frame.gimbal_world.timestamp_ns = fresh_timestamp_ns;
    published_frame.feedback.frame_seq = published_frame.image.frame_seq;
    published_frame.feedback.timestamp_ns = fresh_timestamp_ns;
    published_frame.truth.frame_seq = published_frame.image.frame_seq;
    published_frame.truth.timestamp_ns = fresh_timestamp_ns;
    published_frame.truth.targets[0].frame_seq = published_frame.image.frame_seq;
    published_frame.truth.targets[0].timestamp_ns = fresh_timestamp_ns;
    if (!publisher.publish_frame(published_frame, rgb)) {
      std::cerr << "TalosPublisher rejected the fresh frame\n";
      return 1;
    }

    auto published_packet = publisher_client.try_read_frame();
    assert(published_packet.has_value());
    assert(published_packet->data.feedback.robot_type == io::sim::RobotType::outpost);
    assert(published_packet->image_bgr.at<cv::Vec3b>(0, 0) == cv::Vec3b(33, 22, 11));

    io::sim::VisionCommand received_command{};
    received_command.frame_seq = published_frame.image.frame_seq;
    received_command.command_seq = 1;
    received_command.timestamp_ns = published_frame.image.timestamp_ns;
    received_command.control = 1;
    received_command.distance_m = 5.0F;
    assert(publisher_client.send_command(received_command));
    const auto command_from_client = publisher.try_read_command();
    assert(command_from_client.has_value());
    assert(command_from_client->command_seq == received_command.command_seq);

    io::sim::HitEvent hit{};
    hit.command_seq = received_command.command_seq;
    hit.hit_timestamp_ns = published_frame.image.timestamp_ns + 1;
    hit.target_id = 7;
    hit.hit_type = io::sim::HitType::armor;
    hit.correct = 1;
    hit.outcome = 2;
    assert(publisher.publish_hit_event(hit));
    const auto published_hits = publisher_client.read_hit_events();
    assert(published_hits.size() == 1);
    assert(published_hits.front().event_seq == 1);
    assert(published_hits.front().target_id == hit.target_id);
  }
  std::filesystem::remove(publisher_meta_path);
  std::filesystem::remove(publisher_image_path);
  return 0;
}
