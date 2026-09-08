#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "io/sim/talos_ipc.hpp"

namespace
{
bool create_sized_file(const std::string & path, std::size_t size)
{
  const int fd = open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
  if (fd < 0) return false;
  const bool ok = ftruncate(fd, static_cast<off_t>(size)) == 0;
  close(fd);
  return ok;
}
}  // namespace

int main(int argc, char * argv[])
{
  if (argc != 3) {
    std::cerr << "usage: talos_ipc_fixture <meta-path> <image-path>\n";
    return 2;
  }
  const std::string meta_path = argv[1];
  const std::string image_path = argv[2];
  if (!create_sized_file(meta_path, sizeof(io::sim::ShmMetaRegion)) ||
      !create_sized_file(image_path, io::sim::kImagePoolSize)) {
    return 3;
  }
  const int meta_fd = open(meta_path.c_str(), O_RDWR);
  const int image_fd = open(image_path.c_str(), O_RDWR);
  if (meta_fd < 0 || image_fd < 0) return 4;
  auto * meta = static_cast<io::sim::ShmMetaRegion *>(
    mmap(nullptr, sizeof(io::sim::ShmMetaRegion), PROT_READ | PROT_WRITE, MAP_SHARED, meta_fd, 0));
  auto * image = static_cast<std::uint8_t *>(
    mmap(nullptr, io::sim::kImagePoolSize, PROT_READ | PROT_WRITE, MAP_SHARED, image_fd, 0));
  if (meta == MAP_FAILED || image == MAP_FAILED) return 5;
  std::memset(meta, 0, sizeof(*meta));
  std::memset(image, 0, io::sim::kImagePoolSize);
  const auto timestamp_ns = [] {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  };
  meta->header = {
    io::sim::kShmMagic,
    io::sim::kShmVersion,
    static_cast<std::uint32_t>(sizeof(io::sim::ShmMetaRegion)),
    io::sim::kImageWidth,
    io::sim::kImageHeight,
    io::sim::kImageChannels,
    timestamp_ns(),
    timestamp_ns(),
    {}};
  meta->frame.state = 1;
  meta->frame.write_index = 0;
  meta->frame.read_index = 2;
  meta->command.state = 1;
  meta->command.write_index = 0;
  meta->command.read_index = 2;
  auto & frame = meta->frame.slots[0];
  frame.image = {1, timestamp_ns(), io::sim::kImageWidth, io::sim::kImageHeight, 0, 0, {}};
  frame.camera = {
    1,
    frame.image.timestamp_ns,
    io::sim::kImageWidth,
    io::sim::kImageHeight,
    {1000.0F, 1000.0F, 720.0F, 540.0F},
    {},
    {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F},
    {},
    {}};
  frame.gimbal_world = {1, frame.image.timestamp_ns, {}, {1.0F, 0.0F, 0.0F, 0.0F}, {}};
  frame.feedback = {
    1,
    frame.image.timestamp_ns,
    0.0F,
    0.0F,
    0.0F,
    0.0F,
    24.0F,
    0,
    0,
    io::sim::RobotType::infantry,
    io::sim::AlgorithmMode::idle,
    1,
    0,
    0,
    {}};
  frame.truth.frame_seq = 1;
  frame.truth.timestamp_ns = frame.image.timestamp_ns;
  __atomic_store_n(&meta->frame.state, static_cast<std::uint8_t>(io::sim::kTripleNew), __ATOMIC_RELEASE);

  bool acknowledged = false;
  for (int count = 0; count < 1200; ++count) {
    const auto frame_seq = static_cast<std::uint64_t>(count + 2);
    const auto fresh_timestamp_ns = timestamp_ns();
    frame.image.frame_seq = frame_seq;
    frame.image.timestamp_ns = fresh_timestamp_ns;
    frame.camera.frame_seq = frame_seq;
    frame.camera.timestamp_ns = fresh_timestamp_ns;
    frame.gimbal_world.frame_seq = frame_seq;
    frame.gimbal_world.timestamp_ns = fresh_timestamp_ns;
    frame.feedback.frame_seq = frame_seq;
    frame.feedback.timestamp_ns = fresh_timestamp_ns;
    frame.truth.frame_seq = frame_seq;
    frame.truth.timestamp_ns = fresh_timestamp_ns;
    __atomic_store_n(&meta->frame.state, static_cast<std::uint8_t>(io::sim::kTripleNew), __ATOMIC_RELEASE);

    if (__atomic_load_n(&meta->frame.consumer_ack_frame_seq, __ATOMIC_ACQUIRE) != 0) {
      acknowledged = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  munmap(meta, sizeof(io::sim::ShmMetaRegion));
  munmap(image, io::sim::kImagePoolSize);
  close(meta_fd);
  close(image_fd);
  std::filesystem::remove(meta_path);
  std::filesystem::remove(image_path);
  return acknowledged ? 0 : 6;
}
