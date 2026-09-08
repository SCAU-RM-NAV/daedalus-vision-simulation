#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <thread>

#include "io/sim/talos_ipc.hpp"

namespace
{
std::optional<io::sim::FramePacket> wait_for_frame(
  io::sim::TalosClient & client, std::chrono::milliseconds timeout)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto frame = client.try_read_frame()) return frame;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return std::nullopt;
}

std::optional<std::pair<float, float>> world_yaw_pitch(const std::array<float, 4> & wxyz)
{
  const auto squared_norm =
    wxyz[0] * wxyz[0] + wxyz[1] * wxyz[1] + wxyz[2] * wxyz[2] + wxyz[3] * wxyz[3];
  if (!std::isfinite(squared_norm) || squared_norm < 1e-8F) return std::nullopt;

  const auto inverse_norm = 1.0F / std::sqrt(squared_norm);
  const auto w = wxyz[0] * inverse_norm;
  const auto x = wxyz[1] * inverse_norm;
  const auto y = wxyz[2] * inverse_norm;
  const auto z = wxyz[3] * inverse_norm;
  const auto r00 = 1.0F - 2.0F * (y * y + z * z);
  const auto r10 = 2.0F * (x * y + w * z);
  const auto r20 = 2.0F * (x * z - w * y);
  return std::pair<float, float>{std::atan2(r10, r00), std::atan2(-r20, std::hypot(r00, r10))};
}

std::optional<io::sim::FramePacket> send_and_wait(
  io::sim::TalosClient & client, const io::sim::FramePacket & frame, std::uint64_t command_seq,
  float world_yaw, float world_pitch)
{
  const io::sim::VisionCommand command{
    frame.data.image.frame_seq,
    command_seq,
    frame.data.image.timestamp_ns,
    world_yaw,
    world_pitch,
    0.0F,
    0.0F,
    0.0F,
    0.0F,
    0.0F,
    1,
    0,
    {}};
  if (!client.send_command(command)) return std::nullopt;

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  std::optional<io::sim::FramePacket> applied;
  while (std::chrono::steady_clock::now() < deadline) {
    auto latest = wait_for_frame(client, std::chrono::milliseconds(50));
    if (!latest.has_value()) continue;
    if (latest->data.feedback.last_command_seq >= command_seq) applied = std::move(latest);
  }
  return applied;
}
}  // namespace

int main(int argc, char * argv[])
{
  const float pitch_delta_rad = argc > 1 ? std::strtof(argv[1], nullptr) : 0.1F;
  if (!std::isfinite(pitch_delta_rad) || std::abs(pitch_delta_rad) < 1e-4F) {
    std::cerr << "pitch delta must be finite and non-zero\n";
    return 2;
  }

  io::sim::TalosClient client;
  if (!client.connected()) {
    std::cerr << "cannot connect to Talos IPC: " << client.error() << '\n';
    return 3;
  }
  auto initial = wait_for_frame(client, std::chrono::seconds(2));
  if (!initial.has_value()) {
    std::cerr << "timed out waiting for a Talos frame\n";
    return 4;
  }
  if (initial->data.feedback.simulation_subscription_enabled == 0) {
    std::cerr << "Talos subscription is disabled; focus Daedalus and press F5\n";
    return 5;
  }
  const auto angles = world_yaw_pitch(initial->data.gimbal_world.quaternion_wxyz);
  if (!angles.has_value()) {
    std::cerr << "invalid gimbal world quaternion\n";
    return 6;
  }

  constexpr std::uint64_t kHoldCommandSeq = 10'001;
  constexpr std::uint64_t kPitchCommandSeq = 10'002;
  auto held = send_and_wait(client, *initial, kHoldCommandSeq, angles->first, angles->second);
  if (!held.has_value()) {
    std::cerr << "hold command was not acknowledged\n";
    return 7;
  }
  auto moved =
    send_and_wait(client, *held, kPitchCommandSeq, angles->first, angles->second + pitch_delta_rad);
  if (!moved.has_value()) {
    std::cerr << "pitch command was not acknowledged\n";
    return 8;
  }

  const auto feedback_delta = moved->data.feedback.pitch_rad - held->data.feedback.pitch_rad;
  std::cout << "hold_pitch_rad=" << held->data.feedback.pitch_rad
            << " moved_pitch_rad=" << moved->data.feedback.pitch_rad
            << " feedback_delta_rad=" << feedback_delta << '\n';
  if (std::abs(feedback_delta) < std::abs(pitch_delta_rad) * 0.2F) {
    std::cerr << "world pitch command did not produce a measurable local gimbal response\n";
    return 9;
  }
  return 0;
}
