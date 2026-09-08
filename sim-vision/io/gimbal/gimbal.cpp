#include "gimbal.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  auto brNode = yaml["baud_rate"];
  auto baud_rate = (brNode && !brNode.IsNull()) ? tools::read<int>(yaml, "baud_rate") : 115200;

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(baud_rate);  // Original use the normal link baud rate @ 115200
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial::Timeout timeout(serial::Timeout::max(), 2, 0, 20, 0);
    serial_.setTimeout(timeout);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  try {
    if (serial_.isOpen()) serial_.close();
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to close serial: {}", e.what());
  }
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  if (!serial_.isOpen()) return;

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  if (!serial_.isOpen()) return;

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::sendRaw(const uint8_t * data, size_t size)
{
  try {
    serial_.write(data, size);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::sendRaw(const std::string & data)
{
  try {
    serial_.write(data);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  auto last_valid_rx = std::chrono::steady_clock::now();
  const auto reconnect_timeout = std::chrono::milliseconds(500);
  const std::array<uint8_t, 2> packet_head{'S', 'P'};
  constexpr size_t packet_size = sizeof(GimbalToVision);
  constexpr size_t max_buffer_size = packet_size * 16;
  std::array<uint8_t, 256> chunk{};
  std::vector<uint8_t> rx_buffer;
  rx_buffer.reserve(packet_size * 4);

  auto apply_packet =
    [&](const GimbalToVision & packet, std::chrono::steady_clock::time_point timestamp) {
      rx_data_ = packet;
      last_valid_rx = std::chrono::steady_clock::now();

      Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
      queue_.push({q, timestamp});

      std::lock_guard<std::mutex> lock(mutex_);

      state_.yaw = rx_data_.yaw;
      state_.yaw_vel = rx_data_.yaw_vel;
      state_.pitch = rx_data_.pitch;
      state_.pitch_vel = rx_data_.pitch_vel;
      state_.bullet_speed = rx_data_.bullet_speed;
      state_.bullet_count = rx_data_.bullet_count;
      state_.camp = rx_data_.camp;

      switch (rx_data_.mode) {
        case 0:
          mode_ = GimbalMode::IDLE;
          break;
        case 1:
          mode_ = GimbalMode::AUTO_AIM;
          break;
        case 2:
          mode_ = GimbalMode::SMALL_BUFF;
          break;
        case 3:
          mode_ = GimbalMode::BIG_BUFF;
          break;
        default:
          mode_ = GimbalMode::IDLE;
          tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
          break;
      }
    };

  while (!quit_) {
    if (std::chrono::steady_clock::now() - last_valid_rx > reconnect_timeout) {
      tools::logger()->warn("[Gimbal] No valid packet for 500 ms, attempting to reconnect...");
      reconnect();
      last_valid_rx = std::chrono::steady_clock::now();
      rx_buffer.clear();
      continue;
    }

    if (!serial_.isOpen()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    size_t available = 0;
    try {
      available = serial_.available();
    } catch (const std::exception & e) {
      tools::logger()->debug("[Gimbal] Failed to query serial bytes: {}", e.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    if (available == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    size_t bytes_read = 0;
    const auto read_size = std::min(available, chunk.size());
    try {
      bytes_read = serial_.read(chunk.data(), read_size);
    } catch (const std::exception & e) {
      tools::logger()->debug("[Gimbal] Failed to read serial bytes: {}", e.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    if (bytes_read == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    rx_buffer.insert(rx_buffer.end(), chunk.begin(), chunk.begin() + bytes_read);

    while (rx_buffer.size() >= packet_head.size()) {
      auto head_it =
        std::search(rx_buffer.begin(), rx_buffer.end(), packet_head.begin(), packet_head.end());
      if (head_it == rx_buffer.end()) {
        if (rx_buffer.size() > packet_head.size() - 1) {
          rx_buffer.erase(rx_buffer.begin(), rx_buffer.end() - (packet_head.size() - 1));
        }
        break;
      }

      if (head_it != rx_buffer.begin()) {
        rx_buffer.erase(rx_buffer.begin(), head_it);
      }

      if (rx_buffer.size() < packet_size) {
        break;
      }

      GimbalToVision packet;
      std::memcpy(&packet, rx_buffer.data(), packet_size);
      if (tools::check_crc16(reinterpret_cast<uint8_t *>(&packet), packet_size)) {
        apply_packet(packet, std::chrono::steady_clock::now());
        rx_buffer.erase(rx_buffer.begin(), rx_buffer.begin() + packet_size);
      } else {
        tools::logger()->debug("[Gimbal] CRC16 check failed.");
        rx_buffer.erase(rx_buffer.begin());
      }
    }

    if (rx_buffer.size() > max_buffer_size) {
      tools::logger()->debug("[Gimbal] Serial RX buffer overflow, dropping stale bytes.");
      rx_buffer.erase(rx_buffer.begin(), rx_buffer.end() - (packet_head.size() - 1));
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      if (serial_.isOpen()) serial_.close();
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Failed to close serial before reconnect: {}", e.what());
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    try {
      serial_.open();  // 尝试重新打开
      serial_.flushInput();
      queue_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io
