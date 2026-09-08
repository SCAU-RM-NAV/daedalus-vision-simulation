#ifndef IO__SUBSCRIBE2NAV_HPP
#define IO__SUBSCRIBE2NAV_HPP

#include <Eigen/Geometry>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <robot_msgs/msg/vision_sp_send.hpp>
#include <sp_msgs/msg/detail/autoaim_target_msg__struct.hpp>
#include <vector>

#include "sp_msgs/msg/autoaim_target_msg.hpp"
#include "sp_msgs/msg/enemy_status_msg.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct NavGimbalState
{
  uint8_t mode = 0;
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  float yaw = 0.0F;
  float yaw_vel = 0.0F;
  float pitch = 0.0F;
  float pitch_vel = 0.0F;
  float bullet_speed = 0.0F;
  uint16_t bullet_count = 0;
  uint8_t camp = 0xff;
  std::chrono::steady_clock::time_point stamp = std::chrono::steady_clock::time_point{};
};

class Subscribe2Nav : public rclcpp::Node
{
public:
  Subscribe2Nav();

  ~Subscribe2Nav();

  void start();

  std::vector<int8_t> subscribe_enemy_status();
  std::vector<int8_t> subscribe_autoaim_target();
  std::optional<NavGimbalState> subscribe_gimbal_state();
  std::optional<Eigen::Quaterniond> subscribe_gimbal_q(std::chrono::steady_clock::time_point t);

private:
  void enemy_status_callback(const sp_msgs::msg::EnemyStatusMsg::SharedPtr msg);
  void autoaim_target_callback(const sp_msgs::msg::AutoaimTargetMsg::SharedPtr msg);
  void gimbal_state_callback(const robot_msgs::msg::VisionSpSend::SharedPtr msg);

  int enemy_status_counter_;
  int autoaim_target_counter_;

  rclcpp::TimerBase::SharedPtr enemy_status_timer_;
  rclcpp::TimerBase::SharedPtr autoaim_target_timer_;

  rclcpp::Subscription<sp_msgs::msg::EnemyStatusMsg>::SharedPtr enemy_status_subscription_;
  rclcpp::Subscription<sp_msgs::msg::AutoaimTargetMsg>::SharedPtr autoaim_target_subscription_;
  rclcpp::Subscription<robot_msgs::msg::VisionSpSend>::SharedPtr gimbal_state_subscription_;

  tools::ThreadSafeQueue<sp_msgs::msg::EnemyStatusMsg> enemy_statue_queue_;
  tools::ThreadSafeQueue<sp_msgs::msg::AutoaimTargetMsg> autoaim_target_queue_;

  mutable std::mutex gimbal_state_mutex_;
  std::optional<NavGimbalState> latest_gimbal_state_;
  std::deque<NavGimbalState> gimbal_state_history_;
};
}  // namespace io

#endif  // IO__SUBSCRIBE2NAV_HPP
