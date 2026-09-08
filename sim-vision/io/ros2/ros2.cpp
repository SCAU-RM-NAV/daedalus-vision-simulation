#include "ros2.hpp"
namespace io
{
ROS2::ROS2()
{
  rclcpp::init(0, nullptr);

  publish2nav_ = std::make_shared<Publish2Nav>();

  subscribe2nav_ = std::make_shared<Subscribe2Nav>();

  tracker_publisher_ = std::make_shared<TrackerPublisher>();

  publish_spin_thread_ = std::make_unique<std::thread>([this]() { publish2nav_->start(); });

  subscribe_spin_thread_ = std::make_unique<std::thread>([this]() { subscribe2nav_->start(); });

  tracker_spin_thread_ = std::make_unique<std::thread>([this]() { tracker_publisher_->start(); });
}

ROS2::~ROS2()
{
  rclcpp::shutdown();
  publish_spin_thread_->join();
  subscribe_spin_thread_->join();
  tracker_spin_thread_->join();
}

void ROS2::publish(const Eigen::Vector4d & target_pos) { publish2nav_->send_data(target_pos); }

void ROS2::publish_auto_aim_plan(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  publish2nav_->send_auto_aim_plan(
    control, fire, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
}

void ROS2::publish_omni_target(int status)
{
  publish2nav_->send_omni_target(status);
}

void ROS2::publish_tracker_target(
  const Eigen::Vector3d & position, int armor_id, float yaw, float pitch)
{
  tracker_publisher_->send_target(position, armor_id, yaw, pitch);
}

std::vector<int8_t> ROS2::subscribe_enemy_status()
{
  return subscribe2nav_->subscribe_enemy_status();
}

std::vector<int8_t> ROS2::subscribe_autoaim_target()
{
  return subscribe2nav_->subscribe_autoaim_target();
}

std::optional<NavGimbalState> ROS2::subscribe_gimbal_state()
{
  return subscribe2nav_->subscribe_gimbal_state();
}

std::optional<Eigen::Quaterniond> ROS2::subscribe_gimbal_q(std::chrono::steady_clock::time_point t)
{
  return subscribe2nav_->subscribe_gimbal_q(t);
}

}  // namespace io
