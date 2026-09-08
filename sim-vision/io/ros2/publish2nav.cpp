#include "publish2nav.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <thread>

#include "tools/logger.hpp"

namespace io
{

Publish2Nav::Publish2Nav() : Node("auto_aim_target_pos_publisher")
{
  publisher_ = this->create_publisher<std_msgs::msg::String>("auto_aim_target_pos", 10);
  auto_aim_plan_publisher_ =
    this->create_publisher<robot_msgs::msg::VisionCtrlNew>("sentry_auto_aim_plan", 10);
  omni_target_publisher_ =
    this->create_publisher<robot_msgs::msg::Feeling>("feeling", 10);

  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node initialized.");
}

Publish2Nav::~Publish2Nav()
{
  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node shutting down.");
}

void Publish2Nav::send_data(const Eigen::Vector4d & target_pos)
{
  // 创建消息
  auto message = std::make_shared<std_msgs::msg::String>();

  // 将 Eigen::Vector3d 数据转换为字符串并存储在消息中
  message->data = std::to_string(target_pos[0]) + "," + std::to_string(target_pos[1]) + "," +
                  std::to_string(target_pos[2]) + "," + std::to_string(target_pos[3]);

  // 发布消息
  publisher_->publish(*message);

  // RCLCPP_INFO(
  //   this->get_logger(), "auto_aim_target_pos_publisher node sent message: '%s'",
  //   message->data.c_str());
}

void Publish2Nav::send_auto_aim_plan(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  robot_msgs::msg::VisionCtrlNew message;
  message.mode = control ? (fire ? 2 : 1) : 0;
  message.yaw = yaw;
  message.yaw_vel = yaw_vel;
  message.yaw_acc = yaw_acc;
  message.pitch = pitch;
  message.pitch_vel = pitch_vel;
  message.pitch_acc = pitch_acc;

  auto_aim_plan_publisher_->publish(message);
}

void Publish2Nav::send_omni_target(int status)
{
  if (status != -1 && status !=1 && status !=0) {
    RCLCPP_WARN(this->get_logger(), "Invalid omni target status %d; publishing 0 instead.", status);
    status = 0;
  }

  robot_msgs::msg::Feeling message;
  // Feeling has no status field. is_detect carries 0=no target, 1=found, 2=priority stable.
  message.is_detect = static_cast<double>(status);

  omni_target_publisher_->publish(message);
}

void Publish2Nav::start()
{
  RCLCPP_INFO(this->get_logger(), "auto_aim_target_pos_publisher node starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

}  // namespace io
