#ifndef IO__PBLISH2NAV_HPP
#define IO__PBLISH2NAV_HPP

#include <Eigen/Dense>  // For Eigen::Vector3d
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "robot_msgs/msg/feeling.hpp"
#include "robot_msgs/msg/vision_ctrl_new.hpp"
#include "std_msgs/msg/string.hpp"

namespace io
{
class Publish2Nav : public rclcpp::Node
{
public:
  Publish2Nav();

  ~Publish2Nav();

  void start();

  void send_data(const Eigen::Vector4d & data);
  void send_auto_aim_plan(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);
  void send_omni_target(int status);

private:
  // ROS2 发布者
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Publisher<robot_msgs::msg::VisionCtrlNew>::SharedPtr auto_aim_plan_publisher_;
  rclcpp::Publisher<robot_msgs::msg::Feeling>::SharedPtr omni_target_publisher_;
};

}  // namespace io

#endif  // Publish2Nav_HPP_
