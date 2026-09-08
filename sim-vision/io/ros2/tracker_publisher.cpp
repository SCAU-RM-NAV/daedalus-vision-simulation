#include "tracker_publisher.hpp"

namespace io
{
TrackerPublisher::TrackerPublisher() : Node("tracker")
{
  publisher_ = this->create_publisher<robot_msgs::msg::Tracker>("tracker", 10);
  RCLCPP_INFO(this->get_logger(), "tracker publisher initialized.");
}

TrackerPublisher::~TrackerPublisher()
{
  RCLCPP_INFO(this->get_logger(), "tracker publisher shutting down.");
}

void TrackerPublisher::start()
{
  RCLCPP_INFO(this->get_logger(), "tracker publisher starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

void TrackerPublisher::send_target(
  const Eigen::Vector3d & position, int armor_id, float yaw, float pitch)
{
  if (!position.allFinite()) {
    RCLCPP_WARN(this->get_logger(), "Ignoring tracker target with non-finite camera coordinates.");
    return;
  }

  robot_msgs::msg::Tracker message;
  message.header.stamp = this->get_clock()->now();
  message.header.frame_id = "camera";
  message.x = position.x();
  message.y = position.y();
  message.z = position.z();
  message.yaw = yaw;
  message.pitch = pitch;
  message.armor_id = armor_id;
  publisher_->publish(message);
}

}  // namespace io
