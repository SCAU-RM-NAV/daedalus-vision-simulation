#ifndef IO__TRACKER_PUBLISHER_HPP
#define IO__TRACKER_PUBLISHER_HPP

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <robot_msgs/msg/tracker.hpp>

namespace io
{
class TrackerPublisher : public rclcpp::Node
{
public:
  TrackerPublisher();
  ~TrackerPublisher();

  void start();
  void send_target(const Eigen::Vector3d & position, int armor_id, float yaw, float pitch);

private:
  rclcpp::Publisher<robot_msgs::msg::Tracker>::SharedPtr publisher_;
};

}  // namespace io

#endif  // IO__TRACKER_PUBLISHER_HPP
