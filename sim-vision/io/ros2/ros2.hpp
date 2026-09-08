#ifndef IO__ROS2_HPP
#define IO__ROS2_HPP

#include "publish2nav.hpp"
#include "subscribe2nav.hpp"
#include "tracker_publisher.hpp"

namespace io
{
class ROS2
{
public:
  ROS2();

  ~ROS2();

  void publish(const Eigen::Vector4d & target_pos);
  void publish_auto_aim_plan(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);
  void publish_omni_target(int status);
  void publish_tracker_target(
    const Eigen::Vector3d & position, int armor_id, float yaw = 0.0F, float pitch = 0.0F);

  std::vector<int8_t> subscribe_enemy_status();

  std::vector<int8_t> subscribe_autoaim_target();
  std::optional<NavGimbalState> subscribe_gimbal_state();
  std::optional<Eigen::Quaterniond> subscribe_gimbal_q(std::chrono::steady_clock::time_point t);

  template <typename T>
  std::shared_ptr<rclcpp::Publisher<T>> create_publisher(
    const std::string & node_name, const std::string & topic_name, size_t queue_size)
  {
    auto node = std::make_shared<rclcpp::Node>(node_name);

    auto publisher = node->create_publisher<T>(topic_name, queue_size);

    // 运行一个单独的线程来 spin 这个节点，确保消息可以被正确发布
    std::thread([node]() { rclcpp::spin(node); }).detach();

    return publisher;
  }

private:
  std::shared_ptr<Publish2Nav> publish2nav_;
  std::shared_ptr<Subscribe2Nav> subscribe2nav_;
  std::shared_ptr<TrackerPublisher> tracker_publisher_;

  std::unique_ptr<std::thread> publish_spin_thread_;
  std::unique_ptr<std::thread> subscribe_spin_thread_;
  std::unique_ptr<std::thread> tracker_spin_thread_;
};

}  // namespace io
#endif
