#include "subscribe2nav.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <vector>

namespace io
{

Subscribe2Nav::Subscribe2Nav()
: Node("nav_subscriber"),
  enemy_statue_queue_(1),
  autoaim_target_queue_(1),
  enemy_status_counter_(0),
  autoaim_target_counter_(0)
{
  enemy_status_subscription_ = this->create_subscription<sp_msgs::msg::EnemyStatusMsg>(
    "enemy_status", 10,
    std::bind(&Subscribe2Nav::enemy_status_callback, this, std::placeholders::_1));

  autoaim_target_subscription_ = this->create_subscription<sp_msgs::msg::AutoaimTargetMsg>(
    "autoaim_target", 10,
    std::bind(&Subscribe2Nav::autoaim_target_callback, this, std::placeholders::_1));

  gimbal_state_subscription_ = this->create_subscription<robot_msgs::msg::VisionSpSend>(
    "sentry_gimbal_state", 10,
    std::bind(&Subscribe2Nav::gimbal_state_callback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(), "nav_subscriber node initialized.");
}

Subscribe2Nav::~Subscribe2Nav()
{
  RCLCPP_INFO(this->get_logger(), "nav_subscriber node shutting down.");
}

void Subscribe2Nav::enemy_status_callback(const sp_msgs::msg::EnemyStatusMsg::SharedPtr msg)
{
  enemy_statue_queue_.clear();
  enemy_statue_queue_.push(*msg);

  enemy_status_counter_++;

  if (enemy_status_counter_ >= 2) {
    if (enemy_status_timer_) {
      enemy_status_timer_->cancel();
    }
    enemy_status_timer_ = this->create_wall_timer(std::chrono::milliseconds(1500), [this]() {
      enemy_statue_queue_.clear();
      enemy_status_counter_ = 0;
      RCLCPP_INFO(
        this->get_logger(), "Enemy status queue cleared due to inactivity after two messages.");
    });
  }
}

void Subscribe2Nav::autoaim_target_callback(const sp_msgs::msg::AutoaimTargetMsg::SharedPtr msg)
{
  autoaim_target_queue_.clear();
  autoaim_target_queue_.push(*msg);

  autoaim_target_counter_++;

  if (autoaim_target_counter_ >= 2) {
    if (autoaim_target_timer_) {
      autoaim_target_timer_->cancel();
    }
    autoaim_target_timer_ = this->create_wall_timer(std::chrono::milliseconds(1500), [this]() {
      autoaim_target_queue_.clear();
      autoaim_target_counter_ = 0;
      RCLCPP_INFO(
        this->get_logger(), "Autoaim target queue cleared due to inactivity after two messages.");
    });
  }
}

void Subscribe2Nav::gimbal_state_callback(const robot_msgs::msg::VisionSpSend::SharedPtr msg)
{
  NavGimbalState state;
  Eigen::Quaterniond q(msg->q[0], msg->q[1], msg->q[2], msg->q[3]);
  if (!std::isfinite(q.norm()) || q.norm() < 1e-6) {
    RCLCPP_WARN(this->get_logger(), "Invalid quaternion in sentry_gimbal_state.");
    return;
  }

  state.mode = msg->mode;
  state.q = q.normalized();
  state.yaw = msg->yaw;
  state.yaw_vel = msg->yaw_vel;
  state.pitch = msg->pitch;
  state.pitch_vel = msg->pitch_vel;
  state.bullet_speed = msg->bullet_speed;
  state.bullet_count = msg->bullet_count;
  state.camp = msg->camp;
  state.stamp = std::chrono::steady_clock::now();

  std::lock_guard<std::mutex> lock(gimbal_state_mutex_);
  latest_gimbal_state_ = state;
  gimbal_state_history_.push_back(state);
  while (gimbal_state_history_.size() > 1000) {
    gimbal_state_history_.pop_front();
  }
}

void Subscribe2Nav::start()
{
  RCLCPP_INFO(this->get_logger(), "nav_subscriber node Starting to spin...");
  rclcpp::spin(this->shared_from_this());
}

std::vector<int8_t> Subscribe2Nav::subscribe_enemy_status()
{
  if (enemy_statue_queue_.empty()) {
    return std::vector<int8_t>();
  }
  sp_msgs::msg::EnemyStatusMsg msg;

  enemy_statue_queue_.back(msg);
  RCLCPP_INFO(
    this->get_logger(), "Subscribe enemy_status at: %d.%09u", msg.timestamp.sec,
    msg.timestamp.nanosec);

  return msg.invincible_enemy_ids;
}

std::vector<int8_t> Subscribe2Nav::subscribe_autoaim_target()
{
  if (autoaim_target_queue_.empty()) {
    return std::vector<int8_t>();
  }
  sp_msgs::msg::AutoaimTargetMsg msg;

  autoaim_target_queue_.back(msg);
  RCLCPP_INFO(
    this->get_logger(), "Subscribe autoaim_target at: %d.%09u", msg.timestamp.sec,
    msg.timestamp.nanosec);

  return msg.target_ids;
}

std::optional<NavGimbalState> Subscribe2Nav::subscribe_gimbal_state()
{
  std::lock_guard<std::mutex> lock(gimbal_state_mutex_);
  if (
    latest_gimbal_state_ &&
    std::chrono::steady_clock::now() - latest_gimbal_state_->stamp > std::chrono::milliseconds(500)) {
    return std::nullopt;
  }
  return latest_gimbal_state_;
}

std::optional<Eigen::Quaterniond> Subscribe2Nav::subscribe_gimbal_q(
  std::chrono::steady_clock::time_point t)
{
  std::lock_guard<std::mutex> lock(gimbal_state_mutex_);
  if (gimbal_state_history_.empty()) return std::nullopt;
  if (std::chrono::steady_clock::now() - gimbal_state_history_.back().stamp >
      std::chrono::milliseconds(500)) {
    return std::nullopt;
  }
  if (gimbal_state_history_.size() == 1) return gimbal_state_history_.back().q;

  auto upper = std::lower_bound(
    gimbal_state_history_.begin(), gimbal_state_history_.end(), t,
    [](const NavGimbalState & state, std::chrono::steady_clock::time_point stamp) {
      return state.stamp < stamp;
    });

  if (upper == gimbal_state_history_.begin()) return upper->q;
  if (upper == gimbal_state_history_.end()) return gimbal_state_history_.back().q;

  const auto & before = *std::prev(upper);
  const auto & after = *upper;
  const double dt = std::chrono::duration<double>(after.stamp - before.stamp).count();
  if (dt <= 1e-6) return after.q;

  const double elapsed = std::chrono::duration<double>(t - before.stamp).count();
  const double k = std::clamp(elapsed / dt, 0.0, 1.0);
  return before.q.slerp(k, after.q).normalized();
}

}  // namespace io
