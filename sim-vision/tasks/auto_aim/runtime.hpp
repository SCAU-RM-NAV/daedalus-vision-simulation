#ifndef AUTO_AIM__RUNTIME_HPP
#define AUTO_AIM__RUNTIME_HPP

#include <chrono>
#include <cstdint>
#include <list>
#include <optional>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "planner/planner.hpp"
#include "solver.hpp"
#include "tracker.hpp"

namespace auto_aim
{
enum class RuntimePoseSource
{
  imu_quaternion,
  gimbal_to_world_matrix
};

struct RuntimeFrame
{
  std::chrono::steady_clock::time_point timestamp;
  RuntimePoseSource pose_source = RuntimePoseSource::imu_quaternion;
  Eigen::Quaterniond imu_quaternion = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d R_gimbal2world = Eigen::Matrix3d::Identity();
  float bullet_speed = 0.0F;
  float pitch = 0.0F;
  std::uint8_t camp = 0;
  const cv::Mat * camera_matrix = nullptr;
  const cv::Mat * distort_coeffs = nullptr;
  const Eigen::Matrix3d * R_camera2gimbal = nullptr;
  const Eigen::Vector3d * t_camera2gimbal = nullptr;
};

struct RuntimeDebugResult
{
  Plan plan{};
  std::optional<Target> target;
  Eigen::Vector4d aim_xyza = Eigen::Vector4d::Zero();
};

class Runtime
{
public:
  explicit Runtime(const std::string & config_path);

  std::list<Target> track(std::list<Armor> & armors, const RuntimeFrame & frame);

  Plan plan(std::optional<Target> target, float bullet_speed, float pitch);

  Plan process(std::list<Armor> & armors, const RuntimeFrame & frame);
  RuntimeDebugResult process_with_debug(std::list<Armor> & armors, const RuntimeFrame & frame);

  Solver & solver();

private:
  Solver solver_;
  Tracker tracker_;
  Planner planner_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__RUNTIME_HPP
