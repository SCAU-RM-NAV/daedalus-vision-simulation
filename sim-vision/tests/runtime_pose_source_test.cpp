#include <cassert>

#include <Eigen/Geometry>

#include "tasks/auto_aim/runtime.hpp"

int main()
{
  const auto imu_rotation =
    Eigen::AngleAxisd(0.42, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const auto talos_rotation =
    Eigen::AngleAxisd(-0.31, Eigen::Vector3d::UnitY()).toRotationMatrix();

  auto_aim::Runtime runtime("configs/standard4.yaml");
  std::list<auto_aim::Armor> no_armors;
  auto_aim::RuntimeFrame hardware_frame;
  hardware_frame.timestamp = std::chrono::steady_clock::now();
  hardware_frame.imu_quaternion = Eigen::Quaterniond(imu_rotation);
  runtime.track(no_armors, hardware_frame);
  assert(runtime.solver().R_gimbal2world().isApprox(imu_rotation));

  auto_aim::RuntimeFrame talos_frame;
  talos_frame.timestamp = std::chrono::steady_clock::now();
  talos_frame.pose_source = auto_aim::RuntimePoseSource::gimbal_to_world_matrix;
  talos_frame.R_gimbal2world = talos_rotation;
  runtime.track(no_armors, talos_frame);
  assert(runtime.solver().R_gimbal2world().isApprox(talos_rotation));
  return 0;
}
