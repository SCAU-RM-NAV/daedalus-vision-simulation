#include <cassert>

#include <Eigen/Geometry>

#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_buff/buff_solver.hpp"

namespace
{
Eigen::Matrix3d rotation_z(double radians)
{
  return Eigen::AngleAxisd(radians, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}
}  // namespace

int main()
{
  const auto calibration_rotation = rotation_z(0.35);
  const Eigen::Vector3d translation{0.12, -0.03, 0.08};
  const auto world_rotation = rotation_z(-0.2);
  const cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 1000.0, 0.0, 720.0, 0.0, 1000.0,
                                 540.0, 0.0, 0.0, 1.0);
  const cv::Mat distortion = cv::Mat::zeros(1, 8, CV_64F);

  auto_aim::Solver auto_aim_solver("configs/standard4.yaml");
  auto_aim_solver.set_runtime_calibration(
    camera_matrix, distortion, calibration_rotation, translation);
  auto_aim_solver.set_R_gimbal2world_matrix(world_rotation);
  assert(auto_aim_solver.R_gimbal2world().isApprox(world_rotation));

  auto_buff::Solver buff_solver("configs/standard4.yaml");
  buff_solver.set_runtime_calibration(camera_matrix, distortion, calibration_rotation, translation);
  buff_solver.set_R_gimbal2world_matrix(world_rotation);
  assert(buff_solver.R_gimbal2world().isApprox(world_rotation));
  return 0;
}
