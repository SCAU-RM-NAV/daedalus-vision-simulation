#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <optional>
#include <vector>

#include "buff_type.hpp"
#include "tools/math_tools.hpp"
namespace auto_buff
{
class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  void set_R_gimbal2world_matrix(const Eigen::Matrix3d & R_gimbal2world);

  void set_runtime_calibration(
    const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
    const Eigen::Matrix3d & R_camera2gimbal, const Eigen::Vector3d & t_camera2gimbal);

  void solve(
    std::optional<PowerRune> & ps,
    const std::chrono::steady_clock::time_point * observed_time = nullptr) const;

  // 调试用
  cv::Point2f point_buff2pixel(cv::Point3f x);

  static const std::vector<cv::Point3f> & object_points();

  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, const Eigen::Matrix3d & R_buff2world) const;

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  mutable cv::Vec3d rvec_, tvec_;
  mutable std::optional<double> last_physical_angle_;
  mutable std::optional<std::chrono::steady_clock::time_point> last_pnp_time_;

  double max_reprojection_error_ = 5.0;  // pnp结果容忍阈值
};
}  // namespace auto_buff
#endif  // AUTO_AIM__SOLVER_HPP
