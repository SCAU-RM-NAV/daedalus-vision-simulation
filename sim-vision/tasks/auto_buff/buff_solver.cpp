#include "buff_solver.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <Eigen/Geometry>

#include "tools/logger.hpp"
namespace auto_buff
{
namespace
{
// 第一个点为R标，最后一个点为扇叶中心
const std::vector<cv::Point3f> BUFF_OBJECT_POINTS = {
  cv::Point3f(0.0f, 0.0f, 0.0f), cv::Point3f(0.0f, 0.154f, 0.700f),
  cv::Point3f(0.0f, 0.0f, 0.854f), cv::Point3f(0.0f, -0.154f, 0.700f),
  cv::Point3f(0.0f, 0.0f, 0.700f)};

// 检查点数值是否正常
bool finite_point(const cv::Point2f & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y);
}

// 检查向量数值是否正常
bool finite_vec3(const cv::Vec3d & value)
{
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

double mean_reprojection_error(
  const std::vector<cv::Point2f> & observed, const std::vector<cv::Point2f> & reprojected)
{
  if (observed.size() != reprojected.size() || observed.empty()) return INF;
  double error = 0.0;
  for (std::size_t i = 0; i < observed.size(); ++i) error += cv::norm(observed[i] - reprojected[i]);
  return error / static_cast<double>(observed.size());
}


constexpr double kLeafAngleStep = 2.0 * CV_PI / 5.0;

double leaf_phase_distance(double current, double previous)
{
  return std::abs(std::remainder(current - previous, kLeafAngleStep));
}

}  // namespace

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
  if (yaml["pnp"] && yaml["pnp"]["max_reprojection_error"]) {
    max_reprojection_error_ = yaml["pnp"]["max_reprojection_error"].as<double>();
  } else if (yaml["max_reprojection_error"]) {
    max_reprojection_error_ = yaml["max_reprojection_error"].as<double>();
  }
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

const std::vector<cv::Point3f> & Solver::object_points() { return BUFF_OBJECT_POINTS; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void Solver::set_R_gimbal2world_matrix(const Eigen::Matrix3d & R_gimbal2world)
{
  R_gimbal2world_ = R_gimbal2world;
}

void Solver::set_runtime_calibration(
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
  const Eigen::Matrix3d & R_camera2gimbal, const Eigen::Vector3d & t_camera2gimbal)
{
  camera_matrix_ = camera_matrix.clone();
  distort_coeffs_ = distort_coeffs.clone();
  R_camera2gimbal_ = R_camera2gimbal;
  t_camera2gimbal_ = t_camera2gimbal;
}

void Solver::solve(
  std::optional<PowerRune> & ps, const std::chrono::steady_clock::time_point * observed_time) const
{
  if (!ps.has_value()) return;
  PowerRune & p = ps.value();
  p.pnp_valid = false;
  p.reprojection_error = INF;
  p.pnp_debug = {};
  p.pnp_debug.hard_gate_enabled = false;
  p.pnp_debug.prediction_rotation_enabled = false;
  p.pnp_debug.has_prediction_reference = false;

  if (p.fanblades.empty() || p.target().type == _unlight || p.target().points.size() != 4) {
    tools::logger()->debug("[Buff Solver] Invalid target points for PnP.");
    return;
  }

  std::vector<cv::Point2f> image_points = {
    p.r_center, p.target().points[1], p.target().points[2], p.target().points[3], p.target().center};
  if (!std::all_of(image_points.begin(), image_points.end(), finite_point)) {
    tools::logger()->debug("[Buff Solver] Non-finite image point for PnP.");
    return;
  }

  std::vector<cv::Mat> rvecs, tvecs;
  const bool ok = cv::solvePnPGeneric(
      BUFF_OBJECT_POINTS, image_points, camera_matrix_, distort_coeffs_,
      rvecs, tvecs, false, cv::SOLVEPNP_IPPE);

  if (!ok) {
    tools::logger()->debug("[Buff Solver] solvePnPGeneric failed.");
    return;
  }

  // if (!ok) {
  //   tools::logger()->debug("[Buff Solver] solvePnP failed.");
  //   return;
  // }

  // if (!finite_vec3(rvec) || !finite_vec3(tvec) || tvec[2] <= 1e-6) {
  //   tools::logger()->debug(
  //     "[Buff Solver] Invalid solution. tvec: [{:.3f}, {:.3f}, {:.3f}]",
  //     tvec[0], tvec[1], tvec[2]);
  //   return;
  // }

  // // 计算重投影误差
  // std::vector<cv::Point2f> reprojected;
  // cv::projectPoints(BUFF_OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, reprojected);
  // const double reprojection_error = mean_reprojection_error(image_points, reprojected);

  // if (reprojection_error > max_reprojection_error_) {
  //   tools::logger()->debug(
  //     "[Buff Solver] Reprojection error {:.3f}px exceeds max {:.3f}px. Discarding.",
  //     reprojection_error, max_reprojection_error_);
  //   return;
  // }

  // // 计算旋转矩阵
  // cv::Mat rmat;
  // cv::Rodrigues(rvec, rmat);
  // Eigen::Matrix3d R_buff2camera;
  // cv::cv2eigen(rmat, R_buff2camera);

  // p.pnp_debug.selected_index = 0;
  // p.pnp_debug.candidate_count = 1;
  // p.pnp_debug.selection_reason = PnpSelectionReason::FACING_CAMERA;
  // auto & debug = p.pnp_debug.candidates[0];
  // debug.valid = true;
  // debug.reprojection_error = reprojection_error;
  // debug.rvec = Eigen::Vector3d(rvec[0], rvec[1], rvec[2]);
  // debug.tvec = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  // debug.depth = tvec[2];

  // rvec_ = rvec;
  // tvec_ = tvec;
  // -----------------------------------------------------------------------
  //遍历两解，选normal_camera_z（buff X轴在相机坐标系Z分量）最大的
  // 两解均为正值时选更正的那个，门控要求必须 > 0
  struct Candidate
  {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    Eigen::Matrix3d R;
    double nz;
    double reproj;
    double physical_angle;
  };
  std::vector<Candidate> candidates;
  for (std::size_t i = 0; i < rvecs.size(); ++i) {
    cv::Vec3d rv(rvecs[i].at<double>(0), rvecs[i].at<double>(1), rvecs[i].at<double>(2));
    cv::Vec3d tv(tvecs[i].at<double>(0), tvecs[i].at<double>(1), tvecs[i].at<double>(2));
    if (!finite_vec3(rv) || !finite_vec3(tv) || tv[2] <= 1e-6) continue;
    cv::Mat rmat; cv::Rodrigues(rv, rmat);
    Eigen::Matrix3d R; cv::cv2eigen(rmat, R);
    const double normal_z = R.col(0).z();
    if (normal_z <= 0.0) continue;

    const Eigen::Matrix3d R_buff2world_candidate =
      R_gimbal2world_ * R_camera2gimbal_ * R;
    const Eigen::Vector3d ypr_candidate =
      tools::eulers(R_buff2world_candidate, 2, 1, 0);
    if (!ypr_candidate.allFinite()) continue;

    std::vector<cv::Point2f> reproj;
    cv::projectPoints(BUFF_OBJECT_POINTS, rv, tv, camera_matrix_, distort_coeffs_, reproj);
    candidates.push_back({
      rv, tv, R, normal_z, mean_reprojection_error(image_points, reproj),
      tools::limit_rad(ypr_candidate[2] - CV_PI / 2.0)});
  }

  if (candidates.empty()) {
    tools::logger()->debug("[Buff Solver] No valid IPPE solutions.");
    return;
  }

  const bool has_fresh_phase_reference =
    observed_time != nullptr && last_physical_angle_.has_value() && last_pnp_time_.has_value() &&
    *observed_time >= *last_pnp_time_ &&
    *observed_time - *last_pnp_time_ <= std::chrono::milliseconds(100);

  auto best = std::max_element(
    candidates.begin(), candidates.end(),
    [](const Candidate & a, const Candidate & b) { return a.nz < b.nz; });

  if (has_fresh_phase_reference) {
    best = std::min_element(
      candidates.begin(), candidates.end(), [this](const Candidate & a, const Candidate & b) {
        return leaf_phase_distance(a.physical_angle, *last_physical_angle_) <
               leaf_phase_distance(b.physical_angle, *last_physical_angle_);
      });
  }

  if (best->nz <= 0.0) {
    tools::logger()->debug("[Buff Solver] All solutions face wrong way (nz={:.3f}).", best->nz);
    return;
  }

  if (best->reproj > max_reprojection_error_) {
    tools::logger()->debug("[Buff Solver] Reprojection error {:.3f}px exceeds max.", best->reproj);
    return;
  }

  cv::Vec3d rvec = best->rvec;
  cv::Vec3d tvec = best->tvec;

  // 精化后检查数值有效性
  if (!finite_vec3(rvec) || !finite_vec3(tvec) || tvec[2] <= 1e-6) {
    tools::logger()->debug("[Buff Solver] LM refinement diverged.");
    return;
  }

  cv::Mat rmat_refined;
  cv::Rodrigues(rvec, rmat_refined);
  Eigen::Matrix3d R_buff2camera;
  cv::cv2eigen(rmat_refined, R_buff2camera);

  // 精化后重新检查法向量方向，防止LM把解翻转到背面
  if (R_buff2camera.col(0).z() <= 0.0) {
    tools::logger()->debug(
      "[Buff Solver] LM refinement flipped normal (nz={:.3f}).", R_buff2camera.col(0).z());
    return;
  }

  std::vector<cv::Point2f> reproj_refined;
  cv::projectPoints(BUFF_OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, reproj_refined);
  const double reprojection_error = mean_reprojection_error(image_points, reproj_refined);
  // -----------------------------------------------------------------------


  p.pnp_debug.selected_index = 0;
  p.pnp_debug.candidate_count = static_cast<int>(candidates.size());
  p.pnp_debug.has_prediction_reference = has_fresh_phase_reference;
  p.pnp_debug.selection_reason = has_fresh_phase_reference
                                   ? PnpSelectionReason::ZERO_CONTINUITY
                                   : PnpSelectionReason::FACING_CAMERA;
  auto & debug = p.pnp_debug.candidates[0];
  debug.valid = true;
  debug.reprojection_error = reprojection_error;
  debug.rvec = Eigen::Vector3d(rvec[0], rvec[1], rvec[2]);
  debug.tvec = Eigen::Vector3d(tvec[0], tvec[1], tvec[2]);
  debug.depth = tvec[2];
  debug.normal_camera_z = best->nz;

  rvec_ = rvec;
  tvec_ = tvec;
  // -----------------------------------------------------------------------

  Eigen::Vector3d t_buff2camera;
  cv::cv2eigen(tvec, t_buff2camera);

  // 目标扇叶中心
  Eigen::Vector3d target_center_in_buff{{0.0, 0.0, 700e-3}};

  // buff -> camera
  Eigen::Vector3d xyz_in_camera = t_buff2camera;
  Eigen::Vector3d target_center_in_camera = R_buff2camera * target_center_in_buff + t_buff2camera;

  // camera -> gimbal
  Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal_ * R_buff2camera;
  Eigen::Vector3d xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  Eigen::Vector3d target_center_in_gimbal =
    R_camera2gimbal_ * target_center_in_camera + t_camera2gimbal_;

  /// gimbal -> world
  Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;

  // R标
  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);

  // 扇叶中心
  p.blade_xyz_in_world = R_gimbal2world_ * target_center_in_gimbal;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  p.target_center_world = p.blade_xyz_in_world;
  p.rotation_world = R_buff2world;

  {
    const Eigen::Vector3d ypr_full = tools::eulers(R_buff2world, 2, 1, 0);
    const Eigen::Vector2d yp = tools::yaw_pitch_from_normal(R_buff2world);
    p.ypr_in_world = Eigen::Vector3d(yp.x(), yp.y(), ypr_full.z());
  }
  p.physical_angle = tools::limit_rad(p.ypr_in_world[2] - CV_PI / 2.0);  // 扇叶角度
  p.reprojection_error = reprojection_error;
  p.pnp_valid = true;
  if (observed_time != nullptr) {
    last_physical_angle_ = p.physical_angle;
    last_pnp_time_ = *observed_time;
  }

  debug.ypr_in_world = p.ypr_in_world;
  debug.rotation_world = R_buff2world;
  debug.r_center_world = p.xyz_in_world;
  debug.normal_world = R_buff2world.col(0);
  debug.physical_angle = p.physical_angle;
}

// 调试用
cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  // buff坐标系(单位:m)到像素坐标系
  std::vector<cv::Point3d> world_points;
  std::vector<cv::Point2d> image_points;
  world_points.push_back(x);
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

// xyz_in_world2xyz_in_pix
std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, const Eigen::Matrix3d & R_buff2world) const
{
  // get R_buff2camera t_buff2camera
  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(BUFF_OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff
