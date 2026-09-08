#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <limits>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"
namespace auto_buff
{
const int INF = 1000000;
enum PowerRune_type { SMALL, BIG };
enum FanBlade_type { _target, _unlight, _light };
enum Track_status { TRACK, TEM_LOSE, LOSE };

struct PnpCandidateDebug
{
  bool valid = false;
  double reprojection_error = INF;
  std::array<double, 5> point_reprojection_errors{
    INF, INF, INF, INF, INF};
  Eigen::Vector3d rvec = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d tvec = Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d ypr_in_world =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix3d rotation_world =
    Eigen::Matrix3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d r_center_world =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Vector3d normal_world =
    Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
  double physical_angle = 0.0;
  double depth = 0.0;
  double normal_camera_z = std::numeric_limits<double>::quiet_NaN();
  double normal_facing = std::numeric_limits<double>::quiet_NaN();
  double rotation_distance_to_prediction = INF;
  bool hard_gate_passed = true;
  bool depth_gate_passed = true;
  bool position_gate_passed = true;
  bool rotation_gate_passed = true;
  bool normal_gate_passed = true;
  std::uint32_t hard_gate_reject_mask = 0;
};

enum PnpGateRejectMask : std::uint32_t
{
  PNP_GATE_REJECT_NONE = 0,
  PNP_GATE_REJECT_DEPTH = 1U << 0,
  PNP_GATE_REJECT_POSITION = 1U << 1,
  PNP_GATE_REJECT_ROTATION = 1U << 2,
  PNP_GATE_REJECT_NORMAL = 1U << 3,
};

enum class PnpSelectionReason
{
  NONE,
  LOWEST_REPROJECTION,
  ZERO_CONTINUITY,
  GATE_REJECTED,  // retained for debug compatibility; solver no longer emits this
  FACING_CAMERA,  // IPPE双解中选中法向量正对相机的解
};

struct PnpDebug
{
  int candidate_count = 0;
  int lowest_reprojection_index = -1;
  int selected_index = -1;
  PnpSelectionReason selection_reason = PnpSelectionReason::NONE;
  bool hard_gate_enabled = false;
  bool prediction_rotation_enabled = false;
  bool has_prediction_reference = false;
  std::array<PnpCandidateDebug, 2> candidates;
};

class FanBlade
{
// 注意：fanblades[0] 永远是目标
public:
  cv::Point2f center;               // 扇页击打中心: (points[1] + points[3]) / 2
  std::vector<cv::Point2f> points;  // 0: R中心, 1/2/3: 扇叶三角点
  double angle = 0.0, width = 0.0, height = 0.0;  // 当前帧相对目标扇叶的排序角
  double confidence = 0.0;
  int label = -1;
  int leaf_id = -1; // 全局扇叶id（第一次目标扇叶作为0号扇叶，以此类推）
  double leaf_angle = 0.0; // 世界坐标系下扇叶角度
  bool leaf_angle_valid = false; // pnp后得到leaf_angle置为true
  FanBlade_type type;  // 类型

  explicit FanBlade() = default;

  // explicit FanBlade(const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(FanBlade_type t);
};

class PowerRune
{
public:
  cv::Point2f r_center;
  std::vector<FanBlade> fanblades;  // 按target开始顺时针

  int light_num;

  Eigen::Vector3d xyz_in_world;  // （原点）R标在世界坐标系下的位置，单位：m
  Eigen::Vector3d ypr_in_world;  // 单位：rad
  Eigen::Vector3d ypd_in_world;  // 球坐标系

  Eigen::Vector3d blade_xyz_in_world;  // 目标扇叶中心在世界坐标系下的位置，单位：m
  Eigen::Vector3d blade_ypd_in_world;  // 球坐标系, 单位: m
  Eigen::Matrix3d rotation_world = Eigen::Matrix3d::Identity();
  Eigen::Vector3d target_center_world = Eigen::Vector3d::Zero();

  bool pnp_valid = false; // pnp结果是否可信
  double physical_angle = 0.0;  // 由pnp解算后得来的世界坐标系下的目标扇叶角度，用于拟合
  double reprojection_error = INF; // 重投影误差，用于过滤垃圾解
  PnpDebug pnp_debug;
  int target_leaf_id = -1;
  double last_observed_time = 0.0;

  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
  explicit PowerRune() = default;

  FanBlade & target() { return fanblades[0]; };
  const FanBlade & target() const { return fanblades[0]; };

  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_;
  bool unsolvable_ = false;

  double atan_angle(cv::Point2f v) const;  // [0, 2CV_PI]
};
}  // namespace auto_buff
#endif  // BUFF_TYPE_HPP
