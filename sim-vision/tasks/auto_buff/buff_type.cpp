#include "buff_type.hpp"

#include <algorithm>

#include "tools/logger.hpp"
namespace auto_buff
{
FanBlade::FanBlade(
  const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
: center(keypoints_center), type(t)
{
  points.insert(points.end(), kpt.begin(), kpt.end());
  if (points.size() >= 4) center = (points[1] + points[3]) * 0.5f;
}

FanBlade::FanBlade(FanBlade_type t) : type(t)
{
  if (t != _unlight) exit(-1);
}

PowerRune::PowerRune(
  std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune)
: r_center(center), light_num(ts.size())
{
  (void)last_powerrune;

  if (ts.empty()) {
    tools::logger()->debug("[PowerRune] 未识别到扇叶!");
    unsolvable_ = true;
    return;
  }

  auto target_fanblade_it = std::find_if(ts.begin(), ts.end(), [](const FanBlade & fanblade) {
    return fanblade.type == _target;
  });

  if (target_fanblade_it == ts.end()) {
    tools::logger()->debug("[PowerRune] 未指定目标扇叶!");
    unsolvable_ = true;
    return;
  }

  std::iter_swap(ts.begin(), target_fanblade_it);
  for (auto it = ts.begin() + 1; it != ts.end(); ++it) {
    it->type = _light;
  }
  target_leaf_id = -1;
  for (auto & t : ts) {
    t.leaf_id = -1;
    t.leaf_angle = 0.0;
    t.leaf_angle_valid = false;
  }

  /// 填充FanBlade.angle

  double angle = atan_angle(ts[0].center);
  for (auto & t : ts) {
    t.angle = atan_angle(t.center) - angle;
    if (t.angle < -1e-3) t.angle += CV_2PI;
  }

  /// fanblades调整顺序

  std::sort(ts.begin(), ts.end(), [](const FanBlade & a, const FanBlade & b) {
    return a.angle < b.angle;
  });  // 按照 t.angle 从小到大排序 ts
  const std::vector<double> target_angles = {
    0, 2.0 * CV_PI / 5.0, 4.0 * CV_PI / 5.0, 6.0 * CV_PI / 5.0, 8.0 * CV_PI / 5.0};
  fanblades.reserve(target_angles.size());

  // 将识别到的扇叶状态设为_light，未识别的则设为_unlight
  for (std::size_t i = 0, j = 0; i < target_angles.size(); i++) {
    if (j < ts.size() && std::fabs(ts[j].angle - target_angles[i]) < CV_PI / 5.0)
      fanblades.emplace_back(ts[j++]);
    else
      fanblades.emplace_back(FanBlade(_unlight));
  }
  for (auto & blade : fanblades) {
    blade.leaf_id = -1;
    blade.leaf_angle = 0.0;
    blade.leaf_angle_valid = false;
  }
};

double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI;
}
}  // namespace auto_buff
