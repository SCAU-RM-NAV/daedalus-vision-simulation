#include "debug_visualizer.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>
#include <opencv2/imgproc.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

namespace sim
{
namespace
{
constexpr double kRadToDeg = 57.2957795130823208768;

void draw_frame_id(cv::Mat & image, int frame_seq)
{
  tools::draw_text(image, fmt::format("frame {}", frame_seq), {10, 28}, {255, 255, 255}, 0.7, 2);
}

void draw_armor_detections(cv::Mat & image, const std::list<auto_aim::Armor> & armors)
{
  for (const auto & armor : armors) {
    tools::draw_points(image, armor.points, {0, 255, 0}, 2);
    tools::draw_text(
      image,
      fmt::format(
        "{} {} {} {:.2f} pnp:{}", auto_aim::COLORS[armor.color], auto_aim::ARMOR_NAMES[armor.name],
        auto_aim::ARMOR_TYPES[armor.type], armor.confidence,
        armor.xyz_in_world.allFinite() ? "ok" : "pending"),
      armor.center, {0, 255, 0}, 0.5, 1);
  }
}

void draw_buff_detections(cv::Mat & image, const auto_buff::PowerRune & rune)
{
  tools::draw_point(image, rune.r_center, {0, 0, 255}, 5);
  tools::draw_text(image, "R center", rune.r_center + cv::Point2f(8.0F, -8.0F), {0, 0, 255}, 0.5, 1);
  for (const auto & blade : rune.fanblades) {
    if (blade.type == auto_buff::_unlight || blade.points.size() < 4) continue;
    const auto color = blade.type == auto_buff::_target ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    tools::draw_points(image, blade.points, color, blade.type == auto_buff::_target ? 3 : 2);
    tools::draw_text(
      image, fmt::format("leaf {} {:.2f}", blade.leaf_id, blade.confidence), blade.center, color, 0.5, 1);
  }
}
}  // namespace

void draw_truth_panel(cv::Mat & image, const io::sim::GroundTruthBatch & truth)
{
  if (image.empty()) return;
  const int panel_width = std::min(420, image.cols);
  cv::rectangle(image, {0, 0, panel_width, image.rows}, {24, 24, 24}, cv::FILLED);
  tools::draw_text(
    image, fmt::format("Truth targets:{} runes:{}", truth.target_count, truth.rune_count), {10, 28},
    {255, 255, 255}, 0.6, 1);
  int baseline = 54;
  for (std::uint32_t index = 0; index < truth.target_count && baseline < image.rows - 20; ++index) {
    const auto & target = truth.targets[index];
    tools::draw_text(
      image,
      fmt::format(
        "T{} ({:.2f}, {:.2f}, {:.2f}) w:{:.2f}", target.target_id, target.position_m[0],
        target.position_m[1], target.position_m[2], target.yaw_rate_radps),
      {10, baseline}, {0, 255, 255}, 0.5, 1);
    baseline += 22;
  }
  for (std::uint32_t index = 0; index < truth.rune_count && baseline < image.rows - 20; ++index) {
    const auto & rune = truth.runes[index];
    tools::draw_text(
      image,
      fmt::format(
        "R{} angle:{:.2f} omega:{:.2f} active:{}", rune.rune_id, rune.angle_rad,
        rune.angular_velocity_radps, rune.active_blade_id),
      {10, baseline}, {255, 0, 255}, 0.5, 1);
    baseline += 22;
  }
}

cv::Mat draw_auto_aim_debug(
  const cv::Mat & raw_bgr, const std::list<auto_aim::Armor> & armors, int frame_seq,
  const std::optional<auto_aim::Target> * target, auto_aim::Solver * solver,
  const Eigen::Vector4d * planner_aim,
  const io::sim::GroundTruthBatch * truth)
{
  cv::Mat image = raw_bgr.clone();
  if (image.empty()) return image;
  draw_frame_id(image, frame_seq);
  draw_armor_detections(image, armors);
  if (target != nullptr && target->has_value() && solver != nullptr) {
    for (const auto & xyza : target->value().armor_xyza_list()) {
      const auto points = solver->reproject_armor(
        xyza.head(3), xyza[3], target->value().armor_type, target->value().name);
      tools::draw_points(image, points, {0, 255, 255}, 2);
    }
    if (planner_aim != nullptr && planner_aim->allFinite()) {
      const auto points = solver->reproject_armor(
        planner_aim->head(3), (*planner_aim)[3], target->value().armor_type, target->value().name);
      tools::draw_points(image, points, {0, 0, 255}, 3);
      tools::draw_text(image, "planner aim", {10, 56}, {0, 0, 255}, 0.6, 1);
    }
  }
  if (truth != nullptr) draw_truth_panel(image, *truth);
  return image;
}

cv::Mat draw_buff_debug(
  const cv::Mat & raw_bgr, const std::optional<auto_buff::PowerRune> & rune, int frame_seq,
  auto_buff::Solver * solver, const auto_buff::Aimer * aimer, const io::sim::GroundTruthBatch * truth)
{
  cv::Mat image = raw_bgr.clone();
  if (image.empty()) return image;
  draw_frame_id(image, frame_seq);
  if (rune.has_value()) {
    draw_buff_detections(image, *rune);
    if (rune->pnp_valid && solver != nullptr) {
      const auto projected = solver->reproject_buff(rune->xyz_in_world, rune->rotation_world);
      tools::draw_points(image, projected, {0, 255, 0}, 2);
      tools::draw_text(
        image, fmt::format("phase {:.1f} deg", rune->physical_angle * kRadToDeg), {10, 56},
        {0, 255, 0}, 0.6, 1);
    }
  }
  if (aimer != nullptr) {
    const auto & ballistic = aimer->last_ballistic_result();
    const auto id_match = aimer->id_match_debug();
    tools::draw_text(
      image,
      fmt::format(
        "predict leaf:{} phase:{:.1f} id-match:{}", ballistic.attack_leaf_id,
        ballistic.predicted_angle * kRadToDeg, id_match.accepted ? "yes" : "no"),
      {10, 82}, {0, 0, 255}, 0.6, 1);
  }
  if (truth != nullptr) draw_truth_panel(image, *truth);
  return image;
}
}  // namespace sim
