#ifndef SIM__DEBUG_VISUALIZER_HPP
#define SIM__DEBUG_VISUALIZER_HPP

#include <list>
#include <optional>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "io/sim/talos_ipc.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_type.hpp"

namespace sim
{
void draw_truth_panel(cv::Mat & image, const io::sim::GroundTruthBatch & truth);

cv::Mat draw_auto_aim_debug(
  const cv::Mat & raw_bgr, const std::list<auto_aim::Armor> & armors, int frame_seq,
  const std::optional<auto_aim::Target> * target, auto_aim::Solver * solver,
  const Eigen::Vector4d * planner_aim = nullptr,
  const io::sim::GroundTruthBatch * truth = nullptr);

cv::Mat draw_buff_debug(
  const cv::Mat & raw_bgr, const std::optional<auto_buff::PowerRune> & rune, int frame_seq,
  auto_buff::Solver * solver, const auto_buff::Aimer * aimer,
  const io::sim::GroundTruthBatch * truth = nullptr);
}  // namespace sim

#endif  // SIM__DEBUG_VISUALIZER_HPP
