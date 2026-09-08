#include <cassert>
#include <list>

#include <opencv2/core.hpp>

#include "io/sim/talos_ipc.hpp"
#include "sim/debug_visualizer.hpp"
#include "tasks/auto_aim/armor.hpp"

int main()
{
  cv::Mat raw(180, 320, CV_8UC3, cv::Scalar(0, 0, 0));
  std::list<auto_aim::Armor> armors;
  auto_aim::Armor armor(
    0, 0.95F, {20, 20, 80, 40},
    {{20.0F, 20.0F}, {100.0F, 20.0F}, {100.0F, 60.0F}, {20.0F, 60.0F}});
  armors.push_back(armor);

  io::sim::GroundTruthBatch truth{};
  truth.target_count = 1;
  truth.targets[0].target_id = 7;
  truth.targets[0].position_m = {4.0F, 0.0F, 0.5F};

  const auto overlay = sim::draw_auto_aim_debug(raw, armors, 1, nullptr, nullptr, nullptr, &truth);
  assert(overlay.rows == raw.rows);
  assert(overlay.cols == raw.cols);
  assert(cv::countNonZero(overlay.reshape(1)) > 0);
  assert(cv::countNonZero(raw.reshape(1)) == 0);
  return 0;
}
