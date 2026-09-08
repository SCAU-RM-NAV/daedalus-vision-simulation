#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>

#include "tasks/auto_aim/runtime.hpp"
#include "tasks/auto_aim/tracker.hpp"

int main()
{
  auto_aim::Runtime runtime("configs/standard4.yaml");
  std::list<auto_aim::Armor> armors;
  auto_aim::RuntimeFrame frame;
  frame.timestamp = std::chrono::steady_clock::now();
  frame.bullet_speed = 24.0F;

  const auto debug = runtime.process_with_debug(armors, frame);
  assert(!debug.target.has_value());
  assert(!debug.plan.control);
  assert(!debug.plan.fire);

  const auto temporary_config =
    std::filesystem::temp_directory_path() / "burn_your_bridges_tracker_max_dt.yaml";
  {
    std::ifstream source("configs/standard4.yaml");
    assert(source.good());
    std::ofstream destination(temporary_config);
    destination << std::string(std::istreambuf_iterator<char>(source), {})
                << "\ntracker_max_dt_s: 0.25\n";
  }

  auto_aim::Solver solver(temporary_config.string());
  auto_aim::Tracker tracker(temporary_config.string(), solver);
  if (std::abs(tracker.max_tracking_dt_s() - 0.25) >= 1e-9) {
    std::cerr << "tracker_max_dt_s was not applied\n";
    std::filesystem::remove(temporary_config);
    return 1;
  }
  std::filesystem::remove(temporary_config);
  return 0;
}
