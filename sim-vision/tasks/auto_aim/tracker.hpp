#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <list>
#include <string>
#include <tuple>
#include <vector>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;
  double max_tracking_dt_s() const;
  void set_enemy_color(Color enemy_color);
  void set_enemy_color_from_camp(uint8_t camp);

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  enum class LossReason
  {
    none,
    invisible,
    extinguished,
    invalid
  };

  struct TrackSlot
  {
    Target target;
    std::string state{"lost"};
    int detect_count{0};
    int temp_lost_count{0};
    int extinguish_track_count{0};
    LossReason loss_reason{LossReason::none};
    uint64_t track_epoch{0};
    uint64_t observation_seq{0};
    int observed_frames{0};
    bool live_observation{false};
    std::chrono::steady_clock::time_point first_observed{};
    std::chrono::steady_clock::time_point last_observed{};
  };

  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int max_extinguish_track_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  double max_tracking_dt_s_ = 0.1;
  EkfNoiseConfig ekf_noise_config_;
  TrackSlot current_slot_;
  TrackSlot candidate_slot_;
  std::string pre_state_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;
  std::vector<ArmorName> ignored_armors_;
  std::vector<ArmorName> priority_armors_;
  uint64_t next_track_epoch_{0};

  void reset_slot(TrackSlot & slot, LossReason loss_reason = LossReason::none);
  void sync_tracking_info(TrackSlot & slot);
  bool process_slot(
    TrackSlot & slot, std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
  bool validate_slot(TrackSlot & slot, bool check_nis);
  void state_machine(TrackSlot & slot, bool found);
  void apply_priority_config(std::list<Armor> & armors) const;

  bool set_target(
    TrackSlot & slot, std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(
    TrackSlot & slot, std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
