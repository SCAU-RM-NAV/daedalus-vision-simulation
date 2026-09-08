#ifndef AUTO_AIM__FIRE_GATE_HPP
#define AUTO_AIM__FIRE_GATE_HPP

#include <chrono>
#include <cstdint>
#include <optional>

#include "tracking_info.hpp"

namespace auto_aim
{

enum class FireGateMode
{
  off,
  shadow,
  enforce
};

enum class FireGateState
{
  disabled,
  no_target,
  track_not_ready,
  temp_lost,
  observation_stale,
  warmup,
  stabilizing,
  ready
};

struct FireGateSample
{
  TrackingInfo tracking;
  bool aim_center = false;
  std::optional<double> prediction_change_deg;
  std::chrono::steady_clock::time_point now;
};

struct FireGateDebug
{
  FireGateMode mode = FireGateMode::off;
  FireGateState state = FireGateState::disabled;
  bool ready = true;
  bool allow_fire = true;
  bool raw_fire = false;
  bool actual_fire = false;
  uint64_t track_epoch = 0;
  uint64_t observation_seq = 0;
  int observed_frames = 0;
  int stable_observations = 0;
  double prediction_change_deg = -1.0;
  double observation_age_ms = 0.0;
  double warmup_remaining_ms = 0.0;
  double max_observation_age_ms = 100.0;
};

struct FireGateConfig
{
  FireGateMode mode = FireGateMode::off;
  double warmup_time = 0.25;
  int stable_observations = 4;
  double max_prediction_change_deg = 0.25;
};

class FireGate
{
public:
  explicit FireGate(const FireGateConfig & config);
  FireGate(
    FireGateMode mode, double warmup_time, int stable_observations,
    double max_prediction_change_deg);

  void update(const FireGateSample & sample);
  void update_no_target(std::chrono::steady_clock::time_point now);

  bool ready() const;
  bool allow_fire() const;
  void set_fire_result(bool raw_fire, bool actual_fire);
  const FireGateDebug & debug() const;

  static const char * state_name(FireGateState state);
  static const char * mode_name(FireGateMode mode);

private:
  FireGateMode mode_ = FireGateMode::off;
  double warmup_time_ = 0.25;
  int required_stable_observations_ = 4;
  double max_prediction_change_deg_ = 0.25;

  FireGateState state_ = FireGateState::disabled;
  bool ready_ = true;
  bool has_track_ = false;
  bool aim_center_ = false;
  uint64_t track_epoch_ = 0;
  uint64_t last_observation_seq_ = 0;
  int stable_observations_ = 0;
  double average_observation_period_ = 1.0 / 30.0;
  double average_observation_latency_ = 0.0;
  std::chrono::steady_clock::time_point learning_since_{};
  std::chrono::steady_clock::time_point last_observation_stamp_{};
  FireGateDebug debug_;

  void reset_learning(std::chrono::steady_clock::time_point now);
  void set_state(FireGateState state);
  void update_observation_timing(
    std::chrono::steady_clock::time_point stamp, std::chrono::steady_clock::time_point now);
  double max_observation_age() const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__FIRE_GATE_HPP
