#include "fire_gate.hpp"

#include <algorithm>

namespace auto_aim
{
namespace
{

double seconds_between(
  std::chrono::steady_clock::time_point later, std::chrono::steady_clock::time_point earlier)
{
  return std::chrono::duration<double>(later - earlier).count();
}

}  // namespace

FireGate::FireGate(const FireGateConfig & config)
: FireGate(
    config.mode, config.warmup_time, config.stable_observations,
    config.max_prediction_change_deg)
{
}

FireGate::FireGate(
  FireGateMode mode, double warmup_time, int stable_observations,
  double max_prediction_change_deg)
: mode_(mode),
  warmup_time_(std::max(0.0, warmup_time)),
  required_stable_observations_(std::max(1, stable_observations)),
  max_prediction_change_deg_(std::max(1e-6, max_prediction_change_deg)),
  state_(mode == FireGateMode::off ? FireGateState::disabled : FireGateState::no_target),
  ready_(mode == FireGateMode::off)
{
  debug_.mode = mode_;
  debug_.state = state_;
  debug_.ready = ready_;
  debug_.allow_fire = allow_fire();
}

void FireGate::reset_learning(std::chrono::steady_clock::time_point now)
{
  learning_since_ = now;
  stable_observations_ = 0;
  ready_ = false;
}

void FireGate::set_state(FireGateState state)
{
  state_ = state;
  debug_.state = state_;
}

void FireGate::update_observation_timing(
  std::chrono::steady_clock::time_point stamp, std::chrono::steady_clock::time_point now)
{
  const bool has_previous_stamp = last_observation_stamp_.time_since_epoch().count() != 0;
  if (has_previous_stamp && stamp > last_observation_stamp_) {
    const auto period = seconds_between(stamp, last_observation_stamp_);
    if (period > 0.001 && period < 0.5) {
      average_observation_period_ = 0.8 * average_observation_period_ + 0.2 * period;
    }
  }

  const auto latency = std::max(0.0, seconds_between(now, stamp));
  average_observation_latency_ =
    has_previous_stamp ? 0.8 * average_observation_latency_ + 0.2 * latency : latency;
  last_observation_stamp_ = stamp;
}

double FireGate::max_observation_age() const
{
  return std::clamp(
    average_observation_latency_ + 3.0 * average_observation_period_, 0.10, 0.30);
}

void FireGate::update(const FireGateSample & sample)
{
  debug_.mode = mode_;
  debug_.track_epoch = sample.tracking.track_epoch;
  debug_.observation_seq = sample.tracking.observation_seq;
  debug_.observed_frames = sample.tracking.observed_frames;

  if (mode_ == FireGateMode::off) {
    ready_ = true;
    set_state(FireGateState::disabled);
    debug_.ready = ready_;
    debug_.allow_fire = allow_fire();
    return;
  }

  const bool epoch_changed = !has_track_ || sample.tracking.track_epoch != track_epoch_;
  const bool aim_mode_changed = has_track_ && sample.aim_center != aim_center_;
  if (epoch_changed) {
    has_track_ = true;
    track_epoch_ = sample.tracking.track_epoch;
    last_observation_seq_ = 0;
    last_observation_stamp_ = {};
    average_observation_period_ = 1.0 / 30.0;
    average_observation_latency_ = 0.0;
    debug_.prediction_change_deg = -1.0;
    reset_learning(sample.now);
  } else if (aim_mode_changed) {
    debug_.prediction_change_deg = -1.0;
    reset_learning(sample.now);
  }
  aim_center_ = sample.aim_center;

  const bool new_observation =
    sample.tracking.observation_seq != 0 &&
    sample.tracking.observation_seq != last_observation_seq_;
  bool prediction_unstable = false;
  if (new_observation) {
    last_observation_seq_ = sample.tracking.observation_seq;
    update_observation_timing(sample.tracking.last_observed, sample.now);

    if (
      sample.tracking.confirmed && sample.tracking.live_observation &&
      !sample.tracking.temp_lost && sample.prediction_change_deg.has_value()) {
      debug_.prediction_change_deg = *sample.prediction_change_deg;
      if (*sample.prediction_change_deg <= max_prediction_change_deg_) {
        stable_observations_ =
          std::min(stable_observations_ + 1, required_stable_observations_);
      } else {
        prediction_unstable = true;
        reset_learning(sample.now);
      }
    } else {
      stable_observations_ = 0;
      if (epoch_changed || aim_mode_changed) debug_.prediction_change_deg = -1.0;
    }
  }

  debug_.stable_observations = stable_observations_;
  const auto observation_age = std::max(
    0.0, seconds_between(sample.now, sample.tracking.last_observed));
  const auto observation_age_limit = max_observation_age();
  debug_.observation_age_ms = observation_age * 1e3;
  debug_.max_observation_age_ms = observation_age_limit * 1e3;

  if (!sample.tracking.confirmed) {
    ready_ = false;
    stable_observations_ = 0;
    debug_.stable_observations = 0;
    set_state(FireGateState::track_not_ready);
  } else if (sample.tracking.temp_lost) {
    ready_ = false;
    stable_observations_ = 0;
    debug_.stable_observations = 0;
    set_state(FireGateState::temp_lost);
  } else if (!sample.tracking.live_observation) {
    ready_ = false;
    stable_observations_ = 0;
    debug_.stable_observations = 0;
    set_state(FireGateState::track_not_ready);
  } else if (observation_age > observation_age_limit) {
    ready_ = false;
    stable_observations_ = 0;
    debug_.stable_observations = 0;
    set_state(FireGateState::observation_stale);
  } else {
    const auto learning_time = std::max(0.0, seconds_between(sample.now, learning_since_));
    debug_.warmup_remaining_ms = std::max(0.0, warmup_time_ - learning_time) * 1e3;
    if (prediction_unstable || learning_time < warmup_time_) {
      ready_ = false;
      set_state(FireGateState::warmup);
    } else if (stable_observations_ < required_stable_observations_) {
      ready_ = false;
      set_state(FireGateState::stabilizing);
    } else {
      ready_ = true;
      set_state(FireGateState::ready);
    }
  }

  debug_.ready = ready_;
  debug_.allow_fire = allow_fire();
}

void FireGate::update_no_target(std::chrono::steady_clock::time_point now)
{
  (void)now;
  has_track_ = false;
  track_epoch_ = 0;
  last_observation_seq_ = 0;
  stable_observations_ = 0;
  ready_ = mode_ == FireGateMode::off;
  debug_.track_epoch = 0;
  debug_.observation_seq = 0;
  debug_.observed_frames = 0;
  debug_.stable_observations = 0;
  debug_.prediction_change_deg = -1.0;
  debug_.observation_age_ms = 0.0;
  debug_.warmup_remaining_ms = 0.0;
  set_state(mode_ == FireGateMode::off ? FireGateState::disabled : FireGateState::no_target);
  debug_.ready = ready_;
  debug_.allow_fire = allow_fire();
}

bool FireGate::ready() const { return ready_; }

bool FireGate::allow_fire() const { return mode_ != FireGateMode::enforce || ready_; }

void FireGate::set_fire_result(bool raw_fire, bool actual_fire)
{
  debug_.raw_fire = raw_fire;
  debug_.actual_fire = actual_fire;
}

const FireGateDebug & FireGate::debug() const { return debug_; }

const char * FireGate::state_name(FireGateState state)
{
  switch (state) {
    case FireGateState::disabled:
      return "DISABLED";
    case FireGateState::no_target:
      return "NO_TARGET";
    case FireGateState::track_not_ready:
      return "TRACK_NOT_READY";
    case FireGateState::temp_lost:
      return "TEMP_LOST";
    case FireGateState::observation_stale:
      return "OBSERVATION_STALE";
    case FireGateState::warmup:
      return "WARMUP";
    case FireGateState::stabilizing:
      return "STABILIZING";
    case FireGateState::ready:
      return "READY";
  }
  return "UNKNOWN";
}

const char * FireGate::mode_name(FireGateMode mode)
{
  switch (mode) {
    case FireGateMode::off:
      return "off";
    case FireGateMode::shadow:
      return "shadow";
    case FireGateMode::enforce:
      return "enforce";
  }
  return "unknown";
}

}  // namespace auto_aim
