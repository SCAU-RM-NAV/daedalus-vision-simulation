#ifndef AUTO_AIM__TRACKING_INFO_HPP
#define AUTO_AIM__TRACKING_INFO_HPP

#include <chrono>
#include <cstdint>

namespace auto_aim
{

struct TrackingInfo
{
  uint64_t track_epoch = 0;
  uint64_t observation_seq = 0;
  int observed_frames = 0;
  bool confirmed = false;
  bool temp_lost = false;
  bool live_observation = false;
  std::chrono::steady_clock::time_point first_observed{};
  std::chrono::steady_clock::time_point last_observed{};
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKING_INFO_HPP
