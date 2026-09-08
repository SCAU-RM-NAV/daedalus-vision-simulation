#ifndef OMNIPERCEPTION__OMNI_STATUS_DECIDER_HPP
#define OMNIPERCEPTION__OMNI_STATUS_DECIDER_HPP

#include <cstdint>
#include <list>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{
class OmniStatusDecider
{
public:
  static constexpr int no_target = 0;
  static constexpr int target_found = 1;
  static constexpr int priority_target_stable = 2;

  explicit OmniStatusDecider(const std::string & config_path);

  int update(std::list<auto_aim::Armor> armors, uint8_t camp);
  void reset();

private:
  auto_aim::Color enemy_color_from_camp(uint8_t camp) const;

  auto_aim::Color config_enemy_color_;
  std::vector<auto_aim::ArmorName> ignored_armors_;
  std::vector<auto_aim::ArmorName> priority_armors_;
  int priority_seen_count_;
};
}  // namespace omniperception

#endif  // OMNIPERCEPTION__OMNI_STATUS_DECIDER_HPP
