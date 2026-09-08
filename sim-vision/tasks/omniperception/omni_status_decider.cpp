#include "omni_status_decider.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>

#include "tasks/auto_aim/ignore_num.hpp"

namespace omniperception
{
OmniStatusDecider::OmniStatusDecider(const std::string & config_path)
: config_enemy_color_(auto_aim::Color::blue), priority_seen_count_(0)
{
  auto yaml = YAML::LoadFile(config_path);
  config_enemy_color_ =
    yaml["enemy_color"].as<std::string>() == "red" ? auto_aim::Color::red : auto_aim::Color::blue;
  ignored_armors_ = auto_aim::read_ignore_num(yaml, "OmniStatusDecider");
  priority_armors_ = auto_aim::read_priority_num(yaml, "OmniStatusDecider");
}

int OmniStatusDecider::update(std::list<auto_aim::Armor> armors, uint8_t camp)
{
  const auto enemy_color = enemy_color_from_camp(camp);
  armors.remove_if([&](const auto_aim::Armor & armor) { return armor.color != enemy_color; });
  auto_aim::remove_ignored_armors(armors, ignored_armors_);

  const bool found = !armors.empty();
  const bool found_priority =
    found && std::any_of(armors.begin(), armors.end(), [&](const auto_aim::Armor & armor) {
      return auto_aim::contains_armor_name(armor.name, priority_armors_);
    });

  priority_seen_count_ = found_priority ? priority_seen_count_ + 1 : 0;
  if (priority_seen_count_ > 5) return priority_target_stable;

  int flag=0;
  if (!armors.empty()) {
    // std::cout << "armors[0].center.x " << armors.front().center.x << std::endl;
    if(armors.front().center.x<640) flag = -1;
    else flag = 1;

}
  return flag;
}

void OmniStatusDecider::reset() { priority_seen_count_ = 0; }

auto_aim::Color OmniStatusDecider::enemy_color_from_camp(uint8_t camp) const
{
  switch (camp) {
    case 0:
      return auto_aim::Color::blue;
    case 1:
      return auto_aim::Color::red;
    default:
      return config_enemy_color_;
  }
}
}  // namespace omniperception
