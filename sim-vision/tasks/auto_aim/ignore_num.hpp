#ifndef AUTO_AIM__IGNORE_NUM_HPP
#define AUTO_AIM__IGNORE_NUM_HPP

#include <algorithm>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "armor.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
inline bool to_armor_name(int num, ArmorName & name)
{
  if (num <= 0 || num > static_cast<int>(ArmorName::base) + 1) return false;
  name = static_cast<ArmorName>(num - 1);
  return true;
}

inline std::vector<ArmorName> read_armor_nums(
  const YAML::Node & yaml, const std::string & key, const std::string & owner)
{
  std::vector<ArmorName> armors;
  const auto node = yaml[key];
  if (!node) return armors;

  auto append = [&](const YAML::Node & item) {
    int num = 0;
    try {
      num = item.as<int>();
    } catch (const YAML::Exception & e) {
      tools::logger()->warn("[{}] invalid {} entry: {}", owner, key, e.what());
      return;
    }

    ArmorName name;
    if (!to_armor_name(num, name)) {
      tools::logger()->warn("[{}] {} supports 1..8, got {}", owner, key, num);
      return;
    }
    armors.push_back(name);
  };

  if (node.IsSequence()) {
    for (const auto & item : node) append(item);
  } else {
    append(node);
  }

  std::sort(armors.begin(), armors.end());
  armors.erase(std::unique(armors.begin(), armors.end()), armors.end());

  if (!armors.empty()) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < armors.size(); ++i) {
      const auto name = armors[i];
      if (i > 0) oss << ", ";
      oss << static_cast<int>(name) + 1 << "(" << ARMOR_NAMES.at(static_cast<std::size_t>(name))
          << ")";
    }
    tools::logger()->info("[{}] {}: {}", owner, key, oss.str());
  }

  return armors;
}

inline std::vector<ArmorName> read_ignore_num(const YAML::Node & yaml, const std::string & owner)
{
  return read_armor_nums(yaml, "ignore_num", owner);
}

inline std::vector<ArmorName> read_priority_num(const YAML::Node & yaml, const std::string & owner)
{
  return read_armor_nums(yaml, "priority_num", owner);
}

inline bool contains_armor_name(ArmorName name, const std::vector<ArmorName> & armors)
{
  return std::find(armors.begin(), armors.end(), name) != armors.end();
}

inline bool is_ignored_armor(ArmorName name, const std::vector<ArmorName> & ignored)
{
  return contains_armor_name(name, ignored);
}

inline void remove_ignored_armors(
  std::list<Armor> & armors, const std::vector<ArmorName> & ignored)
{
  if (ignored.empty()) return;
  armors.remove_if([&](const Armor & armor) { return is_ignored_armor(armor.name, ignored); });
}

inline void set_configured_priority(
  std::list<Armor> & armors, const std::vector<ArmorName> & priority)
{
  if (priority.empty()) return;
  for (auto & armor : armors) {
    armor.priority =
      contains_armor_name(armor.name, priority) ? ArmorPriority::first : ArmorPriority::second;
  }
}

}  // namespace auto_aim

#endif  // AUTO_AIM__IGNORE_NUM_HPP
