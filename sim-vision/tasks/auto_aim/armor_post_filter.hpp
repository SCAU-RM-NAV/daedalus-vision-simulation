#ifndef AUTO_AIM__ARMOR_POST_FILTER_HPP
#define AUTO_AIM__ARMOR_POST_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"

namespace auto_aim
{

class ArmorPostFilter
{
public:
  explicit ArmorPostFilter(const std::string & config_path);

  bool keep(const Armor & armor, const cv::Mat & bgr_img) const;
  bool check_geometry(const Armor & armor) const;
  bool check_color(const Armor & armor, const cv::Mat & bgr_img) const;

private:
  bool enabled_;
  bool geometry_enabled_;
  bool color_enabled_;

  double min_armor_ratio_;
  double max_armor_ratio_;
  double max_rectangular_error_;
  double min_light_length_;
  double min_light_ratio_;
  double max_center_angle_;
  double max_light_angle_diff_;

  double color_dominance_;
  double color_min_ratio_;
  int color_sample_stride_;
  int color_min_samples_;

  bool line_matches_color(
    const cv::Point2f & start, const cv::Point2f & end, const cv::Mat & bgr_img,
    Color expected_color) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_POST_FILTER_HPP
