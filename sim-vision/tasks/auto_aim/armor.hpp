#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <array>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace auto_aim
{
enum Color
{
  red,
  blue,
  extinguish
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish"};

enum ArmorType
{
  big,
  small
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

enum ArmorPriority
{
  first = 1,
  second,
  third,
  forth,
  fifth
};

// clang-format off
inline constexpr std::array<std::tuple<Color, ArmorName, ArmorType>, 21> armor_properties = {{
  {red, one, big},            {red, two, small},            {red, three, small},     
  {red, four, small},         {red, sentry, small},         {red, outpost, small},   
  {red, base, small},
  {blue, one, big},           {blue, two, small},           {blue, three, small},     
  {blue, four, small},        {blue, sentry, small},        {blue, outpost, small},  
  {blue, base, small},
  {extinguish, one, big},     {extinguish, two, small},     {extinguish, three, small},
  {extinguish, four, small},  {extinguish, sentry, small},  {extinguish, outpost, small}, 
  {extinguish, base, small}
}};
// clang-format on

struct Lightbar
{
  std::size_t id;
  Color color;
  cv::Point2f center, top, bottom, top2bottom;
  std::vector<cv::Point2f> points;
  double angle, angle_error, length, width, ratio;
  cv::RotatedRect rotated_rect;

  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  Lightbar() {};
};

struct Armor
{
  Color color;
  Lightbar left, right;     //used to be const
  cv::Point2f center;       // 不是对角线交点，不能作为实际中心！
  cv::Point2f center_norm;  // 归一化坐标
  std::vector<cv::Point2f> points;

  double ratio;              // 两灯条的中点连线与长灯条的长度之比
  double side_ratio;         // 长灯条与短灯条的长度之比
  double rectangular_error;  // 灯条和中点连线所成夹角与π/2的差值

  ArmorType type;
  ArmorName name;
  ArmorPriority priority = ArmorPriority::fifth;
  int class_id;
  cv::Rect box;
  cv::Mat pattern;
  double confidence;
  bool duplicated;

  Eigen::Vector3d xyz_in_gimbal;  // 单位：m
  Eigen::Vector3d xyz_in_world;   // 单位：m
  Eigen::Vector3d pnp_xyz_in_camera = Eigen::Vector3d::Constant(
    std::numeric_limits<double>::quiet_NaN());  // Raw solvePnP result, in camera frame.
  Eigen::Vector3d ypr_in_gimbal;  // 单位：rad
  Eigen::Vector3d ypr_in_world;   // 单位：rad
  Eigen::Vector3d ypd_in_world;   // 球坐标系

  double yaw_raw;  // rad

  Armor(const Lightbar & left, const Lightbar & right);
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints);
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
    cv::Point2f offset);
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints);
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP
