#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

bool find_calibration_pattern(
  const cv::Mat & img, const cv::Size & pattern_size, const std::string & pattern_type,
  std::vector<cv::Point2f> & points)
{
  if (pattern_type == "chessboard") {
    cv::Mat gray;
    if (img.channels() == 1)
      gray = img;
    else
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    const bool success = cv::findChessboardCorners(
      gray, pattern_size, points, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    if (success) {
      cv::cornerSubPix(
        gray, points, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
    }
    return success;
  }

  return cv::findCirclesGrid(img, pattern_size, points, cv::CALIB_CB_SYMMETRIC_GRID);
}

void capture_loop(
  const std::string & config_path, const std::string & output_folder,
  const std::function<Eigen::Quaterniond(std::chrono::steady_clock::time_point)> & read_q)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto pattern_cols = yaml["pattern_cols"].as<int>();
  const auto pattern_rows = yaml["pattern_rows"].as<int>();
  const auto pattern_type = yaml["pattern_type"] ? yaml["pattern_type"].as<std::string>() : "circles";
  const auto exposure_ms = yaml["exposure_ms"].as<double>();
  const auto exposure_center_offset =
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double, std::milli>(exposure_ms * 0.5));
  const cv::Size pattern_size(pattern_cols, pattern_rows);
  tools::logger()->info(
    "Image q timestamp offset: -{:.3f} ms (exposure center)", exposure_ms * 0.5);

  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    const auto img_time = timestamp - exposure_center_offset;
    Eigen::Quaterniond q = read_q(img_time);

    // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

    std::vector<cv::Point2f> centers_2d;
    auto success = find_calibration_pattern(img, pattern_size, pattern_type, centers_2d);
    cv::drawChessboardCorners(img_with_ypr, pattern_size, centers_2d, success);  // 显示识别结果
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

    // 按“s”保存图片和对应四元数，按“q”退出程序
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  // 离开该作用域时，camera和cboard会自动关闭
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 新建输出文件夹
  std::filesystem::create_directory(output_folder);

  tools::logger()->info("Use calibration board settings from {}", config_path);
  // 主循环，保存图片和对应四元数
  const auto yaml = YAML::LoadFile(config_path);
  const auto pose_source = yaml["pose_source"] ? yaml["pose_source"].as<std::string>() : "gimbal";
  tools::logger()->info("Use pose source: {}", pose_source);

  if (pose_source == "cboard") {
    io::CBoard cboard(config_path);
    capture_loop(config_path, output_folder, [&](auto timestamp) { return cboard.imu_at(timestamp); });
  } else if (pose_source == "gimbal") {
    io::Gimbal gimbal(config_path);
    capture_loop(config_path, output_folder, [&](auto timestamp) { return gimbal.q(timestamp); });
  } else {
    tools::logger()->error("Invalid pose_source: {}, expected 'gimbal' or 'cboard'", pose_source);
    return 1;
  }

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
