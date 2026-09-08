#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
  "{@config-path   | configs/sentry.yaml    | yaml配置文件的路径}"
  "{tradition t    |  false                 | 是否使用传统方法识别}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto use_tradition = cli.get<bool>("tradition");

  tools::Exiter exiter;

  const auto yaml = tools::load(config_path);
  YAML::Node camera_yaml;
  if (yaml["camera_name"]) {
    camera_yaml = yaml;
  } else if (yaml["combat_camera"]) {
    camera_yaml = yaml["combat_camera"];
    if (!camera_yaml["camera_user_id"]) {
      camera_yaml["camera_user_id"] = "A";
    }
  } else {
    tools::logger()->error("[camera_detect_test] camera_name or combat_camera not found in config.");
    return 1;
  }

  io::Camera camera(camera_yaml);
  auto_aim::Detector detector(config_path, true);
  auto_aim::YOLO yolo(config_path, true);

  std::chrono::steady_clock::time_point timestamp;
  auto report_stamp = std::chrono::steady_clock::now();
  double detect_time_sum = 0.0;
  int frame_count = 0;

  while (!exiter.exit()) {
    cv::Mat img;
    std::list<auto_aim::Armor> armors;

    camera.read(img, timestamp);

    if (img.empty()) break;

    auto last = std::chrono::steady_clock::now();

    if (use_tradition)
      armors = detector.detect(img);
    else
      armors = yolo.detect(img);

    auto now = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(now, last);
    detect_time_sum += dt;
    frame_count++;

    auto report_dt = tools::delta_time(now, report_stamp);
    if (report_dt >= 1.0) {
      tools::logger()->info(
        "detect avg: {:.2f} fps, loop: {:.2f} fps", frame_count / detect_time_sum,
        frame_count / report_dt);
      report_stamp = now;
      detect_time_sum = 0.0;
      frame_count = 0;
    }

    auto key = cv::waitKey(33);
    if (key == 'q') break;
  }

  return 0;
}
