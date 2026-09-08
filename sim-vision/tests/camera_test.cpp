// ./build/camera_test -c configs/camera.yaml -d 单摄车用这个,指海康驱动枚举到的第一台相机
// ./build/camera_test -c configs/camera.yaml --camera=idle -d 双摄车长焦用这个
// ./build/camera_test -c configs/camera.yaml --camera=combat -d 双摄车短焦用这个

#include "io/camera.hpp"

#include <opencv2/opencv.hpp>
#include <string>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/camera.yaml | yaml配置文件路径 }"
  "{d display      |                     | 显示视频流       }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  auto config_path = cli.get<std::string>("config-path");
  std::string camera_name = "root";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const std::string prefix = "--camera=";
    if (arg.rfind(prefix, 0) == 0) {
      camera_name = arg.substr(prefix.size());
    } else if (arg == "--camera" && i + 1 < argc) {
      camera_name = argv[++i];
    }
  }
  auto display = cli.has("display");

  auto yaml = tools::load(config_path);
  YAML::Node camera_yaml;
  if (camera_name == "combat") {
    camera_yaml = yaml["combat_camera"] ? yaml["combat_camera"] : yaml;
  } else if (camera_name == "idle") {
    if (!yaml["idle_camera"]) {
      tools::logger()->error("[camera_test] idle_camera not found in config.");
      return 1;
    }
    camera_yaml = yaml["idle_camera"];
  } else if (camera_name == "root") {
    camera_yaml = yaml;
  } else {
    tools::logger()->error("[camera_test] Unknown camera option: {}", camera_name);
    return 1;
  }

  io::Camera camera(camera_yaml);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    camera.read(img, timestamp);

    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;

    tools::logger()->info("{:.2f} fps", 1 / dt);

    if (!display) continue;
    cv::imshow("img", img);
    if (cv::waitKey(1) == 'q') break;
  }
}
