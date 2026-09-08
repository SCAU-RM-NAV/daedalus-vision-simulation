# Agent 工作硬约束

- 本项目的正确仓库路径是 `B:\GitHub\burn-your-bridges`。
- 以后处理本项目相关代码、配置、文档时，必须先确认当前工作目录是 `B:\GitHub\burn-your-bridges`。
- 不要在 `B:\GitHub\sp_vision_25` 或其他相似仓库中执行本项目的代码改动。

# sp_vision_25 工程 Agent 笔记

这个仓库是同济大学 SuperPower 战队 25 赛季 RoboMaster 自瞄视觉工程。核心是 C++17 + CMake，负责在上位机完成装甲板检测、PnP 解算、EKF 整车跟踪、弹道/轨迹规划和开火决策，再把云台与射击命令发给下位机。

## 目标环境

正式调试建议使用 Ubuntu 22.04 的 NUC 或等价 Linux 小电脑。当前工程不适合在 Windows 原生环境直接跑实车链路，因为代码依赖 Linux SocketCAN、Linux 串口设备、相机 SDK `.so`、OpenVINO Linux Runtime 和 udev 设备规则。

推荐硬件/软件基线：
  
- 系统：Ubuntu 22.04
- 运算平台：NUC12 或同等级 x86_64 Linux 小电脑
- 相机：海康 HikRobot MV-CS016-10UC 或迈德威视 MindVision
- 下位机：RoboMaster C 板或兼容自定义控制板
- 通信：新方案为 MicroUSB 虚拟串口 `/dev/gimbal`，旧方案为 USB2CAN / SocketCAN `can0`
- 推理：OpenVINO，仓库 CMake 默认写死 `/opt/intel/openvino_2024.6.0/runtime/cmake`

基础依赖：

```bash
sudo apt update
sudo apt install -y \
  git g++ cmake can-utils \
  libopencv-dev libfmt-dev libeigen3-dev libspdlog-dev \
  libyaml-cpp-dev libusb-1.0-0-dev nlohmann-json3-dev \
  libceres-dev openssh-server screen
```

还需要安装：

- HikRobot MVS SDK 或 MindVision SDK
- OpenVINO 2024.x，最好安装到 `/opt/intel/openvino_2024.6.0`
- 如果使用 `device: GPU`，还要安装 Intel GPU OpenCL/Level Zero Runtime

## 编译与离线跑通

先创建日志和录制目录，否则 logger 可能因为 `logs/` 不存在而失败。

```bash
mkdir -p logs records
source /opt/intel/openvino_2024.6.0/setupvars.sh
cmake -B build -S .
cmake --build build -j$(nproc)
```

第一步只跑离线 demo，验证 OpenVINO、模型、检测-跟踪-瞄准算法链路是否可用：

```bash
./build/auto_aim_test
```

默认读取：

- 视频：`assets/demo/demo.avi`
- 四元数文本：`assets/demo/demo.txt`
- 配置：`configs/demo.yaml`

OpenCV 窗口中按 `q` 退出。

如果没有 Intel GPU 环境，先把配置里的 `device: GPU` 改为 `device: CPU`。

## 关键目录和入口

- `assets/`：OpenVINO 模型、ONNX 分类器、demo 视频
- `configs/`：不同机器人/场景的 YAML 配置
- `io/`：硬件抽象层，相机、串口云台、CAN、USB 摄像头
- `tasks/auto_aim/`：自瞄算法主逻辑
- `tasks/auto_buff/`：打符逻辑
- `src/`：实车主程序入口
- `tests/`：模块测试程序
- `calibration/`：相机内参、手眼标定、采集工具

常用可执行程序：

- `auto_aim_test`：离线自瞄 demo
- `camera_test`：工业相机取流测试
- `camera_detect_test`：工业相机 + 检测器测试
- `gimbal_test`：新串口云台通信测试，可选开火位测试
- `cboard_test`：旧 CAN C 板通信测试
- `standard_mpc`：步兵新串口实车主程序，自瞄 + 打符 + MPC 轨迹规划
- `mt_standard`：旧 CAN 多线程实车主程序，自瞄 + 打符
- `auto_aim_debug_mpc`：新串口自瞄调试程序，带重投影窗口和 PlotJuggler 数据

## 配置文件要点

建议复制一份专用配置，不要直接改模板：

```bash
cp configs/standard3.yaml configs/my_robot.yaml
```

必须根据实车修改的字段：

- `enemy_color`：敌方颜色，红方机器人打蓝色装甲，蓝方机器人打红色装甲
- `yolo_name`：`yolov5` / `yolov8` / `yolo11`
- `device`：先用 `CPU` 跑通，再改 `GPU`
- `camera_name`：`hikrobot` 或 `mindvision`
- `exposure_ms`、`gain` 或 `gamma`：相机曝光和增益
- `vid_pid`：相机 USB VID/PID
- `camera_matrix`、`distort_coeffs`：相机内参和畸变
- `R_gimbal2imubody`：云台坐标系到 IMU body 坐标系的转换
- `R_camera2gimbal`、`t_camera2gimbal`：手眼标定结果
- `yaw_offset`、`pitch_offset`：实弹落点补偿
- `high_speed_delay_time`、`low_speed_delay_time`：预测时间补偿
- `auto_fire`：旧 CAN 逻辑是否自动开火
- `fire_thresh`：新串口 MPC 规划器开火阈值
- `com_port`：新串口协议设备名，默认 `/dev/gimbal`
- `can_interface`、`quaternion_canid`、`bullet_speed_canid`、`send_canid`：旧 CAN 协议参数

## 本地到实车的推荐调试顺序

1. 离线 demo：

```bash
./build/auto_aim_test
```

2. 相机取流：

```bash
./build/camera_test -c=configs/camera.yaml -d
```

3. 相机检测：

```bash
./build/camera_detect_test configs/my_robot.yaml
```

4. 新串口通信测试：

```bash
./build/gimbal_test configs/my_robot.yaml
```

如果要测试开火位，才加 `-f`，并且必须先清空弹丸、断摩擦轮或做好安全隔离。

5. 旧 CAN 通信测试：

```bash
sudo ip link set can0 up type can bitrate 1000000
./build/cboard_test configs/my_robot.yaml
```

6. 新串口实车自瞄调试：

```bash
./build/auto_aim_debug_mpc configs/my_robot.yaml
```

7. 新串口实车主程序：

```bash
./build/standard_mpc configs/my_robot.yaml
```

8. 旧 CAN 实车主程序：

```bash
./build/mt_standard configs/my_robot.yaml
```

## 新串口协议

代码位置：

- `io/gimbal/gimbal.hpp`
- `io/gimbal/gimbal.cpp`

下位机发给视觉 `GimbalToVision`：

- 帧头：`'S', 'P'`
- `mode`：`0` 空闲，`1` 自瞄，`2` 小符，`3` 大符
- `q[4]`：四元数，顺序为 `wxyz`
- `yaw`、`yaw_vel`、`pitch`、`pitch_vel`
- `bullet_speed`
- `bullet_count`
- `crc16`

视觉发给下位机 `VisionToGimbal`：

- 帧头：`'S', 'P'`
- `mode`：`0` 不控制，`1` 控云台不开火，`2` 控云台并开火
- `yaw`、`yaw_vel`、`yaw_acc`
- `pitch`、`pitch_vel`、`pitch_acc`
- `crc16`

注意：当前 `Gimbal` 代码只 `setPort(com_port)` 后 `open()`，没有显式设置波特率。`serial` 库默认构造值是 9600。如果下位机不是 USB CDC 自动协商，建议给 `Gimbal` 增加 `baudrate` 配置并调用 `serial_.setBaudrate()`。

## 旧 CAN 协议

代码位置：

- `io/cboard.hpp`
- `io/cboard.cpp`
- `io/socketcan.hpp`

接收四元数帧：

- CAN ID：`quaternion_canid`
- 数据：`x y z w`，每项 `int16 / 1e4`

接收弹速/模式帧：

- CAN ID：`bullet_speed_canid`
- `bullet_speed = int16 / 1e2`
- `mode = frame.data[2]`
- `shoot_mode = frame.data[3]`
- `ft_angle = int16 / 1e4`

发送视觉命令：

- CAN ID：`send_canid`
- `data[0]`：control
- `data[1]`：shoot
- `data[2:3]`：yaw，`int16 = yaw * 1e4`
- `data[4:5]`：pitch，`int16 = pitch * 1e4`
- `data[6:7]`：horizon_distance，`int16 = distance * 1e4`

## 标定流程

标定入口在 `calibration/`。

1. 修改 `configs/calibration.yaml`：

- `pattern_cols`
- `pattern_rows`
- `center_distance_mm`
- 相机类型和曝光
- CAN ID
- 初始 `R_gimbal2imubody`

2. 采集图像和四元数：

```bash
./build/capture configs/calibration.yaml -o=assets/img_with_q
```

窗口中按 `s` 保存一组图像和四元数，按 `q` 退出。

注意：当前 `capture.cpp` 使用的是 `io::CBoard`，也就是旧 CAN 读四元数。如果机器人只走新串口协议，需要把采集程序改成 `io::Gimbal` 读 `q()`。

3. 相机内参：

```bash
./build/calibrate_camera assets/img_with_q -c=configs/calibration.yaml
```

把输出的 `camera_matrix`、`distort_coeffs` 写回 `configs/my_robot.yaml`。

4. 手眼标定：

```bash
./build/calibrate_robotworld_handeye assets/img_with_q -c=configs/calibration.yaml
```

把输出的 `R_gimbal2imubody`、`R_camera2gimbal`、`t_camera2gimbal` 写回 `configs/my_robot.yaml`。

5. 验证重投影：

```bash
./build/auto_aim_debug_mpc configs/my_robot.yaml
```

看装甲板重投影点是否贴合实际灯条。重投影明显偏，优先检查坐标系和手眼标定，不要先靠 offset 硬补。

## 上车射击安全顺序

1. 首次上电只跑通信，不装弹。
2. 先把 `auto_fire: false` 或 `fire_thresh: 0`，确保视觉不会主动触发发射。
3. 静态目标下只控云台，确认 yaw/pitch 方向正确。
4. 开 PlotJuggler 监听 UDP `127.0.0.1:9870`，观察 `gimbal_yaw`、`plan_yaw`、`target_yaw`、`fire` 等字段。
5. 移动靶不开火，调下位机云台控制和 `max_yaw_acc/max_pitch_acc`。
6. 低射频、低功率、安全靶场实弹测试。
7. 逐步调：

- `yaw_offset`
- `pitch_offset`
- `low_speed_delay_time`
- `high_speed_delay_time`
- `fire_thresh`
- `first_tolerance`
- `second_tolerance`

## 已知坑

- `autostart.sh` 当前调用 `./watchdog.sh`，但仓库里没有 `watchdog.sh`。部署自启前要改成实际启动命令，例如：

```bash
./build/standard_mpc configs/my_robot.yaml
```

- CMake 中 OpenVINO 路径被硬编码到 `/opt/intel/openvino_2024.6.0/runtime/cmake`。
- `logs/` 不存在时 logger 可能创建文件失败，建议启动前 `mkdir -p logs records`。
- `standard.cpp` 使用旧 CAN `CBoard`，但没有调用 `Shooter`，基本只发瞄准命令；旧 CAN 要自动开火优先看 `mt_standard`。
- `standard_mpc` 使用新串口 `Gimbal` 和 `Planner`，开火由 `Planner::plan()` 中的 `fire_thresh` 决定。
- `MultiThreadDetector` 默认按 640 输入处理，注释里写了暂不支持 ROI，调试 ROI 时不要默认相信多线程路径。
- Tracker 中图像中心写死过 `1440/2, 1080/2`，如果相机分辨率变化大，目标优先级排序可能不理想。
- `camera_name` 只支持 `mindvision` 和 `hikrobot`，普通 USB 摄像头测试走 `USBCamera` 测试程序，不是主 `io::Camera`。

## 推荐给后续 Agent 的处理原则

- 先读 `configs/my_robot.yaml` 和正在使用的主程序入口，再判断运行路径。
- 不要先改算法，先确认相机、IMU 时间戳、坐标系、通信协议和弹速都正常。
- 上车改参数时一次只改少量参数，并保留每次实弹测试的配置版本。
- 涉及开火测试时默认把安全放在第一位：无弹、断摩擦轮、低射频、安全朝向、有人确认。
- 如果用户仍在 Windows 环境，优先建议准备 Ubuntu 22.04 NUC；Windows 只适合读代码和准备配置，WSL 只适合离线 demo。
