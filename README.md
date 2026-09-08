# Daedalus Vision Simulation

面向 RoboMaster 哨兵视觉算法的闭环仿真工程，将 Daedalus 仿真器和从
`burn-your-bridges` 拆出的 `sim-vision` 放在同一个独立仓库中。

## 目录

- `daedalus-simulator/`：场景、机器人运动、模拟相机、碰撞与命中统计。
- `sim-vision/`：目标识别、跟踪预测、火控决策和 PlotJuggler 调试数据。

两个子目录只是源码，不包含原仓库的 `.git`、编译产物或测试报告；本仓库统一管理版本。

## 运行环境

推荐 Ubuntu 24.04、NVIDIA Vulkan 驱动、Rust/Cargo、CMake、Ninja、OpenCV、
OpenVINO/ONNX Runtime 以及项目各自 README 中列出的开发库。

## 编译

```bash
cd daedalus-simulator
cargo build --release

cd ../sim-vision
cmake -S . -B build -G Ninja
cmake --build build --target vision_sim_runner vision_sim_runner_debug -j2
```

## 仿真与自瞄联调

终端 1：启动仿真器并输出 Talos 模拟相机画面。

```bash
cd /home/kop/daedalus-vision-simulation/daedalus-simulator
rm -f /tmp/talos_ipc_meta /tmp/talos_ipc_image_pool
DAEDALUS_FORCE_TALOS_CAPTURE=1 cargo run --release
```

终端 2：启动带调试画面的视觉程序。

```bash
cd /home/kop/daedalus-vision-simulation/sim-vision
./build/vision_sim_runner_debug
```

正式运行可改为：

```bash
./build/vision_sim_runner
```

默认配置已经面向仿真：真实哨兵参数以 `configs/sentry.yaml` 为基准，
仿真接口差异由 `configs/daedalus_overlay.yaml` 覆盖。这样算法主体和真机保持一致，
仿真专用的相机、通信及模型选择不会污染真机配置。

## 常用按键

- `F5`：装甲板自瞄开关
- `F6`：识别颜色切换
- `F7` / `F8`：小能量机关 / 大能量机关模式
- `F9`：普通匀速自旋
- `F10` / `F11` / `F12`：自旋相关控制

具体控制、配置边界、火控逻辑和调试说明分别见两个子项目的 README。

