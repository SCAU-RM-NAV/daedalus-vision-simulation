# Agent Change Log

## 2026-08-22 Orin ONNX Runtime backend groundwork

- Added the optional CMake switch `SP_VISION_ENABLE_ONNXRUNTIME` and discovery of a local ONNX Runtime installation via `ONNXRUNTIME_ROOT`. OpenVINO remains a build dependency until the other OpenVINO-only detectors are migrated.
- Added runtime backend selection for the sentry's synchronous YOLOv5 detector. `inference_backend: openvino` preserves the NUC path; `onnxruntime_cuda` uses ONNX Runtime CUDA when compiled with the optional switch.
- Added `yolov5_onnx_model_path` to `configs/sentry.yaml`. It intentionally defaults to empty: the existing OpenVINO IR (`yolov5.xml`/`.bin`) is not a usable ONNX model and must not be silently substituted.
- This is source/config groundwork only. It has not been built on an Orin, and the buff detector still requires a separate ONNX Runtime migration.

## 2026-08-24 Orin OpenCV DNN CUDA compatibility

- Inspection of the previously working Nano project showed that it used OpenCV DNN CUDA to run `0526.onnx`; it did not use ONNX Runtime.
- Added `inference_backend: opencv_dnn_cuda` for YOLOv5 and made OpenVINO optional for auto-aim-only builds. The NUC OpenVINO implementation is unchanged.
- Without OpenVINO, YOLOv8/YOLO11, buff, omni and full sentry builds are deliberately unavailable. `camera_detect_test` can be used to validate the Orin camera plus YOLOv5 path first.
- Kept the newer HikRobot implementation unchanged: it selects the bundled `io/hikrobot/lib/arm64` library on Nano and contains more robust camera selection/cleanup than the verified older version. Made the unrelated OpenVINO link in `io` conditional so it no longer blocks an Orin camera plus YOLOv5 build.
- Added the missing standard `<numeric>` include for `std::accumulate` in the detector; GCC 11 on Nano correctly rejected the previous implicit-header assumption.
- Restored USB-only HikRobot device enumeration. The prior Nano-proven code used `MV_USB_DEVICE`; the newer combined USB/GigE enumeration failed on the USB3 camera with `0x80000006`.
- Added the `HIKROBOT_LIBRARY_DIR` CMake cache option. It selects a Nano-specific MVS library directory at link time and records it in the executable's runtime search path, while retaining the repository's architecture-specific RM library as the default.
- Changed the custom HikRobot SDK selection to link the requested library by absolute path rather than with a search directory, so Nano builds cannot silently resolve to the RM ARM64 library.
- Set `configs/sentry.yaml` to default to the Orin Nano OpenCV DNN CUDA backend and the repository's `assets/best2-sim.onnx` model, as requested. NUC deployments must explicitly switch `inference_backend` back to `openvino`.

这个文件专门记录 Codex 为了构建、调试、部署本工程而建议或执行过的源码/配置改动。

原则：

- 只记录会改变仓库文件内容的操作。
- 不记录纯命令查询，例如 `lsusb`、`groups`、`dpkg -l`。
- 不记录系统级安装命令，除非它要求同步修改仓库文件。
- 每条记录写清楚原因、影响范围、是否是临时改动。

## 2026-05-23 NUC Ubuntu 22.04 编译准备

### OpenVINO 路径修正

背景：

- NUC 上已有 OpenVINO，实际安装路径是 `/opt/intel/openvino_2024.2.0`。
- 工程里的 `CMakeLists.txt` 原本写死为 `/opt/intel/openvino_2024.6.0/runtime/cmake/`。
- 如果不修正，CMake 可能找不到 `OpenVINOConfig.cmake`。

建议/执行过的改动：

```bash
cp CMakeLists.txt CMakeLists.txt.bak
sed -i 's#/opt/intel/openvino_2024.6.0/runtime/cmake/#/opt/intel/openvino_2024.2.0/runtime/cmake#' CMakeLists.txt
```

当前更稳的后续改法：

```cmake
set(OpenVINO_DIR "$ENV{OpenVINO_DIR}")
```

或者在 CMake 命令里传入：

```bash
cmake -S . -B build \
  -DOpenVINO_DIR=/opt/intel/openvino_2024.2.0/runtime/cmake
```

### 子目录 OpenVINO 路径

背景：

- `tasks/auto_aim/CMakeLists.txt` 里也有一处写死的 OpenVINO 2024.6.0 路径。
- 根目录修完后，子目录仍可能再次覆盖 `OpenVINO_DIR`。

建议修正：

```bash
sed -i 's#/opt/intel/openvino_2024.6.0/runtime/cmake#${OpenVINO_DIR}#g' CMakeLists.txt tasks/auto_aim/CMakeLists.txt
```

注意：

- 这个写法是为了让工程使用当前终端 `source /opt/intel/openvino_2024.2.0/setupvars.sh` 后得到的 `OpenVINO_DIR`。
- 后续更推荐把 CMake 文件改成“不写死具体 OpenVINO 版本”，避免换机器后再次出错。

### fmt / spdlog 版本冲突

背景：

- 编译时报错：

```text
/usr/include/spdlog/common.h: error: 'basic_runtime' is not a member of 'fmt'
```

判断：

- 这是系统 `spdlog` 与 `fmt` 版本不匹配。
- Ubuntu 22.04 apt 的 `fmt 8.1.1` 对当前 `spdlog` 头文件来说偏旧。

建议处理：

- 安装新版 `fmt 10.2.1` 到 `/usr/local`。
- 这一步属于系统依赖安装，不直接改仓库源码。
- 重新 CMake 时显式指定：

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/usr/local \
  -Dfmt_DIR=/usr/local/lib/cmake/fmt
```

### Hikrobot MVS runtime for camera_test

Background:

- `lsusb` detected the camera as `2bdf:0001 Hikrobot MV-CS016-10UC`.
- `camera_test` repeatedly failed at SDK enumeration:

```text
MV_CC_EnumDevices failed: 0x80000006
```

- In Hikrobot SDK headers, `0x80000006` is `MV_E_RESOURCE`, meaning resource allocation failed.
- The NUC has the official Hikrobot MVS SDK installed under `/opt/MVS`.

Cause:

- Linux could see the USB device, but the runtime environment used by `camera_test` did not fully use the official MVS runtime stack.
- The project also contains a local `io/hikrobot/lib/amd64/libMvCameraControl.so`; using only that local library can miss or mismatch MVS runtime components such as U3V transport libraries / CTI producers.

Working runtime setup:

```bash
source /opt/intel/openvino_2024.2.0/setupvars.sh
export LD_LIBRARY_PATH=/opt/MVS/lib/64:$PWD/io/hikrobot/lib/amd64:$PWD/io/mindvision/lib/amd64:$LD_LIBRARY_PATH
./build/camera_test -c=configs/standard3.yaml
```

Verified:

- `camera_test` prints FPS after prioritizing `/opt/MVS/lib/64` in `LD_LIBRARY_PATH`.

Notes:

- This is an environment/runtime fix, not a source-code change.
- Keep `/opt/MVS/lib/64` before the project-local Hikrobot library path when running camera programs.

## 2026-05-23 Gimbal serial timeout

Background:

- `gimbal_test` printed `[Gimbal] First q received.`, proving the lower-board `SP` frame could be received and parsed.
- Immediately afterward it repeatedly printed `Too many errors, attempting to reconnect...`.

Cause:

- `io/gimbal/gimbal.cpp` opened the serial port without a timeout.
- `Gimbal::read_thread()` then treated temporary "no bytes available yet" reads as errors in a tight loop.
- The lower board sends roughly every 3 ms, but the upper loop could accumulate 5000 failed nonblocking reads before the next frame arrived.

Source change:

- Configured the gimbal serial port with baud/8N1/no-flow-control and `serial::Timeout::simpleTimeout(20)`.
- The bundled `serial` API requires `setTimeout()` to receive a non-const lvalue, so the timeout object is stored in a local variable before calling `setTimeout()`.
- Replaced the loop-count reconnect rule with a time-based rule: reconnect only after 500 ms without a valid CRC-passing packet.
- Added a 1 ms sleep after timed-out header/body reads to avoid CPU spinning and false reconnect storms.

Impact:

- This should stop reconnect spam while preserving reconnect behavior for a real cable/device loss.

## 2026-05-23 Standard3 camera intrinsics

Background:

- A ROS camera calibration YAML was provided for a 1280x1024 camera stream.
- The vision solver reads `camera_matrix` and `distort_coeffs` from `configs/standard3.yaml` as flat arrays.

Source change:

- Replaced `configs/standard3.yaml` camera intrinsics with the provided calibration:
  - `camera_matrix`: `[1746.43563, 0, 720.01496, 0, 1744.86104, 573.67393, 0, 0, 1]`
  - `distort_coeffs`: `[-0.064765, 0.070565, -0.000057, -0.000703, 0.000000]`

Notes:

- `projection_matrix` and `rectification_matrix` from the ROS calibration YAML are not used by this project.
- `R_camera2gimbal` and `t_camera2gimbal` are hand-eye extrinsics and were not changed.
- The intrinsics must match the actual camera output resolution; if the runtime image size differs from 1280x1024, reprojected boxes can shift.

## 2026-05-23 Standard3 camera extrinsics

Background:

- A ROS-style `odom2camera` parameter was provided:
  - `xyz`: interpreted as `[0.0, 0.030, 0.080]` meters.
  - `rpy`: interpreted as `[roll=0.0, pitch=0.0450, yaw=-0.015]` radians.

Source change:

- Converted that small mounting offset into this project's `R_camera2gimbal` convention.
- This project expects OpenCV camera coordinates to gimbal coordinates, so the conversion applies the ideal camera-to-gimbal basis first:
  - camera `x` right -> gimbal `-y`
  - camera `y` down -> gimbal `-z`
  - camera `z` forward -> gimbal `x`
- Updated `configs/standard3.yaml`:
  - `R_camera2gimbal`: `[-0.014999438, -0.044979753, 0.998875287, -0.999887502, 0.000674747, -0.014984253, 0.000000000, -0.998987671, -0.044984814]`
  - `t_camera2gimbal`: `[0.0, 0.030, 0.080]`

Notes:

- If the provided `xyz` string was not intended as `0.0 0.030 0.080`, adjust `t_camera2gimbal` before testing.
- If reprojection moves left/right opposite to expectation, flip the yaw sign and regenerate the rotation matrix.
## 2026-05-24 EKF adaptive angular process noise TODO

Background:

- Future field test may include targets that translate while starting/stopping small-top spin suddenly.
- The current target EKF uses a nearly constant angular velocity model:
  - state `a` is target body angle.
  - state `w` is target angular velocity.
  - normal target angular process noise is hard-coded as `v2 = 400` in `tasks/auto_aim/target.cpp`.
- A single fixed `v2` has a trade-off:
  - small `v2`: stable during steady motion, but slow during sudden spin start/stop.
  - large `v2`: reacts faster to sudden spin changes, but can make `w` and predicted aim jitter.

Deferred implementation idea:

- Keep a lower base `v2` during stable tracking.
- Temporarily raise `v2` when the EKF sees signs of spin acceleration or model mismatch, such as:
  - large angle residual / NIS spike.
  - frequent armor id jump.
  - measured `w` changing quickly over several frames.
- Decay `v2` back to the base value after several stable frames.
- Make the base/high `v2` and trigger thresholds yaml-configurable before tuning on recorded data.

Trial implementation:

- Added yaml-configurable EKF process-noise parameters:
  - `ekf_normal_v1`, `ekf_normal_v2`
  - `ekf_outpost_v1`, `ekf_outpost_v2`
  - `ekf_adaptive_v1`, `ekf_adaptive_v1_high`, `ekf_adaptive_v1_decay`
  - `ekf_adaptive_v1_residual_yaw`, `ekf_adaptive_v1_residual_pitch`, `ekf_adaptive_v1_residual_distance`
  - `ekf_adaptive_v2`, `ekf_adaptive_v2_high`, `ekf_adaptive_v2_decay`
  - `ekf_adaptive_v2_residual_angle`, `ekf_adaptive_v2_nis`
- `Tracker` now reads these values and passes them into each new `Target`.
- `Target` now uses the configured linear/angular process noise when building EKF `Q`.
- When adaptive `v1` is enabled for normal targets, `Target` raises linear process noise after yaw/pitch/distance residual spikes; then it decays back toward the base value.
- When adaptive `v2` is enabled for normal targets, `Target` raises angular process noise after a residual angle spike or NIS spike; then it decays back toward the base value.
- Added Plotter fields in debug/offline paths:
  - `process_v1`
  - `process_v2`
  - `adaptive_v1_boost`
  - `adaptive_v2_boost`
- Added a one-shot info log when adaptive `v1` boost is triggered, including `process_v1`, yaw residual, pitch residual, and distance residual.
- Added a one-shot info log when adaptive `v2` boost is triggered, including `process_v2`, residual angle, and NIS.
- Removed armor id switching / `last_id` changes from the adaptive `v2` trigger to avoid boosting during normal armor transitions.

## 2026-05-24 EKF adaptive linear process noise v1

Background:

- Field testing showed pure translation direction changes can lag even when the target is not small-top spinning.
- The previous adaptive `v2` only reacts to target body-angle / spin mismatch, so it does not necessarily trigger on a pure lateral or depth acceleration.

Source change:

- Added dynamic `v1` for normal targets. It boosts the EKF linear acceleration process noise when yaw, pitch, or distance residual exceeds yaml thresholds.
- The boosted `v1` decays back toward `ekf_normal_v1` using `ekf_adaptive_v1_decay`.
- `v1` and `v2` are independent now:
  - `v1`: translation / sudden direction change.
  - `v2`: spin angular acceleration / small-top start-stop.
- Added Plotter debug fields `process_v1` and `adaptive_v1_boost`.

Tuning note:

- If pure translation still lags, lower `ekf_adaptive_v1_residual_yaw` / `ekf_adaptive_v1_residual_pitch` or raise `ekf_adaptive_v1_high`.
- If the aim becomes jittery during stable tracking, raise the residual thresholds or lower `ekf_adaptive_v1_high`.

## 2026-05-25 Ubuntu 24.04 OpenVINO 2025.4.0 path

Background:

- The user's Ubuntu 24.04 machine has OpenVINO under `/opt/intel/openvino_2025` and `/opt/intel/openvino_2025.4.0`.
- This branch should target that machine directly for now.
- Previous CMake files mixed OpenVINO 2024.2 and 2024.6 paths, which could make CMake select the wrong install or fail to find `OpenVINOConfig.cmake`.

Source change:

- Set the root `OpenVINO_DIR` to `/opt/intel/openvino_2025.4.0/runtime/cmake`.
- Removed hard-coded `OpenVINO_DIR` assignments from task subdirectories.
- The task subdirectories now inherit the root setting, so `auto_aim`, `auto_buff`, and `omniperception` all use the same OpenVINO install.

Recommended usage:

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j$(nproc)
```

## 2026-05-25 fmt 10 ArmorName formatting

Background:

- Ubuntu 24.04 build used fmt v10 from `/usr/local/include/fmt`.
- fmt v10 rejects formatting project enums such as `auto_aim::ArmorName` unless a formatter specialization exists.
- Build failed in `Detector::save()` and YOLO save helpers when generating debug image filenames.

Source change:

- Changed filename formatting to use the existing `ARMOR_NAMES[armor.name]` string table instead of passing `armor.name` directly to `fmt::format`.
- Updated `detector.cpp`, `yolov5.cpp`, `yolov8.cpp`, and `yolo11.cpp`.

## 2026-05-25 camera_detect_test FPS smoothing

Background:

- `camera_detect_test` printed instantaneous `1 / detect_time` every frame.
- Single-frame timing jumps due to inference, drawing, display, and logging made the output noisy.

Source change:

- Changed `camera_detect_test` to print once per second.
- The output now includes:
  - `detect avg`: average detector-only processing FPS over the last reporting window.
  - `loop`: whole test loop FPS over the same window, including camera wait and display wait.

## 2026-05-25 Gimbal serial unplug/reflash crash guard

Background:

- While reflashing the lower controller or when the USB serial cable is loose, `/dev/ttyACM*` can disappear and reappear.
- The process crashed with `*** bit out of range 0 - FD_SETSIZE on fd_set ***`.
- This points to the serial library reaching `FD_SET(fd_, ...)` with an invalid descriptor, commonly caused by close/open racing with read/write during reconnect.

Source change:

- `send()` now returns early when the serial port is not open.
- `Gimbal::~Gimbal()` logs close errors instead of throwing during destruction.
- Added defensive fd range checks before `FD_SET` in the bundled Linux serial implementation. Invalid descriptors now throw `IOException`, which the gimbal read/write paths can catch, instead of letting glibc abort the process.
- Removed the earlier `Gimbal`-level serial mutex because it serialized `read()` and `send()` and could reduce command send rate. The remaining fd guard has negligible per-frame overhead.

Test note:

- Rebuild, run `gimbal_test` or the main program, then unplug/replug or reflash the lower controller.
- Expected behavior is warning logs and reconnect attempts, not process termination.

## 2026-05-25 run_camera.sh watchdog

Background:

- `run_camera.sh` started `camera_detect_test` once and then waited for Enter after the program exited.
- For field/debug use, a process-level watchdog is useful so transient crashes or exits restart the camera detection program automatically.
- User clarified that the real boot script starts `./build/standard_mpc configs/standard3.yaml`.

Source change:

- Reworked `run_camera.sh` into a watchdog loop.
- It sets OpenVINO and camera SDK library paths, enters `/home/auto/burn-your-bridges`, creates `logs/`, `patterns/`, and `imgs/`, then starts `./build/standard_mpc configs/standard3.yaml`.
- Program output and watchdog events are appended to `logs/standard_mpc_watchdog.log`.
- If the program exits, the script waits 2 seconds and restarts it.
- `Ctrl+C`/`SIGTERM` stops the child process and exits the watchdog.

Operational note:

- If `standard_mpc` exits, the watchdog will restart it. Use `Ctrl+C` in the terminal to stop the watchdog itself.
- `set -u` is enabled only after sourcing OpenVINO `setupvars.sh`, because that script may reference unset internal variables such as `python_version`.

## 2026-05-25 Hero hik_camera_b calibration

Background:

- A new hero machine camera calibration was provided for `hik_camera_b` at 1280x1024.
- The provided extrinsic was ROS-style `odom2camera_b`:
  - `xyz`: `[0.05644, -0.0605, 0.06623]` m.
  - `rpy`: `[roll=0.0, pitch=0.35500077, yaw=-0.00]` rad.

Source change:

- Updated `configs/hero.yaml` camera intrinsics:
  - `camera_matrix`: `[7218.64522, 0, 561.76119, 0, 7207.20144, 510.45250, 0, 0, 1]`
  - `distort_coeffs`: `[-0.044582, 3.874065, 0.000465, -0.000231, 0.000000]`
- Converted `odom2camera_b` into this project's `R_camera2gimbal` convention using the same camera-to-gimbal basis as the previous standard3/hero extrinsic conversion.
- Updated `t_camera2gimbal` to `[0.05644, -0.06050, 0.06623]`.

Notes:

- `camera_name: "hikrobot"` was left unchanged because this project uses it as the camera driver selector, not as the ROS calibration camera name.
- `projection_matrix` and `rectification_matrix` from the ROS calibration are not read by this project.

## 2026-05-28 YOLO11 armor pose model switch

Background:

- A new Ultralytics YOLO11 pose OpenVINO model was exported under
  `C:/Users/19566/Desktop/RM/net/armor/rm_armor_export_20260528(1)`.
- `metadata.yaml` shows:
  - `task: pose`
  - 28 classes:
    red, blue, gray, purple groups, each containing hero/engineer/3/4/sentry/outpost/base.
  - `kpt_shape: [4, 3]`, meaning each armor has four keypoints and each keypoint is `x, y, confidence`.
- The previous YOLO11 postprocess path assumed the old class order and did not fully handle keypoint confidence.

Source/config change:

- Updated `armor_properties` order to match the new model metadata:
  - class ids 0-6: red
  - class ids 7-13: blue
  - class ids 14-20: extinguish/gray
  - class ids 21-27: purple
- Changed `YOLO11::class_num_` to use `armor_properties.size()` so class count follows the mapping table.
- Updated YOLO11 postprocess to parse `4 + class_num + 4 * 3` output columns:
  - `xywh`
  - class scores
  - four keypoints, each as `x, y, confidence`
- Added a keypoint confidence filter in YOLO11 postprocess.
- Updated `configs/standard3.yaml`:
  - `yolo_name: yolo11`
  - `yolo11_model_path: 'C:/Users/19566/Desktop/RM/net/armor/rm_armor_export_20260528(1)/best.xml'`

Notes:

- The OpenVINO XML output shape was checked as `[1, 44, 8400]`, matching `4 + 28 + 12`.
- This path is Windows-local; Linux/NUC deployment needs a Linux-visible model path.

## 2026-05-31 Gimbal serial stream parser

Background:

- The previous gimbal receive path read the fixed packet in two blocking pieces: header first, then the rest of the packet.
- When USB CDC split packets, dropped bytes, or left the stream misaligned, each failed body read could wait for the read timeout and reduce the effective receive frame rate.
- A previous mutex-based reconnect guard was not reintroduced because it can serialize read/write and reduce command send rate.

Source change:

- Reworked `Gimbal::read_thread()` to parse the serial port as a byte stream:
  - Read only currently available bytes into a small chunk buffer.
  - Append bytes to an RX buffer.
  - Search for the `SP` header.
  - Keep incomplete packets for the next read.
  - On CRC success, apply the packet immediately and continue parsing any queued packets.
  - On CRC failure, drop one byte and resync instead of waiting for another fixed-size read.
- Changed serial timeout from `simpleTimeout(20)` to separate read/write timeouts:
  - read timeout: `2 ms`
  - write timeout: `20 ms`
- Kept the existing `send()` path and planner send cadence unchanged.
- Clear stale input bytes after reconnect with `flushInput()`.
- Removed the old private fixed-size `read()` helper so future receive code does not accidentally return to the blocking two-step packet read.

Expected effect:

- Bad or partial frames should be discarded without slowing normal receive rate.
- If several lower-board packets arrive before the read thread runs, the parser can consume multiple packets from the buffer in one loop.
- Send-to-gimbal rate should not drop from this change because no new `Gimbal`-level read/write mutex was added and write timeout was preserved.

## 2026-06-01 standard_mpc fixed-rate auto-aim send

Background:

- `standard_mpc` previously sent AUTO_AIM control packets with `sleep_for(10ms)` after each `planner.plan()` and serial write.
- The real send period was therefore `planner.plan + serial.write + 10ms`, so the effective rate could be below 100 Hz and had compute-time jitter.
- The desired trial rate is about 150 Hz while keeping the camera / YOLO loop independent.

Source change:

- Changed the AUTO_AIM plan thread to a fixed-period schedule:
  - `plan_send_period = 6667 us`, approximately 150 Hz.
  - The thread uses `sleep_until(next_send)` so `planner.plan()` and `gimbal.send()` run inside the period budget.
  - If the thread falls more than one period behind, it resets the schedule to avoid chasing stale send times.
- Non-AUTO_AIM plan-thread sleep was reduced from `200ms` to `20ms` so entering AUTO_AIM does not wait up to 200 ms before the first control packet.
- The serial `send()` path, planner implementation, and camera / YOLO loop were not changed.

Expected effect:

- AUTO_AIM command output should be close to 150 Hz when `planner.plan()` and serial write stay under the period budget.
- Send timing jitter should be lower than the previous `sleep_for(10ms)` loop.
