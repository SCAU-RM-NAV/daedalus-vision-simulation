cmake_minimum_required(VERSION 3.16.3)

project(sp_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

list(PREPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/sp_ws/install")
unset(robot_msgs_DIR CACHE)
unset(sp_msgs_DIR CACHE)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
if(NOT OpenVINO_DIR OR NOT EXISTS "${OpenVINO_DIR}/OpenVINOConfig.cmake")
    file(GLOB _openvino_configs
      "/opt/intel/openvino_*/runtime/cmake/OpenVINOConfig.cmake"
      "/opt/intel/l_openvino_*/runtime/cmake/OpenVINOConfig.cmake")
    list(SORT _openvino_configs COMPARE NATURAL ORDER DESCENDING)
    if(_openvino_configs)
        list(GET _openvino_configs 0 _openvino_config)
        get_filename_component(OpenVINO_DIR "${_openvino_config}" DIRECTORY)
    endif()
endif()
find_package(OpenVINO QUIET)
if(OpenVINO_FOUND)
    message(STATUS "OpenVINO found: enabling Intel inference paths.")
else()
    message(WARNING "OpenVINO not found: only the OpenCV DNN CUDA YOLOv5 path is available.")
endif()

option(SP_VISION_ENABLE_ONNXRUNTIME "Build the generic ONNX Runtime inference backend" OFF)
option(ENABLE_ONNXRUNTIME_CUDA "Build the simulation ONNX Runtime CUDA backend" ON)
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime CUDA installation directory")
if((SP_VISION_ENABLE_ONNXRUNTIME OR ENABLE_ONNXRUNTIME_CUDA) AND ONNXRUNTIME_ROOT)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    add_library(onnxruntime_cuda STATIC tools/onnxruntime_cuda.cpp)
    target_include_directories(onnxruntime_cuda PUBLIC "${PROJECT_SOURCE_DIR}" "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(onnxruntime_cuda PUBLIC "${ONNXRUNTIME_LIBRARY}" ${OpenCV_LIBS} dl)
    target_compile_definitions(onnxruntime_cuda
      INTERFACE SP_VISION_HAS_ONNXRUNTIME_CUDA=1
      PRIVATE SP_VISION_ONNXRUNTIME_CUDA_RUNTIME_ROOT="${ONNXRUNTIME_ROOT}/python")
    message(STATUS "ONNX Runtime CUDA backend enabled from ${ONNXRUNTIME_ROOT}")
  else()
    message(WARNING "ONNXRUNTIME_ROOT does not contain an ONNX Runtime C++ SDK; CUDA ONNX backend disabled")
  endif()
endif()
if(SP_VISION_ENABLE_ONNXRUNTIME AND (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY))
  message(FATAL_ERROR "SP_VISION_ENABLE_ONNXRUNTIME=ON requires a valid ONNXRUNTIME_ROOT")
endif()

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})


add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)
add_subdirectory(tasks/omniperception)
add_subdirectory(tasks/hero_deploy_vt EXCLUDE_FROM_ALL)

add_library(sim_debug STATIC sim/debug_visualizer.cpp)
target_link_libraries(sim_debug PUBLIC ${OpenCV_LIBS} fmt::fmt auto_aim auto_buff tools)

####################src###################
add_executable(standard_mpc src/standard_mpc.cpp)
add_executable(auto_aim_debug_mpc src/auto_aim_debug_mpc.cpp)
add_executable(auto_buff_debug src/auto_buff_debug.cpp)
add_executable(auto_buff_debug_mpc src/auto_buff_debug_mpc.cpp)
add_executable(uav src/uav.cpp)
add_executable(uav_debug src/uav_debug.cpp)
add_executable(remote_hero_debug src/remote_hero_debug.cpp)
add_executable(vision_sim_runner src/vision_sim_runner.cpp)
add_executable(vision_sim_runner_debug src/vision_sim_runner.cpp)
target_compile_definitions(vision_sim_runner_debug PRIVATE VISION_SIM_DEBUG_BUILD=1)
add_executable(sim_model_probe tools/sim_model_probe.cpp)
if(TARGET onnxruntime_cuda)
  add_executable(onnxruntime_cuda_smoke tests/onnxruntime_cuda_smoke.cpp)
  target_link_libraries(onnxruntime_cuda_smoke onnxruntime_cuda ${OpenCV_LIBS})
endif()

target_link_libraries(standard_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_aim_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(auto_buff_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(auto_buff_debug_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io ${CERES_LIBRARIES})
target_link_libraries(uav ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)
target_link_libraries(uav_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

target_link_libraries(remote_hero_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)
target_link_libraries(vision_sim_runner ${OpenCV_LIBS} fmt::fmt yaml-cpp nlohmann_json::nlohmann_json auto_aim auto_buff sim_io sim_debug tools)
target_link_libraries(vision_sim_runner_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp nlohmann_json::nlohmann_json auto_aim auto_buff sim_io sim_debug tools)
target_link_libraries(sim_model_probe nlohmann_json::nlohmann_json openvino::runtime)

##################calibration################
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
add_executable(calibrate_robotworld_handeye calibration/calibrate_robotworld_handeye.cpp)
add_executable(split_video calibration/split_video.cpp)

target_link_libraries(capture ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(calibrate_camera ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_robotworld_handeye ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(split_video ${OpenCV_LIBS} fmt::fmt tools)

##################tests##################
add_executable(auto_aim_test tests/auto_aim_test.cpp)
add_executable(auto_aim_srt tools/auto_aim_srt.cpp)
add_executable(auto_buff_test tests/auto_buff_test.cpp)
add_executable(camera_detect_test tests/camera_detect_test.cpp)
add_executable(camera_test tests/camera_test.cpp)
add_executable(camera_thread_test tests/camera_thread_test.cpp)
add_executable(cboard_test tests/cboard_test.cpp)
add_executable(fire_test tests/fire_test.cpp)
add_executable(detector_video_test tests/detector_video_test.cpp)
add_executable(gimbal_response_test tests/gimbal_response_test.cpp)
add_executable(assembly_error_calculator tests/assembly_error_calculator.cpp)
add_executable(multi_usbcamera_test tests/multi_usbcamera_test.cpp)
add_executable(usbcamera_detect_test tests/usbcamera_detect_test.cpp)
add_executable(usbcamera_test tests/usbcamera_test.cpp)
add_executable(handeye_test tests/handeye_test.cpp)
add_executable(dm_test tests/dm_test.cpp)
add_executable(minimum_vision_system tests/minimum_vision_system.cpp)

target_link_libraries(auto_aim_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(auto_aim_srt ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim auto_buff)
target_link_libraries(auto_buff_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_buff tools io)
target_link_libraries(camera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim io)
target_link_libraries(camera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(camera_thread_test ${OpenCV_LIBS} fmt::fmt auto_aim tools io)
target_link_libraries(cboard_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(fire_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(detector_video_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools auto_aim)
target_link_libraries(gimbal_response_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(assembly_error_calculator ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(multi_usbcamera_test ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(usbcamera_detect_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(usbcamera_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(handeye_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)
target_link_libraries(dm_test ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)
target_link_libraries(minimum_vision_system ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io auto_aim)

add_executable(gimbal_test tests/gimbal_test.cpp)
target_link_libraries(gimbal_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test tests/planner_test.cpp)
target_link_libraries(planner_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(planner_test_offline tests/planner_test_offline.cpp)
target_link_libraries(planner_test_offline ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(fire_gate_test tests/fire_gate_test.cpp)
target_link_libraries(fire_gate_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(target_observation_only_test tests/target_observation_only_test.cpp)
target_link_libraries(target_observation_only_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(talos_ipc_test tests/talos_ipc_test.cpp)
target_link_libraries(talos_ipc_test ${OpenCV_LIBS} fmt::fmt sim_io tools)

add_executable(talos_gimbal_probe tests/talos_gimbal_probe.cpp)
target_link_libraries(talos_gimbal_probe ${OpenCV_LIBS} fmt::fmt sim_io)

add_executable(solver_runtime_calibration_test tests/solver_runtime_calibration_test.cpp)
target_link_libraries(solver_runtime_calibration_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff tools io)

add_executable(runtime_pose_source_test tests/runtime_pose_source_test.cpp)
target_link_libraries(runtime_pose_source_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(talos_ipc_fixture tests/talos_ipc_fixture.cpp)
target_link_libraries(talos_ipc_fixture ${OpenCV_LIBS} fmt::fmt sim_io)

add_executable(sim_debug_visualizer_test tests/sim_debug_visualizer_test.cpp)
target_link_libraries(sim_debug_visualizer_test ${OpenCV_LIBS} fmt::fmt sim_debug)

add_executable(runtime_debug_result_test tests/runtime_debug_result_test.cpp)
target_link_libraries(runtime_debug_result_test ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

# 检查 ROS 环境
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)
find_package(robot_msgs QUIET)
# 只有在找到 ROS 环境时才会添加 sentry 和 publish_test
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND AND robot_msgs_FOUND)
    if(TARGET deploy_vt)
        add_executable(remote_hero src/remote_hero.cpp)
        target_link_libraries(remote_hero ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim deploy_vt tools io)
        ament_target_dependencies(remote_hero rclcpp robot_msgs)
    endif()

    # 编译 sentry 和 publish_test
    add_executable(sentry_mpc src/sentry_mpc.cpp)
    add_executable(sentry_mpc_debug src/sentry_mpc_debug.cpp)
    add_executable(publish_test tests/publish_test.cpp)
    add_executable(subscribe_test tests/subscribe_test.cpp)
    add_executable(topic_loop_test tests/topic_loop_test.cpp)

    target_link_libraries(sentry_mpc ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff omniperception tools io)
    target_link_libraries(sentry_mpc_debug ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim auto_buff omniperception tools io)
    target_link_libraries(publish_test ${OpenCV_LIBS} fmt::fmt tools io auto_aim)
    target_link_libraries(subscribe_test ${OpenCV_LIBS} fmt::fmt tools io)
    target_link_libraries(topic_loop_test ${OpenCV_LIBS} fmt::fmt tools io)
else()
    message(STATUS "ROS2 environment not found, skipping ROS2-related code.")
endif()
