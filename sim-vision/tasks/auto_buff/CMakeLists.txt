cmake_minimum_required(VERSION 3.16)

if(NOT OpenVINO_FOUND)
    # Kept as a target so auto-aim-only tools can configure on Orin. Buff and
    # sentry executables are intentionally not supported until this backend is migrated.
    add_library(auto_buff INTERFACE)
    return()
endif()

find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    angle_estimate.cpp
    buff_ballistic.cpp
    buff_pose_filter.cpp
    buff_type.cpp
    buff_detector.cpp
    buff_solver.cpp
    buff_aimer.cpp
    yolo11_buff.cpp
)

target_link_libraries(auto_buff auto_aim openvino::runtime ${CERES_LIBRARIES})
if(TARGET onnxruntime_cuda)
  target_link_libraries(auto_buff onnxruntime_cuda)
endif()
