cmake_minimum_required(VERSION 3.16)

add_subdirectory(planner/tinympc)

set(AUTO_AIM_SOURCES
    armor.cpp
    armor_post_filter.cpp
    classifier.cpp 
    detector.cpp
    fire_gate.cpp
    runtime.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    multithread/commandgener.cpp
    planner/planner.cpp
)
if(OpenVINO_FOUND)
    list(APPEND AUTO_AIM_SOURCES yolos/yolov8.cpp yolos/yolo11.cpp multithread/mt_detector.cpp)
endif()

add_library(auto_aim OBJECT ${AUTO_AIM_SOURCES})

target_link_libraries(auto_aim io tinympcstatic)
if(OpenVINO_FOUND)
    target_link_libraries(auto_aim openvino::runtime)
    target_compile_definitions(auto_aim PRIVATE SP_VISION_WITH_OPENVINO=1)
endif()

if(SP_VISION_ENABLE_ONNXRUNTIME)
    target_include_directories(auto_aim PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(auto_aim ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(auto_aim PRIVATE SP_VISION_WITH_ONNXRUNTIME=1)
endif()
if(TARGET onnxruntime_cuda)
  target_link_libraries(auto_aim onnxruntime_cuda)
endif()
