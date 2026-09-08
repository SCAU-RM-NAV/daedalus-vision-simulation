cmake_minimum_required(VERSION 3.16)

if(NOT OpenVINO_FOUND)
    add_library(omniperception INTERFACE)
    return()
endif()

add_library(omniperception OBJECT 
    decider.cpp
    omni_status_decider.cpp
    perceptron.cpp
)
target_link_libraries(omniperception openvino::runtime)
