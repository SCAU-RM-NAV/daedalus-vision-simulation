cmake_minimum_required(VERSION 3.8)
project(sp_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(sp_msgs
  "msg/EnemyStatusMsg.msg"
  "msg/AutoaimTargetMsg.msg"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
