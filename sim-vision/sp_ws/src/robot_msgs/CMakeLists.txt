cmake_minimum_required(VERSION 3.8)
project(robot_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DecisionCtrl.msg"
  "msg/EnemyInfo.msg"
  "msg/GameState.msg"
  "msg/GimbalNav.msg"
  "msg/DetectFlag.msg"
  "msg/ShootCmd.msg"
  "msg/GimbalFdb.msg"
  "msg/RemoteHeroFdb.msg"
  "msg/LocationInfo.msg"
  "msg/Feeling.msg"
  "msg/RobotStatus.msg"
  "msg/NumberCtrl.msg"
  "msg/Diff.msg"
  "msg/EnemyDir.msg"
  "msg/VisionCtrl.msg"
  "msg/VisualStatus.msg"
  "msg/IsSub.msg"
  "msg/EnemyLocationID.msg"
  "msg/Tracker.msg"
  "msg/VisionStatus.msg"
  "msg/SpeedCtrl.msg"
  "msg/Pursue.msg"
  "msg/DecisionState.msg"
  "msg/CameraDetectInfo.msg"
  "msg/InRegion.msg"
  "msg/CmdVelStr.msg"
  "srv/PidPlannerStatus.srv"
  "srv/Relocate.srv"
  "srv/Decision.srv"
  "msg/AreaDetectInfo.msg"
  "msg/ArmorSenseMap.msg"
  "msg/Bool.msg"
  "msg/CartTreeInput.msg"
  "msg/VisionSpSend.msg"
  "msg/VisionCtrlNew.msg"
  DEPENDENCIES
    std_msgs
    geometry_msgs
)


ament_package()
