cmake_minimum_required(VERSION 3.8)
project(hero_deploy_vt)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TARGET_EXPORT_NAME HeroDeployVt)

set(LIB_NAME deploy_vt)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GST QUIET IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_check_modules(GST_APP QUIET IMPORTED_TARGET gstreamer-app-1.0>=1.20)

if(NOT GST_FOUND OR NOT GST_APP_FOUND)
  message(WARNING "Required GStreamer libraries not found, hero_deploy_vt will not be built.")
endif()

if(GST_VERSION VERSION_LESS "1.22")
  message(STATUS "GStreamer version ${GST_VERSION} < 1.22, using vaapih265enc")
  set(VT_USE_OLD_VAENC 1)
else()
  message(STATUS "GStreamer version ${GST_VERSION} >= 1.22, using vah265enc")
  set(VT_USE_OLD_VAENC 0)
endif()

configure_file(
  conf.hpp.in 
  "${CMAKE_CURRENT_SOURCE_DIR}/include/hero_deploy_vt/conf.hpp"
  @ONLY 
  NEWLINE_STYLE UNIX
)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*.cpp")

if(GST_FOUND AND GST_APP_FOUND)
  # ------------------------------------------------------
  #                 libdeploy_vt (HeroDeployVt)
  # ------------------------------------------------------
  add_library(${LIB_NAME} STATIC
    ${SRC_FILES}  
  )
  target_include_directories(${LIB_NAME} PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(${LIB_NAME} 
    PRIVATE
      PkgConfig::GST 
      PkgConfig::GST_APP 
      tools
      fmt::fmt
      yaml-cpp
      io
      ${OpenCV_LIBS}
    PUBLIC
      ${OpenCV_LIBS}
  )
  target_compile_features(${LIB_NAME} PRIVATE cxx_std_20)

  install(DIRECTORY include/ 
    DESTINATION include 
  )

  install(TARGETS ${LIB_NAME}
    EXPORT ${TARGET_EXPORT_NAME}
    ARCHIVE DESTINATION lib 
    LIBRARY DESTINATION lib 
    RUNTIME DESTINATION bin 
  )
  # ------------------------------------------------------

  # ------------------------------------------------------
  #                  hero_vt_test exe
  # ------------------------------------------------------
  add_executable(hero_vt_test 
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/encode_pull_and_push.cpp
  )
  target_include_directories(hero_vt_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(hero_vt_test PRIVATE 
    ${LIB_NAME}
    PkgConfig::GST 
    PkgConfig::GST_APP 
    ${OpenCV_LIBS}
  )
  target_compile_features(hero_vt_test PRIVATE cxx_std_20)

  install(TARGETS hero_vt_test
    DESTINATION lib/${PROJECT_NAME}
    LIBRARY DESTINATION lib/${PROJECT_NAME}
  )
  # ------------------------------------------------------
endif()
