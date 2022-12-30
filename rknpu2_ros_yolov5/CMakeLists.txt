cmake_minimum_required(VERSION 3.5)
project(rknpu2_ros_yolov5)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(CMAKE_CC_COMPILER  aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(bboxes_ex_msgs REQUIRED)
find_package(rknpu2_ros_common REQUIRED)

# CMAKE_SYSTEM_NAME: Linux
# CMAKE_SYSTEM_PROCESSOR: aarch64 or armhf
# TARGET_SOC: RK3588 or RK356X or RV110X
set(RGA_PATH ${CMAKE_SOURCE_DIR}/../rknpu2/examples/3rdparty/rga/${TARGET_SOC})
set(RGA_INCLUDE ${RGA_PATH}/include)
set(RGA_LIB ${RGA_PATH}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librga.so)

set(RKNN_RT_PATH ${CMAKE_SOURCE_DIR}/../rknpu2/runtime/${TARGET_SOC}/${CMAKE_SYSTEM_NAME}/librknn_api)
set(RKNN_RT_LIB ${RKNN_RT_PATH}/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
set(RKNN_RT_INCLUDE ${RKNN_RT_PATH}/include)

# executable
add_executable(rknpu2_ros_yolov5
  src/rknpu2_yolov5.cpp
  src/rknpu2_yolov5_node.cpp
)

ament_target_dependencies(rknpu2_ros_yolov5
  rclcpp
  rclcpp_components
  cv_bridge
  image_transport
  builtin_interfaces
  std_msgs
  sensor_msgs
  OpenCV
  bboxes_ex_msgs
  rknpu2_ros_common
)

target_include_directories(rknpu2_ros_yolov5 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${RGA_INCLUDE}
  ${RKNN_RT_INCLUDE}
)

target_link_libraries(rknpu2_ros_yolov5
  ${RGA_LIB}
  ${RKNN_RT_LIB}
)

install(TARGETS
rknpu2_ros_yolov5
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/../rknpu2/examples/rknn_yolov5_demo/model
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
