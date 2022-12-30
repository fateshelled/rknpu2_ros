cmake_minimum_required(VERSION 3.5)
project(rknpu2_ros_common)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# # CMAKE_SYSTEM_NAME: Linux
# # CMAKE_SYSTEM_PROCESSOR: aarch64 or armhf
# # TARGET_SOC: RK3588 or RK356X or RV110X
# set(RGA_PATH ${CMAKE_SOURCE_DIR}/../rknpu2/examples/3rdparty/rga/${TARGET_SOC})
# set(RGA_INCLUDE ${RGA_PATH}/include)
# set(RGA_LIB ${RGA_PATH}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/librga.so)

# set(RKNN_RT_PATH ${CMAKE_SOURCE_DIR}/../rknpu2/runtime/${TARGET_SOC}/${CMAKE_SYSTEM_NAME}/librknn_api)
# set(RKNN_RT_LIB ${RKNN_RT_PATH}/${CMAKE_SYSTEM_PROCESSOR}/librknnrt.so)
# set(RKNN_RT_INCLUDE ${RKNN_RT_PATH}/include)

# library
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
# target_link_libraries(${PROJECT_NAME} INTERFACE
#   ${RKNNRT_LIB}
#   ${RGA_LIB}
# )

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ament_export_dependencies(
# )
ament_export_targets(export_${PROJECT_NAME})

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
