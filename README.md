# rknpu2_ros
ROS2 Inference sample for using Rockchip NPU.

tested OrangePi5 (RK3588s) Ubuntu 22.04 + ROS2 Humble.

## build
```bash
# clone repository
cd ros2_ws/src
git clone --recursive https://github.com/fateshelled/rknpu2_ros
cd ../

# build
# TARGET_SOC = RK3588 or RK356X or RV110X
colcon build --symlink-install --packages-up-to rknpu2_ros --cmake-args -DTARGET_SOC=RK3588
```

## run
```bash
ros2 run v4l2_camera v4l2_camera_node

ros2 run rknpu2_ros_yolov5 rknpu2_ros_yolov5 
```
