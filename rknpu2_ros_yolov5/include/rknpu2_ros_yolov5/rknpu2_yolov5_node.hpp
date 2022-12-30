#ifndef RKNPU2_ROS_YOLOV5_NODE_HPP_
#define RKNPU2_ROS_YOLOV5_NODE_HPP_

#include <math.h>
#include <chrono>

#include <rclcpp/rclcpp.hpp>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

#include "bboxes_ex_msgs/msg/bounding_box.hpp"
#include "bboxes_ex_msgs/msg/bounding_boxes.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "rknpu2_ros_yolov5/rknpu2_yolov5.hpp"

#include "rknpu2_ros_common/coco_names.hpp"
#include "rknpu2_ros_common/utils.hpp"

namespace rknpu2_ros
{

class RKNPU2_YoloV5_Node : public rclcpp::Node
{
public:
    RKNPU2_YoloV5_Node(const rclcpp::NodeOptions& options);
    RKNPU2_YoloV5_Node(const std::string &node_name, const rclcpp::NodeOptions& options);
    ~RKNPU2_YoloV5_Node();

private:
    std::shared_ptr<RKNPU2_YoloV5> model_;
    void initializeParameter();
    void loadLabel(const std::string label_filename);
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& ptr);

    image_transport::Subscriber sub_image_;
    rclcpp::Publisher<bboxes_ex_msgs::msg::BoundingBoxes>::SharedPtr pub_bboxes_;
    image_transport::Publisher pub_image_;

    std::vector<std::string> labels_ = COCO_CLASSES;

    std::string model_path_;
    float nms_th_;
    float conf_th_;
    int num_classes_;
    std::string label_path_;
    bool imshow_;
    std::string src_image_topic_name_;
    std::string publish_image_topic_name_;
    std::string publish_boundingbox_topic_name_;

    const std::string WINDOW_NAME_ = "RKNPU2_ROS_YOLOV5";

};

} // namespace rknpu2_ros

#endif // RKNPU2_ROS_YOLOV5_NODE_HPP_