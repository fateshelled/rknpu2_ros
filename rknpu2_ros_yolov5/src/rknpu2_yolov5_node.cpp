#include "rknpu2_ros_yolov5/rknpu2_yolov5_node.hpp"

namespace rknpu2_ros
{
    RKNPU2_YoloV5_Node::RKNPU2_YoloV5_Node(const rclcpp::NodeOptions &options)
        : RKNPU2_YoloV5_Node::RKNPU2_YoloV5_Node("", options)
    {}

    RKNPU2_YoloV5_Node::RKNPU2_YoloV5_Node(const std::string &node_name, const rclcpp::NodeOptions &options)
        : rclcpp::Node("rknpu2_yolov5_node", node_name, options)
    {        
        this->initializeParameter();

        this->model_ = std::make_unique<RKNPU2_YoloV5>(this->model_path_, this->nms_th_, this->conf_th_, this->num_classes_);
        RCLCPP_INFO(this->get_logger(), "model loaded");

        this->sub_image_ = image_transport::create_subscription(
            this, this->src_image_topic_name_,
            std::bind(&RKNPU2_YoloV5_Node::image_callback, this, std::placeholders::_1),
            "raw");
        this->pub_bboxes_ = this->create_publisher<bboxes_ex_msgs::msg::BoundingBoxes>(
            this->publish_boundingbox_topic_name_,
            10
        );
        this->pub_image_ = image_transport::create_publisher(this, this->publish_image_topic_name_);
    }

    RKNPU2_YoloV5_Node::~RKNPU2_YoloV5_Node()
    {}

    void RKNPU2_YoloV5_Node::initializeParameter()
    {
        this->model_path_  = this->declare_parameter<std::string>("model_path", "install/rknpu2_ros_yolov5/share/rknpu2_ros_yolov5/model/RK3588/yolov5s-640-640.rknn");
        this->label_path_  = this->declare_parameter<std::string>("label_path", "");
        this->nms_th_      = this->declare_parameter<float>("nms_th", 0.45);
        this->conf_th_     = this->declare_parameter<float>("conf_th", 0.30);
        this->num_classes_ = this->declare_parameter<int>("num_classes", 80);
        this->imshow_      = this->declare_parameter<bool>("imshow", true);
        this->src_image_topic_name_             = this->declare_parameter<std::string>("src_image_topic_name", "image_raw");
        this->publish_image_topic_name_         = this->declare_parameter<std::string>("publish_image_topic_name", "yolov5/image_raw");
        this->publish_boundingbox_topic_name_   = this->declare_parameter<std::string>("publish_boundingbox_topic_name", "yolov5/bounding_boxes");

        RCLCPP_INFO(this->get_logger(), "Set parameter model_path: '%s'", this->model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter label_path: '%s'", this->label_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter conf_th: %f", this->conf_th_);
        RCLCPP_INFO(this->get_logger(), "Set parameter nms_th: %f", this->nms_th_);
        RCLCPP_INFO(this->get_logger(), "Set parameter num_classes: %i", this->num_classes_);
        RCLCPP_INFO(this->get_logger(), "Set parameter imshow: %i", this->imshow_);
        RCLCPP_INFO(this->get_logger(), "Set parameter src_image_topic_name: '%s'", this->src_image_topic_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter publish_image_topic_name: '%s'", this->publish_image_topic_name_.c_str());
        RCLCPP_INFO(this->get_logger(), "Set parameter publish_boundingbox_topic_name: '%s'", this->publish_boundingbox_topic_name_.c_str());
    }
    
    void RKNPU2_YoloV5_Node::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& ptr)
    {
        // auto img = cv_bridge::toCvShare(ptr, "bgr8");
        auto img = cv_bridge::toCvCopy(ptr, "bgr8");
        cv::Mat frame = img->image;
        
        // fps
        auto now = std::chrono::system_clock::now();

        auto objects = this->model_->inference(frame);

        auto end = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - now);
        RCLCPP_INFO(this->get_logger(), "Inference: %f FPS", 1000.0f / elapsed.count());

        // draw
        cv::Mat drawn;
        frame.copyTo(drawn);
        utils::draw_objects(drawn, objects, this->labels_);

        // imshow
        if(this->imshow_){
            cv::imshow(this->WINDOW_NAME_, drawn);
            auto key = cv::waitKey(1);
            if(key == 27){
                rclcpp::shutdown();
            }
        }

        // pub bbox
        auto bboxes = utils::objects_to_bboxes(frame, ptr->header, objects, this->labels_);
        this->pub_bboxes_->publish(bboxes);

        // pub img
        sensor_msgs::msg::Image::SharedPtr pub_img;
        pub_img = cv_bridge::CvImage(img->header, "bgr8", drawn).toImageMsg();
        this->pub_image_.publish(pub_img);
    }

}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    rclcpp::spin(std::make_shared<rknpu2_ros::RKNPU2_YoloV5_Node>(node_options));
    rclcpp::shutdown();
    return 0;
}

// #include <rclcpp_components/register_node_macro.hpp>
// RCLCPP_COMPONENTS_REGISTER_NODE(rknpu2_ros::RKNPU2_YoloV5_Node)