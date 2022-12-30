#ifndef RKNPU2_ROS_YOLOV5_HPP_
#define RKNPU2_ROS_YOLOV5_HPP_

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv4/opencv2/opencv.hpp"
#include "rga.h"
#include "rknn_api.h"

#include "rknpu2_ros_common/object.hpp"
#include "rknpu2_ros_common/model_loader.h"

namespace rknpu2_ros{

class RKNPU2_YoloV5{
public:
    RKNPU2_YoloV5(std::string model_path,
                  float nms_th=0.45, float conf_th=0.3,
                  int num_classes=80);
    ~RKNPU2_YoloV5();

    std::vector<Object> inference(const cv::Mat& bgr);
private:
    std::string model_path_;
    float nms_th_;
    float conf_th_;
    int num_classes_;

    const int anchor0[6] = {10, 13, 16, 30, 33, 23};
    const int anchor1[6] = {30, 61, 62, 45, 59, 119};
    const int anchor2[6] = {116, 90, 156, 198, 373, 326};

    rknn_context ctx_;
    int model_data_size_ = 0;
    unsigned char* model_data_;
    rga_buffer_t src_;
    rga_buffer_t dst_;
    im_rect src_rect_;
    im_rect dst_rect_;

    rknn_input_output_num io_num_;
    rknn_tensor_attr* output_attrs_;
    int input_c_;
    int input_w_;
    int input_h_;
};

} // namespace rknpu2_ros

#endif // RKNPU2_ROS_YOLOV5_HPP_
