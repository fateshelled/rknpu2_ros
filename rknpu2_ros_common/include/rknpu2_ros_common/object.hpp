#pragma once 

#include <opencv4/opencv2/core/types.hpp>

namespace rknpu2_ros
{
    
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

}