#include "rknpu2_ros_yolov5/rknpu2_yolov5.hpp"
#include "rknpu2_ros_yolov5/postprocess.hpp"


namespace rknpu2_ros
{
    RKNPU2_YoloV5::RKNPU2_YoloV5(std::string model_path, float nms_th, float conf_th, int num_classes)
        : model_path_(model_path), nms_th_(nms_th), conf_th_(conf_th), num_classes_(num_classes)
    {
        int ret = 0;
        memset(&this->src_rect_, 0, sizeof(this->src_rect_));
        memset(&this->dst_rect_, 0, sizeof(this->dst_rect_));
        memset(&this->src_, 0, sizeof(this->src_));
        memset(&this->dst_, 0, sizeof(this->dst_));
        
        this->model_data_ = load_model(model_path_.c_str(), &this->model_data_size_);
        ret = rknn_init(&this->ctx_, this->model_data_, this->model_data_size_, 0, NULL);
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            exit -1;
        }

        rknn_sdk_version version;
        ret = rknn_query(this->ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            exit -1;
        }
        printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

        // input/output num
        ret = rknn_query(this->ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            exit -1;
        }
        printf("model input num: %d, output num: %d\n", io_num_.n_input, io_num_.n_output);

        rknn_tensor_attr input_attrs[io_num_.n_input];
        memset(input_attrs, 0, sizeof(input_attrs));
        for (int i = 0; i < io_num_.n_input; i++) {
            input_attrs[i].index = i;
            ret = rknn_query(this->ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret < 0) {
                printf("rknn_init error ret=%d\n", ret);
                exit -1;
            }
        }

        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            printf("model is NCHW input fmt\n");
            input_c_ = input_attrs[0].dims[1];
            input_h_ = input_attrs[0].dims[2];
            input_w_ = input_attrs[0].dims[3];
        } else {
            printf("model is NHWC input fmt\n");
            input_h_ = input_attrs[0].dims[1];
            input_w_ = input_attrs[0].dims[2];
            input_c_ = input_attrs[0].dims[3];
        }

        printf("model input height=%d, width=%d, channel=%d\n", input_h_, input_w_, input_c_);

    }

    RKNPU2_YoloV5::~RKNPU2_YoloV5()
    {
        rknn_destroy(this->ctx_);
        free(this->model_data_);
    }

    std::vector<Object> RKNPU2_YoloV5::inference(const cv::Mat &bgr)
    {
        cv::Mat img;
        int ret = 0;

        int img_width  = bgr.cols;
        int img_height = bgr.rows;

        // preprocess
        cv::cvtColor(bgr, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(input_w_, input_h_));

        // set input
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index        = 0;
        inputs[0].type         = RKNN_TENSOR_UINT8;
        inputs[0].size         = input_w_ * input_h_ * input_c_;
        inputs[0].fmt          = RKNN_TENSOR_NHWC;
        inputs[0].pass_through = 0;
        inputs[0].buf = (void*)img.data;

        rknn_inputs_set(this->ctx_, this->io_num_.n_input, inputs);

        rknn_output outputs[this->io_num_.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < this->io_num_.n_output; i++) {
            outputs[i].want_float = 0;
        }

        rknn_tensor_attr output_attrs[this->io_num_.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < io_num_.n_output; i++) {
            output_attrs[i].index = i;
            ret = rknn_query(this->ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        }

        // inference
        ret = rknn_run(this->ctx_, NULL);

        // get output
        ret = rknn_outputs_get(this->ctx_, this->io_num_.n_output, outputs, NULL);

        // post process
        float scale_w = (float)input_w_ / img_width;
        float scale_h = (float)input_h_ / img_height;

        std::vector<Object> results;
        std::vector<float>    out_scales;
        std::vector<int32_t>  out_zps;
        for (int i = 0; i < this->io_num_.n_output; ++i) {
            out_scales.push_back(output_attrs[i].scale);
            out_zps.push_back(output_attrs[i].zp);
        }
        postprocess::execute((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, input_h_, input_w_,
                             this->conf_th_, this->nms_th_, scale_w, scale_h, out_zps, out_scales, results, this->num_classes_);
        rknn_outputs_release(this->ctx_, this->io_num_.n_output, outputs);
        return results;
    }
    
}
