
import os
import argparse
import ast

from rknn.api import RKNN

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--onnx_model_path', type=str, required=True, help='onnx model path')
    parser.add_argument('--platform', type=str, default="rk3588", help='platform name. Currently support rk3566 / rk3568 / rk3588 / rv1103 / rv1106. default="rk3588"')
    parser.add_argument('--output_dir', type=str, default="rknn_models", help='rknn model output directory. default="rknn_models"')
    parser.add_argument('--dataset_file', type=str, default="./dataset.txt", help='dataset file path. default="./dataset.txt"')
    args = parser.parse_args()

    PLATFORM = args.platform
    ONNX_MODEL_PATH = './yolox_s.onnx'
    EXP = os.path.splitext(os.path.basename(ONNX_MODEL_PATH))[0]
    OUT_DIR = args.output_dir
    DATASET = args.dataset_file
    RKNN_MODEL_PATH = './{}/{}.rknn'.format(OUT_DIR, EXP)
    # OPT_ONNX_MODEL_PATH = './{}/{}_opt.onnx'.format(OUT_DIR, EXP)
    MEAN_VALUES = [[0, 0, 0]]
    STD_VALUES  = [[255, 255, 255]]

    print()
    print('--- Convert onnx model to rknn model ---')

    # Create RKNN object
    rknn = RKNN(verbose=False)

    # :param mean_values: Channel mean value list.
    # :param std_values: Channel std value list.
    # :param quantized_dtype: quantize data type, currently support: asymmetric_quantized-8.
    # :param quantized_algorithm: currently support: normal, mmse (Min Mean Square Error), kl_divergence.
    # :param quantized_method: quantize method, currently support: layer, channel.
    # :param target_platform: target chip platform, default is None, means target platform is rk3566. Currently support rk3566 / rk3568 / rk3588 / rv1103 / rv1106.
    # :param quant_img_RGB2BGR: whether to do RGB2BGR when load quantize image (jpg/jpeg/png/bmp), default is False.
    # :param float_dtype: non quantize data type, currently support: float16, default is float16.
    # :param optimization_level: set optimization level, default 3 means use all default optimization options.
    # :param custom_string: add custom string information to rknn model, then can query the information at runtime.
    # :param remove_weight: generate a slave rknn model which removes conv2d weight, need share weight with rknn model of complete weights.
    # :param compress_weight: compress the weights of the model, which can reduce the size of rknn model.
    # :param inputs_yuv_fmt: add yuv preprocess at the top of model.
    # :param single_core_mode: only for rk3588. single_core_mode=True can reduce the size of rknn model.
    rknn.config(mean_values=MEAN_VALUES, std_values=STD_VALUES, target_platform=PLATFORM)

    # Load onnx model
    print()
    print('--- Loading ONNX model ---')
    ret = rknn.load_onnx(ONNX_MODEL_PATH)
    if ret == 0:
        print("Success")
    else:
        print('Load onnx model failed!')
        exit(ret)

    # # Optimize onnx model
    # # Only support onnx model that with 'quantization_annotation' information!
    # print()
    # print('--- Optimize ONNX model ---')
    # ret = rknn.optimize_onnx(ONNX_MODEL_PATH, OPT_ONNX_MODEL_PATH)
    # if ret == 0:
    #     print("Success")
    # else:
    #     print('Failed to optimize onnx model!')

    # Build model
    print()
    print('--- Building model ---')
    ret = rknn.build(do_quantization=True, dataset=DATASET)
    if ret == 0:
        print("Success")
    else:
        print('Failed to build rknn model.')
        exit(ret)

    # Export rknn model
    print()
    print('--- Export RKNN model: {} ---'.format(RKNN_MODEL_PATH))
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret == 0:
        print("Success")
    else:
        print('Failed to export rknn model.')
        exit(ret)

    # Import Test
    print()
    print('--- Import Test RKNN model: {} ---'.format(RKNN_MODEL_PATH))
    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret == 0:
        print("Success")
    else:
        print('Failed to import rknn model.')
        exit(ret)

    print('done')

    rknn.release()

