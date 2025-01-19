#ifndef TRT_INFERENCE_HPP
#define TRT_INFERENCE_HPP
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <stdlib.h>
#include <numeric>
#include <iterator>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/quality.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;
#include "TRTGeneration.hpp"
#include "ImageProcessingUtil.hpp" 


class TRTInference {
public:
    // Measures the performance of TRT inference on super-resolution models
    static void measure_trt_performance(const std::string& trt_plan, const std::string& original_image_path,
                                        torch::Tensor img_tensor, int num_trials, bool compare_img_bool);

    // Measures the performance of TRT inference on segmentation models
    static void measure_segmentation_trt_performance(const std::string& trt_plan, torch::Tensor img_tensor, int num_trials);
   
    // Modified measure_segmentation_trt_performance_mul to return a vector of grayscale images
    static std::vector<cv::Mat> measure_segmentation_trt_performance_mul(const std::string& trt_plan, torch::Tensor img_tensor, int num_trials);
};

#endif
