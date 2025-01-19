#ifndef IMAGE_PROCESSING_UTIL_HPP
#define IMAGE_PROCESSING_UTIL_HPP
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


class ImageProcessingUtil {
public:
    // Gets all image paths from a specified folder
    static std::vector<std::string> getImagePaths(const std::string& folderPath);

    // Extracts the shape of an image in tensor form
    static cv::Vec4f get_input_shape_from_image(const std::string& img_path);

    // Compares two images using PSNR and SSIM
    static void compareImages(const cv::Mat& generated_img, const cv::Mat& gray_original);

    // Processes an image, applies necessary augmentations, and returns it as a torch tensor
    static torch::Tensor process_img(const std::string& img_path, bool grayscale = false);
    static torch::Tensor process_img_batch(const vector<string>& img_paths, bool grayscale = false);
};

#endif 