/**
* No need to modify this file
*/

#include "ImageProcessingUtil.hpp"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
using namespace std;


std::vector<std::string> ImageProcessingUtil::getImagePaths(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    for (const auto& entry : filesystem::recursive_directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            // Check if the file extension is one of the common image formats
            std::string extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                imagePaths.push_back(entry.path().string());
            }
        }
    }
    return imagePaths;
}


cv::Vec4f ImageProcessingUtil::get_input_shape_from_image(const string& img_path) {
    cv::Mat image = cv::imread(img_path);
    return cv::Vec4f(1, image.channels(), image.rows, image.cols);
}

void ImageProcessingUtil::compareImages(const cv::Mat& generated_img, const cv::Mat& gray_original) {
    cv::Mat generated_img_clamped;
    cv::min(generated_img, 1.0, generated_img_clamped); // Clamp to 1.0
    cv::max(generated_img_clamped, 0.0, generated_img_clamped); // Clamp to 0.0

    cout << "generated_img size: " << generated_img.rows << "x" << generated_img.cols << " type: " << generated_img.type() << endl;
    cout << "gray_original size: " << gray_original.rows << "x" << gray_original.cols << " type: " << gray_original.type() << endl;   

    double psnr = cv::PSNR(generated_img, gray_original);
    double ssim = cv::quality::QualitySSIM::compute(generated_img, gray_original, cv::noArray())[0];

    cout << "PSNR: " << psnr << endl;
    cout << "SSIM: " << ssim << endl;
}

torch::Tensor ImageProcessingUtil::process_img(const string& img_path, bool grayscale) {
    cv::Mat img;
    if (grayscale) {
        img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            throw invalid_argument("Failed to load image at " + img_path);
        }
        img.convertTo(img, CV_32FC1, 1.0 / 255);

        auto img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, torch::kFloat32).clone();
        img_tensor = img_tensor.unsqueeze(0); // Add batch dimension
        cout << "Processed BW tensor.shape: " << img_tensor.sizes() << endl;
        return img_tensor;
    } 
    else {
        img = cv::imread(img_path, cv::IMREAD_COLOR); // Loaded in color (BGR format)
        if (img.empty()) {
            throw invalid_argument("Failed to load image at " + img_path);
        }
        
        img.convertTo(img, CV_32FC3, 1.0 / 255);
        
        // Create a tensor from the RGB image data, add a batch dimension, and clone it to ensure it owns its data
        auto img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32).clone();

        img_tensor = img_tensor.permute({2, 0, 1});
        auto rgb_tensor = img_tensor.index_select(0, torch::tensor({2, 1, 0}));
        auto din = rgb_tensor.unsqueeze(0);

        // Normalize the tensor
        auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}).to(din.options());
        auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}).to(din.options());

        auto din_normalized = (din - mean) / std;

        cout << "Processed din_normalized.shape: " << din_normalized.sizes() << endl;
        auto min_val = din_normalized.min().item<float>();
        auto max_val = din_normalized.max().item<float>();
        auto avg_val = din_normalized.mean().item<float>();
        cout << "din_normalized IMG Tensor - Min: " << min_val << ", Max: " << max_val << ", Avg: " << avg_val << endl;

        return din_normalized;
    }
}

torch::Tensor ImageProcessingUtil::process_img_batch(const vector<string>& img_paths, bool grayscale) {
    vector<torch::Tensor> img_tensors;
    img_tensors.reserve(img_paths.size());

    for (const auto& img_path : img_paths) {
        cv::Mat img;
        if (grayscale) {
            img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                throw invalid_argument("Failed to load image at " + img_path);
            }
            img.convertTo(img, CV_32FC1, 1.0 / 255);

            auto img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 1}, torch::kFloat32).clone();
            img_tensor = img_tensor.unsqueeze(0); // Add batch dimension
            img_tensors.push_back(img_tensor);
        } else {
            img = cv::imread(img_path, cv::IMREAD_COLOR); // Loaded in color (BGR format)
            if (img.empty()) {
                throw invalid_argument("Failed to load image at " + img_path);
            }

            img.convertTo(img, CV_32FC3, 1.0 / 255);

            // Create a tensor from the RGB image data, add a batch dimension, and clone it to ensure it owns its data
            auto img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kFloat32).clone();

            img_tensor = img_tensor.permute({2, 0, 1});
            auto rgb_tensor = img_tensor.index_select(0, torch::tensor({2, 1, 0}));
            auto din = rgb_tensor.unsqueeze(0);

            // Normalize the tensor
            auto mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}).to(din.options());
            auto std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}).to(din.options());

            auto din_normalized = (din - mean) / std;
            img_tensors.push_back(din_normalized);
        }
    }

    auto batched_tensor = torch::cat(img_tensors, 0); // Concatenate along the batch dimension

    return batched_tensor;
}
