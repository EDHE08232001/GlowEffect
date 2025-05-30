/**
 * @file ImageProcessingUtil.cpp
 * @brief Implements image-related utility functions for loading and processing images.
 *
 * This file includes functions for retrieving image paths, extracting image shapes,
 * comparing images (using PSNR), and converting images or batches of images
 * to Torch tensors for inference.
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
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <nvToolsExt.h>

 /**
  * @brief Retrieves all valid image file paths (jpg, jpeg, png, bmp) under a given folder.
  *
  * @param folderPath The path to the folder to scan (recursively).
  * @return A std::vector<std::string> containing full file paths for valid image files.
  * @throws Any filesystem exceptions if directory access fails.
  */
std::vector<std::string> ImageProcessingUtil::getImagePaths(const std::string& folderPath) {
	std::vector<std::string> imagePaths;

	for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
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

/**
 * @brief Gets the shape (1, channels, rows, cols) of an image at @p img_path.
 *
 * @param img_path The file path of the image.
 * @return A 4D vector in the form (1, channels, rows, cols).
 * @throws std::runtime_error or cv::Exception if the image fails to load.
 */
cv::Vec4f ImageProcessingUtil::get_input_shape_from_image(const std::string& img_path) {
	cv::Mat image = cv::imread(img_path);
	if (image.empty()) {
		throw std::runtime_error("Failed to load image at " + img_path);
	}
	return cv::Vec4f(1, static_cast<float>(image.channels()), static_cast<float>(image.rows), static_cast<float>(image.cols));
}

/**
 * @brief Compares a generated image with a reference grayscale image, computing PSNR.
 *
 * @param generated_img The generated image (float or double Mat, range typically 0..1).
 * @param gray_original The reference grayscale image.
 * @note Logs the PSNR result to std::cout.
 */
void ImageProcessingUtil::compareImages(const cv::Mat& generated_img, const cv::Mat& gray_original) {
	cv::Mat generated_img_clamped;

	// Clamp values to the range [0, 1]
	cv::min(generated_img, 1.0, generated_img_clamped);
	cv::max(generated_img_clamped, 0.0, generated_img_clamped);

	std::cout << "generated_img size: " << generated_img.rows << "x" << generated_img.cols
		<< " type: " << generated_img.type() << std::endl;
	std::cout << "gray_original size: " << gray_original.rows << "x" << gray_original.cols
		<< " type: " << gray_original.type() << std::endl;

	double psnr = cv::PSNR(generated_img, gray_original);
	std::cout << "PSNR: " << psnr << std::endl;

//    std::cout << "PSNR: " << psnr << std::endl;
//    std::cout << "SSIM: " << ssim << std::endl;

	// The following SSIM computation was removed because cv::quality::QualitySSIM
	// is not available in your OpenCV build.
	std::cout << "SSIM: not computed (cv::quality::QualitySSIM not available)" << std::endl;

}

/**
 * @brief Loads an image from @p img_path and optionally converts it to grayscale, then to a Torch tensor.
 *
 * @param img_path The path to the image file.
 * @param grayscale If true, loads as grayscale and returns shape [1, H, W]. If false, returns shape [1, 3, H, W].
 * @return A Torch tensor suitable for inference (including normalization if @p grayscale is false).
 * @throws std::invalid_argument if the image fails to load.
 */
torch::Tensor ImageProcessingUtil::process_img(const std::string& img_path, bool grayscale) {
	cv::Mat img;
	if (grayscale) {
		img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			throw std::invalid_argument("Failed to load image at " + img_path);
		}
		img.convertTo(img, CV_32FC1, 1.0f / 255.0f);

		auto img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 1 }, torch::kFloat32).clone();
		img_tensor = img_tensor.unsqueeze(0); // Add batch dimension
		std::cout << "Processed BW tensor.shape: " << img_tensor.sizes() << std::endl;
		return img_tensor;
	}
	else {
		img = cv::imread(img_path, cv::IMREAD_COLOR); // BGR format
		if (img.empty()) {
			throw std::invalid_argument("Failed to load image at " + img_path);
		}

        /*std::cout << "[DEBUG] Loaded image from " << img_path
            << " with dimensions: " << img.rows << " x " << img.cols
            << " x " << img.channels() << std::endl;*/

        img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

        // std::cout << "[DEBUG] Converted image to CV_32FC3 and scaled pixel values to [0, 1]." << std::endl;

        auto img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kFloat32).clone();
        /*std::cout << "[DEBUG] Created tensor from image blob, initial shape: "
            << img_tensor.sizes() << std::endl;*/

        img_tensor = img_tensor.permute({ 2, 0, 1 }); // [H, W, C] -> [C, H, W]
        /*std::cout << "[DEBUG] Permuted tensor to [C, H, W] format, new shape: "
            << img_tensor.sizes() << std::endl;*/

        // Convert BGR to RGB: 
        auto rgb_tensor = img_tensor.index_select(0, torch::tensor({ 2, 1, 0 }));
       /* std::cout << "[DEBUG] Reordered channels from BGR to RGB using index_select, shape: "
            << rgb_tensor.sizes() << std::endl;*/
        auto din = rgb_tensor.unsqueeze(0); // Add batch dimension: [1, 3, H, W]
        /*std::cout << "[DEBUG] Added batch dimension, tensor shape: "
            << din.sizes() << std::endl;*/

        // Normalize the tensor
        auto mean = torch::tensor({ 0.485f, 0.456f, 0.406f }).view({ 1, 3, 1, 1 }).to(din.options());
        auto std = torch::tensor({ 0.229f, 0.224f, 0.225f }).view({ 1, 3, 1, 1 }).to(din.options());
        auto din_normalized = (din - mean) / std;
        /*std::cout << "[DEBUG] Normalized tensor using ImageNet statistics, tensor shape: "
            << din_normalized.sizes() << std::endl;*/

		std::cout << "Processed din_normalized.shape: " << din_normalized.sizes() << std::endl;
		float min_val = din_normalized.min().item<float>();
		float max_val = din_normalized.max().item<float>();
		float avg_val = din_normalized.mean().item<float>();
		std::cout << "din_normalized IMG Tensor - Min: " << min_val
			<< ", Max: " << max_val << ", Avg: " << avg_val << std::endl;

		return din_normalized;
	}
}

torch::Tensor ImageProcessingUtil::process_img(const cv::cuda::GpuMat& gpu_img, bool grayscale)
{
    /*cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, gpu_img.data);
    if (err == cudaSuccess) {
        if (attributes.type == cudaMemoryTypeDevice) {
            std::cout << "The passed in frame is on the GPU." << std::endl;
        }
        else {
            std::cout << "The passed in frame is NOT on the GPU." << std::endl;
        }
    }
    else {
        std::cerr << "Error obtaining pointer attributes: " << cudaGetErrorString(err) << std::endl;
    }*/

    if (grayscale) {
        std::cout << "Grayscale" << std::endl;
        // --- Grayscale Processing ---
        cv::cuda::GpuMat gpu_gray;
        // If the input image is in color (3 channels), convert it to grayscale on the GPU.
        if (gpu_img.channels() == 3) {
            cv::cuda::cvtColor(gpu_img, gpu_gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gpu_gray = gpu_img;
        }

        // Convert to 32-bit float in the range [0, 1] if necessary.
        cv::cuda::GpuMat gpu_gray_float;
        if (gpu_gray.type() != CV_32FC1) {
            gpu_gray.convertTo(gpu_gray_float, CV_32FC1, 1.0f / 255.0f);
        }
        else {
            gpu_gray_float = gpu_gray;
        }

        // Get image dimensions.
        int rows = gpu_gray_float.rows;
        int cols = gpu_gray_float.cols;
        int channels = 1;

        // Wrap the GPU memory into a Torch tensor. Note that the data pointer is assumed to be contiguous
        // in GPU memory. We clone the tensor so that it owns its memory.
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        torch::Tensor temp_tensor = torch::from_blob(gpu_gray_float.data, { rows, cols, channels }, options); // comfirm with nvtx
        torch::Tensor img_tensor = temp_tensor.clone();

        // Add a batch dimension. From shape [rows, cols, 1] to [1, rows, cols, 1],
        // then permute to [1, 1, rows, cols].
        img_tensor = img_tensor.unsqueeze(0);          // shape becomes {1, rows, cols, 1}
        img_tensor = img_tensor.permute({ 0, 3, 1, 2 });   // shape becomes {1, 1, rows, cols}

        std::cout << "Processed BW tensor.shape: " << img_tensor.sizes() << std::endl;
        return img_tensor;
    }
    else {
        std::cout << "Color" << std::endl;
        // --- Color Processing ---
        cv::cuda::GpuMat gpu_color;
        // If the image is not already a 32-bit 3-channel float in [0,1], convert it.
        if (gpu_img.channels() != 3 || gpu_img.type() != CV_32FC3) {
            gpu_img.convertTo(gpu_color, CV_32FC3, 1.0f / 255.0f);
        }
        else {
            gpu_color = gpu_img;
        }

        int rows = gpu_color.rows;
        int cols = gpu_color.cols;
        int channels = gpu_color.channels();
        // std::cout << "[DEBUG] cv::cuda::GpuMat dimensions: " << rows << " x " << cols << " x " << channels << std::endl;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        torch::Tensor temp_tensor = torch::from_blob(gpu_color.data, { rows, cols, channels }, options);

        // nvtxRangePush("clone operation");
        torch::Tensor img_tensor = temp_tensor.clone();
       //  nvtxRangePop();
        // std::cout << "[Important] Cloned tensor device: " << img_tensor.device() << std::endl;
        // std::cout << "[DEBUG] After from_blob and clone, tensor shape: " << img_tensor.sizes() << std::endl;

        // Rearrange dimensions from [rows, cols, 3] to [3, rows, cols].
        img_tensor = img_tensor.permute({ 2, 0, 1 });
        // std::cout << "[DEBUG] After permute({2,0,1}), tensor shape: " << img_tensor.sizes() << std::endl;

        // Convert BGR to RGB.
        // In the original function, this was done using index_select:
        //    auto rgb_tensor = img_tensor.index_select(0, torch::tensor({ 2, 1, 0 }));
        // However, to ensure compatibility on the GPU, we replicate this behavior with gather.
        int r = img_tensor.size(1);
        int c = img_tensor.size(2);
        // Create an index tensor with desired order: [2, 1, 0].
        torch::Tensor indices = torch::tensor({ 2, 1, 0 },
            torch::TensorOptions().dtype(torch::kLong)
            .device(torch::kCUDA))
            .view({ 3, 1, 1 })
            .expand({ 3, r, c });
        // std::cout << "[DEBUG] Indices tensor shape: " << indices.sizes() << std::endl;
        torch::Tensor rgb_tensor = img_tensor.gather(0, indices);
        // std::cout << "[DEBUG] After gather for BGR->RGB, tensor shape: " << rgb_tensor.sizes() << std::endl;

        // Add batch dimension to get shape [1, 3, rows, cols].
        torch::Tensor din = rgb_tensor.unsqueeze(0);
        // std::cout << "[DEBUG] After unsqueeze(0), tensor shape: " << din.sizes() << std::endl;

        // Normalize the tensor using ImageNet mean and std.
        auto mean = torch::tensor({ 0.485f, 0.456f, 0.406f }, options).view({ 1, 3, 1, 1 });
        auto std = torch::tensor({ 0.229f, 0.224f, 0.225f }, options).view({ 1, 3, 1, 1 });
        torch::Tensor din_normalized = (din - mean) / std;
        // std::cout << "[DEBUG] After normalization, tensor shape: " << din_normalized.sizes() << std::endl;
        std::cout << "Processed din_normalized.shape: " << din_normalized.sizes() << std::endl;
        float min_val = din_normalized.min().item<float>();
        float max_val = din_normalized.max().item<float>();
        float avg_val = din_normalized.mean().item<float>();
        std::cout << "din_normalized IMG Tensor - Min: " << min_val
            << ", Max: " << max_val << ", Avg: " << avg_val << std::endl;

        return din_normalized;
    }
}

/**
 * @brief Processes a batch of images into a single batched Torch tensor.
 *
 * Each image is processed individually and then concatenated along the batch dimension.
 *
 * @param img_paths A vector of image file paths.
 * @param grayscale Whether to load images as grayscale.
 * @return A 4D Torch tensor with shape [batch_size, channels, height, width].
 * @throws std::invalid_argument if any image fails to load.
 */
torch::Tensor ImageProcessingUtil::process_img_batch(const std::vector<std::string>& img_paths, bool grayscale) {
	std::vector<torch::Tensor> img_tensors;
	img_tensors.reserve(img_paths.size());

	for (const auto& img_path : img_paths) {
		cv::Mat img;
		if (grayscale) {
			img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
			if (img.empty()) {
				throw std::invalid_argument("Failed to load image at " + img_path);
			}
			img.convertTo(img, CV_32FC1, 1.0f / 255.0f);

			auto img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 1 }, torch::kFloat32).clone();
			img_tensor = img_tensor.unsqueeze(0); // Add batch dimension
			img_tensors.push_back(img_tensor);
		}
		else {
			img = cv::imread(img_path, cv::IMREAD_COLOR);
			if (img.empty()) {
				throw std::invalid_argument("Failed to load image at " + img_path);
			}
			img.convertTo(img, CV_32FC3, 1.0f / 255.0f);

			auto img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kFloat32).clone();
			img_tensor = img_tensor.permute({ 2, 0, 1 });
			// Convert BGR to RGB:
			auto rgb_tensor = img_tensor.index_select(0, torch::tensor({ 2, 1, 0 }));
			auto din = rgb_tensor.unsqueeze(0);

			// Normalize the tensor
			auto mean = torch::tensor({ 0.485f, 0.456f, 0.406f }).view({ 1, 3, 1, 1 }).to(din.options());
			auto std = torch::tensor({ 0.229f, 0.224f, 0.225f }).view({ 1, 3, 1, 1 }).to(din.options());
			auto din_normalized = (din - mean) / std;

			img_tensors.push_back(din_normalized);
		}
	}

	// Concatenate along the batch dimension (dim=0)
	auto batched_tensor = torch::cat(img_tensors, 0);
	return batched_tensor;
}
