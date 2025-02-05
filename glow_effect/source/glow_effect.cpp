/*******************************************************************************************************************
 * FILE NAME   : glow_effect.cpp
 * PROJECT NAME: Cuda Learning
 * DESCRIPTION : Implements various glow-related effects such as "blow" highlighting,
 *               mipmapping, and alpha blending to create bloom/glow effects on images and video frames.
 *               Integrates CUDA kernels, OpenCV, and TensorRT (for segmentation in the video pipeline).
 *               Intermediate file I/O is removed (except final video output).
 * VERSION     : 2022 DEC 14 - Yu Liu - Creation (Modified 2025 FEB 04 - Modified by ChatGPT)
 *******************************************************************************************************************/

#include "dilate_erode.hpp"
#include "gaussian_blur.hpp"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include "glow_effect.hpp"
#include "old_movies.cuh"
#include "all_common.h"
#include <torch/torch.h>
#include <vector>
#include "imageprocessingutil.hpp"
#include "trtinference.hpp"
#include <iostream>
#include <string>
#include <opencv2/cudawarping.hpp>
#include <filesystem>
#include "mipmap.h"

namespace fs = std::filesystem;

// Global boolean array indicating button states in GUI (for demonstration/testing).
bool button_State[5] = { false, false, false, false, false };

/**
 * @brief Applies a simple "blow" or highlight effect to an output RGBA image based on a grayscale mask.
 *
 * If any pixel in the input mask is within a tolerance range (Delta) of param_KeyLevel,
 * the entire output image is filled with a pink overlay; otherwise, it remains transparent.
 *
 * @param mask          A single-channel (CV_8UC1) mask.
 * @param dst_rgba      Output RGBA image (CV_8UC4); will be created/overwritten.
 * @param param_KeyLevel Key level parameter controlling the highlight trigger.
 * @param Delta         Tolerance range around param_KeyLevel.
 */
void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta) {
	if (mask.empty()) {
		std::cerr << "Error: Segmentation mask is empty." << std::endl;
		return;
	}
	if (mask.type() != CV_8UC1) {
		std::cerr << "Error: Mask is not of type CV_8UC1." << std::endl;
		return;
	}

	dst_rgba = cv::Mat::zeros(mask.size(), CV_8UC4);
	cv::Vec4b overlay_color = { 199, 170, 255, 255 }; // B, G, R, A
	bool has_target_region = false;

	for (int i = 0; i < mask.rows; ++i) {
		for (int j = 0; j < mask.cols; ++j) {
			int mask_pixel = mask.at<uchar>(i, j);
			if (std::abs(mask_pixel - param_KeyLevel) < Delta) {
				has_target_region = true;
				break;
			}
		}
		if (has_target_region)
			break;
	}

	if (has_target_region) {
		for (int i = 0; i < dst_rgba.rows; ++i) {
			for (int j = 0; j < dst_rgba.cols; ++j) {
				dst_rgba.at<cv::Vec4b>(i, j) = overlay_color;
			}
		}
	}

	std::cout << "glow_blow completed. Target region "
		<< (has_target_region ? "found and applied." : "not found.")
		<< std::endl;
}

/**
 * @brief Synchronously applies a CUDA-based mipmap filter to a grayscale image and outputs an RGBA image.
 *
 * Converts the input grayscale image to an RGBA buffer where only pixels equal to param_KeyLevel are opaque,
 * applies the CUDA mipmap filter synchronously, and converts the result back into an OpenCV RGBA image.
 *
 * @param input_gray     The source single-channel (CV_8UC1) grayscale image.
 * @param output_image   The destination RGBA image (CV_8UC4) after mipmap filtering.
 * @param scale          The scale factor used by the mipmap filter.
 * @param param_KeyLevel Grayscale value determining which pixels become opaque.
 */
void apply_mipmap(const cv::Mat& input_gray, cv::Mat& output_image, float scale, int param_KeyLevel) {
	int width = input_gray.cols;
	int height = input_gray.rows;

	// Initialize button states (for simulation/testing).
	for (int k = 0; k < 5; k++) {
		button_State[k] = true;
	}

	if (input_gray.channels() != 1 || input_gray.type() != CV_8UC1) {
		std::cerr << "Error: Input image must be a single-channel grayscale image." << std::endl;
		return;
	}

	// Allocate host memory for RGBA buffers.
	uchar4* src_img = new uchar4[width * height];
	uchar4* dst_img = new uchar4[width * height];

	// Convert grayscale image to an RGBA buffer, preserving only pixels equal to param_KeyLevel.
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			unsigned char gray_value = input_gray.at<uchar>(i, j);
			if (gray_value == param_KeyLevel) {
				src_img[i * width + j] = { gray_value, gray_value, gray_value, 255 };
			}
			else {
				src_img[i * width + j] = { 0, 0, 0, 0 };
			}
		}
	}

	// Synchronously apply the CUDA mipmap filter.
	filter_mipmap(width, height, scale, src_img, dst_img);

	output_image.create(height, width, CV_8UC4);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar4 value = dst_img[i * width + j];
			output_image.at<cv::Vec4b>(i, j) = cv::Vec4b(value.x, value.y, value.z, value.w);
		}
	}

	std::cout << "apply_mipmap: Completed synchronous mipmap filtering." << std::endl;

	delete[] src_img;
	delete[] dst_img;
}

/**
 * @brief Asynchronously applies a CUDA-based mipmap filter to a grayscale image and outputs an RGBA image.
 *
 * Converts the input grayscale image to an RGBA buffer (keeping only pixels equal to param_KeyLevel as opaque),
 * then uses the asynchronous filter_mipmap_async function to apply the mipmap filter, and converts the result
 * back into an OpenCV RGBA image.
 *
 * @param input_gray  The source single-channel (CV_8UC1) grayscale image.
 * @param output_image The destination RGBA image (CV_8UC4) after mipmap filtering.
 * @param scale       The scale factor used by the mipmap filter.
 * @param param_KeyLevel Grayscale value determining which pixels become opaque.
 */
void apply_mipmap_async(const cv::Mat& input_gray, cv::Mat& output_image, float scale, int param_KeyLevel) {
	int width = input_gray.cols;
	int height = input_gray.rows;

	// Allocate host memory for RGBA buffers.
	uchar4* src_img = new uchar4[width * height];
	uchar4* dst_img = new uchar4[width * height];

	// Convert the grayscale image to an RGBA buffer.
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			unsigned char gray_value = input_gray.at<uchar>(i, j);
			if (gray_value == param_KeyLevel)
				src_img[i * width + j] = { gray_value, gray_value, gray_value, 255 };
			else
				src_img[i * width + j] = { 0, 0, 0, 0 };
		}
	}

	// Create a dedicated CUDA stream for asynchronous mipmap processing.
	cudaStream_t mipmapStream;
	checkCudaErrors(cudaStreamCreate(&mipmapStream));

	// Asynchronously apply the CUDA mipmap filter.
	filter_mipmap_async(width, height, scale, src_img, dst_img, mipmapStream);

	// Synchronize the stream to ensure all asynchronous operations are complete.
	checkCudaErrors(cudaStreamSynchronize(mipmapStream));
	cudaStreamDestroy(mipmapStream);

	// Convert the output RGBA buffer back into an OpenCV image.
	output_image.create(height, width, CV_8UC4);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar4 value = dst_img[i * width + j];
			output_image.at<cv::Vec4b>(i, j) = cv::Vec4b(value.x, value.y, value.z, value.w);
		}
	}

	std::cout << "apply_mipmap_async: Completed asynchronous mipmap filtering." << std::endl;

	delete[] src_img;
	delete[] dst_img;
}

/**
 * @brief Blends two images using a mask and per-pixel alpha blending.
 *
 * Blends the source image with the highlighted image using a grayscale mask (converted to an alpha channel
 * scaled by param_KeyScale), and outputs the final blended RGBA image.
 *
 * @param src_img        First source image (converted to RGBA if needed).
 * @param dst_rgba       Second source image (highlighted; converted to RGBA if needed).
 * @param mipmap_result  Grayscale mask image used as the alpha channel.
 * @param output_image   Destination blended RGBA image.
 * @param param_KeyScale Blending factor scaling.
 */
void mix_images(const cv::Mat& src_img, const cv::Mat& dst_rgba, const cv::Mat& mipmap_result, cv::Mat& output_image, float param_KeyScale) {
	if (src_img.empty() || dst_rgba.empty() || mipmap_result.empty()) {
		std::cerr << "Error: One or more input images are empty." << std::endl;
		return;
	}
	if (src_img.size() != dst_rgba.size() || src_img.size() != mipmap_result.size()) {
		std::cerr << "Error: Images must have the same dimensions." << std::endl;
		return;
	}

	cv::Mat src_rgba;
	if (src_img.channels() != 4) {
		cv::cvtColor(src_img, src_rgba, cv::COLOR_BGR2BGRA);
	}
	else {
		src_rgba = src_img.clone();
	}

	cv::Mat high_lighted_rgba;
	if (dst_rgba.channels() != 4) {
		cv::cvtColor(dst_rgba, high_lighted_rgba, cv::COLOR_BGR2BGRA);
	}
	else {
		high_lighted_rgba = dst_rgba.clone();
	}

	cv::Mat mipmap_gray;
	if (mipmap_result.channels() != 1) {
		cv::cvtColor(mipmap_result, mipmap_gray, cv::COLOR_BGR2GRAY);
	}
	else {
		mipmap_gray = mipmap_result.clone();
	}

	output_image = src_rgba.clone();

	for (int i = 0; i < src_rgba.rows; ++i) {
		for (int j = 0; j < src_rgba.cols; ++j) {
			uchar original_alpha = mipmap_gray.at<uchar>(i, j);
			uchar alpha = (original_alpha * static_cast<int>(param_KeyScale)) >> 8;
			cv::Vec4b src_pixel = src_rgba.at<cv::Vec4b>(i, j);
			cv::Vec4b dst_pixel = high_lighted_rgba.at<cv::Vec4b>(i, j);
			cv::Vec4b& output_pixel = output_image.at<cv::Vec4b>(i, j);
			for (int k = 0; k < 4; ++k) {
				int temp_pixel = (src_pixel[k] * (255 - alpha) + dst_pixel[k] * alpha) >> 8;
				output_pixel[k] = static_cast<uchar>(std::min(255, std::max(0, temp_pixel)));
			}
		}
	}

	std::cout << "mix_images: Image blending completed successfully." << std::endl;
}

/**
 * @brief Applies a glow effect to a single image using a grayscale mask.
 *
 * Loads the source image, applies a glow highlight via glow_blow, performs a mipmap transformation,
 * blends the results, and displays the final output.
 *
 * @param image_nm       Path to the source image file.
 * @param grayscale_mask Single-channel mask guiding the glow effect.
 */
void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask) {
	cv::Mat src_img = cv::imread(image_nm);
	if (src_img.empty()) {
		std::cerr << "Error: Could not load source image." << std::endl;
		return;
	}

	cv::Mat dst_rgba;
	glow_blow(grayscale_mask, dst_rgba, param_KeyLevel, 10);

	cv::Mat mipmap_result;
	// For single-image processing, we use the synchronous mipmap filtering.
	apply_mipmap(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

	cv::Mat final_result;
	mix_images(src_img, dst_rgba, mipmap_result, final_result, param_KeyScale);

	cv::imshow("Final Result", final_result);
	cv::waitKey(0);
}

/**
 * @brief Applies a glow effect to a video file.
 *
 * Processes each frame of the input video by performing TensorRT segmentation, applying a glow highlight,
 * asynchronous mipmap filtering, and blending. The processed frames are written to an output video file.
 * Intermediate file-writing operations are removed except for the final video output.
 *
 * @param video_nm     Path to the input video file.
 * @param planFilePath Path to the TRT plan file.
 */
void glow_effect_video(const char* video_nm, std::string planFilePath) {
	cv::String info = cv::getBuildInformation();
	std::cout << info << std::endl;

	cv::VideoCapture video;
	if (!video.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY)) {
		std::cerr << "Error: Could not open video file: " << video_nm << std::endl;
		return;
	}

	int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps = static_cast<int>(video.get(cv::CAP_PROP_FPS));

	if (!fs::exists("./VideoOutput/")) {
		if (fs::create_directory("./VideoOutput/"))
			std::cout << "Video Output Directory successfully created." << std::endl;
		else
			std::cerr << "Failed to create video output folder." << std::endl;
	}
	else {
		std::cout << "Video Output Directory already exists." << std::endl;
	}

	std::string output_video_path = "./VideoOutput/processed_video.avi";
	cv::VideoWriter output_video(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps, cv::Size(frame_width, frame_height));
	if (!output_video.isOpened()) {
		std::cerr << "Error: Could not open the output video for writing: " << output_video_path << std::endl;
		return;
	}

	float* h_frame;
	size_t frameSize = frame_width * frame_height * 3;
	cudaMallocHost((void**)&h_frame, frameSize * sizeof(float));

	cv::cuda::GpuMat gpu_frame;
	std::vector<torch::Tensor> batch_frames;
	int frame_count = 0;

	while (video.isOpened()) {
		std::vector<cv::Mat> original_frames;
		batch_frames.clear();

		// Attempt to read 4 frames for batch processing.
		for (int i = 0; i < 4; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty())
					break;
				batch_frames.push_back(batch_frames.back());
				original_frames.push_back(original_frames.back().clone());
				continue;
			}
			original_frames.push_back(frame.clone());

			gpu_frame.upload(frame);
			cv::cuda::GpuMat resized_gpu_frame;
			cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));

			torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
			frame_tensor = frame_tensor.to(torch::kFloat);
			batch_frames.push_back(frame_tensor);
		}

		if (batch_frames.empty())
			break;
		while (batch_frames.size() < 4) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}

		torch::Tensor batch_tensor = torch::stack(batch_frames, 0);
		std::vector<cv::Mat> grayscale_masks = TRTInference::measure_segmentation_trt_performance_mul(planFilePath, batch_tensor, 1);

		if (!grayscale_masks.empty()) {
			for (int i = 0; i < 4; ++i) {
				cv::Mat grayscale_mask;
				cv::resize(grayscale_masks[i], grayscale_mask, original_frames[i].size());

				cv::Mat dst_rgba;
				glow_blow(grayscale_mask, dst_rgba, param_KeyLevel, 10);
				if (dst_rgba.channels() != 4) {
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				}

				cv::Mat mipmap_result;

				// Use asynchronous mipmap filtering for faster processing.
				apply_mipmap_async(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

				cv::Mat final_result;
				mix_images(original_frames[i], dst_rgba, mipmap_result, final_result, param_KeyScale);

				cv::imshow("Processed Frame", final_result);
				int key = cv::waitKey(30);
				if (key == 'q') {
					video.release();
					cv::destroyAllWindows();
					return;
				}

				output_video.write(final_result);
			}
		}
		else {
			std::cerr << "Warning: No grayscale mask generated for this batch." << std::endl;
		}
	}

	video.release();
	output_video.release();
	cudaFreeHost(h_frame);
	cv::destroyAllWindows();

	std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;
}
