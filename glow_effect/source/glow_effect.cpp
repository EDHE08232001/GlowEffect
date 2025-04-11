/*******************************************************************************************************************
 * FILE NAME   :    glow_effect.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    dilate or erode algorithm
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 DEC 14      Yu Liu          Creation
 *
 ********************************************************************************************************************/

 /* FILE NAME   : glow_effect.cpp
 * PROJECT NAME: Cuda Learning
 * DESCRIPTION : Implements various glow-related effects such as "blow" highlighting,
 *               mipmapping, and alpha blending to create bloom/glow effects on images and video frames.
 *               Integrates CUDA kernels, OpenCV, and TensorRT (for segmentation in the video pipeline).
 *               Uses triple buffering with non-blocking streams (created using cudaStreamNonBlocking)
 *               to accelerate asynchronous mipmap filtering.
 * IMPORTANT:  The engine is built for a fixed input shape ([4,3,384,384]). We ensure that every batch
 *             is padded to 4 images and remove any extra singleton dimensions. Additionally, before
 *             calling cv::resize, we check that the input image has valid dimensions.
 * VERSION     : Updated 2025 FEB 04 (with additional error-checking and exception handling)
 *******************************************************************************************************************/


#include "dilate_erode.hpp"
#include "gaussian_blur.hpp"
#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include "glow_effect.hpp"
#include "old_movies.cuh"
#include "glow_effect/all_common.h"
#include <torch/torch.h>
#include <vector>
#include "imageprocessingutil.hpp"
#include "trtinference.hpp"
#include <iostream>
#include <string>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <nvToolsExt.h>
#include "resizeWithTexture.h"
#include <filesystem>
#include "mipmap.h"
#include "helper_cuda.h"  // For checkCudaErrors
#include <future>         // For std::async, std::future
#include <exception>
#include <segmentation_kernels.h>

#define FILTER_MORPH 1
#define FILTER_GAUSS 1


namespace fs = std::filesystem;

// Global boolean array indicating button states (for demonstration/testing).
bool button_State[5] = { false, false, false, false, false };

// Helper Visualization
void visualize_segmentation_regions(const cv::Mat& original_frame, const cv::Mat& mask, int param_KeyLevel, int Delta) {
	// Create a visualization image by blending original frame with colored regions
	cv::Mat visualization;
	if (original_frame.channels() == 3) {
		visualization = original_frame.clone();
	}
	else {
		cv::cvtColor(original_frame, visualization, cv::COLOR_BGRA2BGR);
	}

	// Apply a colored overlay to matched regions
	for (int i = 0; i < mask.rows; ++i) {
		for (int j = 0; j < mask.cols; ++j) {
			int mask_pixel = mask.at<uchar>(i, j);
			if (std::abs(mask_pixel - param_KeyLevel) < Delta) {
				// Draw a bright cyan highlight
				cv::Vec3b& pixel = visualization.at<cv::Vec3b>(i, j);
				// Blend with original (50% original, 50% highlight)
				pixel[0] = pixel[0] * 0.5 + 255 * 0.5; // B
				pixel[1] = pixel[1] * 0.5 + 255 * 0.5; // G
				pixel[2] = pixel[2] * 0.5;             // R
			}
		}
	}

	// Show the visualization
	cv::imshow("Segmentation Visualization", visualization);
}

////////////////////////////////////////////////////////////////////////////////
// Helper Function: convert_mask_to_rgba_buffer
////////////////////////////////////////////////////////////////////////////////
void convert_mask_to_rgba_buffer(const cv::Mat& mask, uchar4* dst, int frame_width, int frame_height, int param_KeyLevel) {
	for (int i = 0; i < frame_height; ++i) {
		for (int j = 0; j < frame_width; ++j) {
			unsigned char gray_value = mask.at<uchar>(i, j);
			if (gray_value == param_KeyLevel)
				dst[i * frame_width + j] = { gray_value, gray_value, gray_value, 255 };
			else
				dst[i * frame_width + j] = { 0, 0, 0, 0 };
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Helper Function: triple_buffered_mipmap_pipeline
////////////////////////////////////////////////////////////////////////////////
std::vector<cv::Mat> triple_buffered_mipmap_pipeline(const std::vector<cv::Mat>& resized_masks,
	int frame_width, int frame_height,
	float default_scale, int param_KeyLevel) {
	int N = resized_masks.size();
	const int numBuffers = 3;
	std::vector<cv::Mat> outputImages(N);

	std::vector<uchar4*> tripleSrc(numBuffers, nullptr);
	std::vector<uchar4*> tripleDst(numBuffers, nullptr);
	std::vector<cudaStream_t> mipmapStreams(numBuffers);
	std::vector<cudaEvent_t> mipmapDone(numBuffers);

	for (int i = 0; i < numBuffers; ++i) {
		checkCudaErrors(cudaStreamCreateWithFlags(&mipmapStreams[i], cudaStreamNonBlocking));
		checkCudaErrors(cudaEventCreate(&mipmapDone[i]));
		checkCudaErrors(cudaMallocHost((void**)&tripleSrc[i], frame_width * frame_height * sizeof(uchar4)));
		checkCudaErrors(cudaMallocHost((void**)&tripleDst[i], frame_width * frame_height * sizeof(uchar4)));
	}

	for (int i = 0; i < N + 2; ++i) {
		if (i < N) {
			int bufIdx = i % numBuffers;
			convert_mask_to_rgba_buffer(resized_masks[i], tripleSrc[bufIdx], frame_width, frame_height, param_KeyLevel);
			apply_mipmap_async(resized_masks[i], tripleDst[bufIdx], default_scale, param_KeyLevel, mipmapStreams[bufIdx]);
			checkCudaErrors(cudaEventRecord(mipmapDone[bufIdx], mipmapStreams[bufIdx]));
		}
		if (i - 2 >= 0 && (i - 2) < N) {
			int bufIdx = (i - 2) % numBuffers;
			cudaError_t query = cudaEventQuery(mipmapDone[bufIdx]);
			if (query == cudaSuccess) {
				cv::Mat mipmapResult(frame_height, frame_width, CV_8UC4);
				memcpy(mipmapResult.data, tripleDst[bufIdx], frame_width * frame_height * sizeof(uchar4));
				outputImages[i - 2] = mipmapResult;
			}
			else if (query != cudaErrorNotReady) {
				checkCudaErrors(query);
			}
		}
	}

	for (int i = 0; i < numBuffers; ++i) {
		checkCudaErrors(cudaFreeHost(tripleSrc[i]));
		checkCudaErrors(cudaFreeHost(tripleDst[i]));
		checkCudaErrors(cudaStreamDestroy(mipmapStreams[i]));
		checkCudaErrors(cudaEventDestroy(mipmapDone[i]));
	}

	return outputImages;
}

////////////////////////////////////////////////////////////////////////////////
// Function: glow_blow
////////////////////////////////////////////////////////////////////////////////
void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta) {
	if (mask.empty()) {
		std::cerr << "Error: Segmentation mask is empty." << std::endl;
		return;
	}
	if (mask.type() != CV_8UC1) {
		std::cerr << "Error: Mask is not of type CV_8UC1." << std::endl;
		return;
	}

	// Create a destination image with zeros
	dst_rgba = cv::Mat::zeros(mask.size(), CV_8UC4);

	// Use a more vibrant color for better visibility
	cv::Vec4b overlay_color = { 147, 20, 226, 255 };

	// Variables to track target region information
	bool has_target_region = false;
	int target_pixel_count = 0;
	int min_x = mask.cols, max_x = 0;
	int min_y = mask.rows, max_y = 0;

	// Process the mask
	for (int i = 0; i < mask.rows; ++i) {
		for (int j = 0; j < mask.cols; ++j) {
			int mask_pixel = mask.at<uchar>(i, j);

			if (std::abs(mask_pixel - param_KeyLevel) < Delta) {
				has_target_region = true;
				target_pixel_count++;

				// Track bounding box of target region
				min_x = std::min(min_x, j);
				max_x = std::max(max_x, j);
				min_y = std::min(min_y, i);
				max_y = std::max(max_y, i);

				// Apply overlay ONLY to pixels that match the target value
				dst_rgba.at<cv::Vec4b>(i, j) = overlay_color;
			}
		}
	}
	// Print the result of the operation.
	// std::cout << "glow_blow completed. Target region " << (has_target_region ? "found and applied." : "not found.") << std::endl;

	// Print the target region information that was specifically requested
	if (has_target_region) {
		// Calculate percentage of frame covered
		double coverage_percent = (static_cast<double>(target_pixel_count) / (mask.rows * mask.cols)) * 100.0;
		std::cout << "Target region found!" << std::endl;
		std::cout << "  - Pixels matching target: " << target_pixel_count << " (" << coverage_percent << "% of frame)" << std::endl;
		std::cout << "  - Region bounding box: (" << min_x << "," << min_y << ") to (" << max_x << "," << max_y << ")" << std::endl;
		std::cout << "  - Box dimensions: " << (max_x - min_x + 1) << "x" << (max_y - min_y + 1) << std::endl;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Function: apply_mipmap (Synchronous Version)
////////////////////////////////////////////////////////////////////////////////

void apply_mipmap(const cv::Mat& input_gray, cv::Mat& output_image, float scale, int param_KeyLevel) {
	// Retrieve image dimensions.
	int width = input_gray.cols;
	int height = input_gray.rows;

	// Initialize button_State array (used for simulation purposes).
	for (int k = 0; k < 5; k++) {
		button_State[k] = true; // Simulate initialization similar to test_mipmap.
	}

	// Check if the input image is a valid single-channel grayscale image.
	if (input_gray.channels() != 1 || input_gray.type() != CV_8UC1) {
		std::cerr << "Error: Input image must be a single-channel grayscale image." << std::endl;
		return;
	}

	// Save the input grayscale image for debugging purposes.
	cv::imwrite("./pngOutput/input_gray_before_mipmap.png", input_gray);
	std::cout << "Input gray image saved as input_gray_before_mipmap.png" << std::endl;

	// Allocate memory for uchar4 arrays to store image data.
	uchar4* src_img = new uchar4[width * height];
	uchar4* dst_img = new uchar4[width * height];

	// Process the input grayscale image:
	// Convert it to an RGBA format (uchar4), preserving only pixels matching the key level.
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			unsigned char gray_value = input_gray.at<uchar>(i, j);

			// Preserve pixels matching param_KeyLevel; others are set to transparent.
			if (gray_value == param_KeyLevel) {
				unsigned char num = param_KeyLevel;
				src_img[i * width + j] = { num, num, num, 255 }; // Fully opaque.
			}
			else {
				src_img[i * width + j] = { 0, 0, 0, 0 }; // Transparent.
			}
		}
	}

	// Convert uchar4 array to OpenCV RGBA image for debugging purposes.
	cv::Mat uchar4_image_before(height, width, CV_8UC4);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar4 value = src_img[i * width + j];
			uchar4_image_before.at<cv::Vec4b>(i, j) = cv::Vec4b(value.x, value.y, value.z, value.w);
		}
	}
	cv::imwrite("./pngOutput/converted_uchar4_before_mipmap.png", uchar4_image_before);
	std::cout << "Converted uchar4 image saved as converted_uchar4_before_mipmap.png" << std::endl;

	// Apply the mipmap filter operation.
	filter_mipmap(width, height, scale, src_img, dst_img);

	// Convert the filtered uchar4 array back to an OpenCV RGBA image.
	output_image.create(height, width, CV_8UC4);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar4 value = dst_img[i * width + j];
			output_image.at<cv::Vec4b>(i, j) = cv::Vec4b(value.x, value.y, value.z, value.w);
		}
	}

	// Save the final output RGBA image for debugging purposes.
	cv::imwrite("./pngOutput/output_image_after_mipmap.png", output_image);
	std::cout << "Output RGBA image saved as output_image_after_mipmap.png" << std::endl;

	// Release dynamically allocated memory.
	delete[] src_img;
	delete[] dst_img;
}

/**
 * @brief Mixes three images (source, highlighted, and mipmap) into a single output image using alpha blending.
 *
 * This function combines the source image, a highlighted image, and a mipmap result image. The mipmap result's
 * grayscale values are used as the alpha channel to blend the source and highlighted images, scaled by a key factor.
 *
 * @param src_img Input source image (must have 3 or 4 channels).
 * @param dst_rgba Input highlighted image (must have 4 channels).
 * @param mipmap_result Input mipmap image (must be single-channel grayscale or convertible to grayscale).
 * @param output_image Output image after blending (CV_8UC4).
 * @param param_KeyScale Scaling factor for alpha channel values.
 */
////////////////////////////////////////////////////////////////////////////////
// Function: apply_mipmap_async (Asynchronous Version)
////////////////////////////////////////////////////////////////////////////////
void apply_mipmap_async(const cv::Mat& input_gray, uchar4* dst_img, float scale, int param_KeyLevel, cudaStream_t stream) {
	int width = input_gray.cols;
	int height = input_gray.rows;

	uchar4* src_img = nullptr;
	checkCudaErrors(cudaMallocHost((void**)&src_img, width * height * sizeof(uchar4)));

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			unsigned char gray_value = input_gray.at<uchar>(i, j);
			if (gray_value == param_KeyLevel)
				src_img[i * width + j] = { gray_value, gray_value, gray_value, 255 };
			else
				src_img[i * width + j] = { 0, 0, 0, 0 };
		}
	}

	filter_mipmap_async(width, height, scale, src_img, dst_img, stream);

	cudaStreamAddCallback(stream,
		[](cudaStream_t stream, cudaError_t status, void* userData) {
			uchar4* ptr = static_cast<uchar4*>(userData);
			cudaFreeHost(ptr);
		},
		src_img, 0);

	std::cout << "apply_mipmap_async: Launched asynchronous mipmap filtering on non-blocking stream." << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Function: mix_images
////////////////////////////////////////////////////////////////////////////////
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
	if (src_img.channels() != 4)
		cv::cvtColor(src_img, src_rgba, cv::COLOR_BGR2BGRA);
	else
		src_rgba = src_img.clone();

	cv::Mat high_lighted_rgba;
	if (dst_rgba.channels() != 4)
		cv::cvtColor(dst_rgba, high_lighted_rgba, cv::COLOR_BGR2BGRA);
	else
		high_lighted_rgba = dst_rgba.clone();

	cv::Mat mipmap_gray;
	if (mipmap_result.channels() != 1)
		cv::cvtColor(mipmap_result, mipmap_gray, cv::COLOR_BGR2GRAY);
	else
		mipmap_gray = mipmap_result.clone();

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


cv::Mat threshold_mask_to_rgba(const cv::Mat& grayscale_mask, int param_KeyLevel, int tolerance)
{
	CV_Assert(grayscale_mask.type() == CV_8UC1);
	cv::Mat mask_rgba(grayscale_mask.size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

	for (int i = 0; i < grayscale_mask.rows; ++i) {
		for (int j = 0; j < grayscale_mask.cols; ++j) {
			uchar pixel = grayscale_mask.at<uchar>(i, j);
			// If pixel is exactly equal (or within tolerance) to param_KeyLevel, make it opaque.
			if (std::abs(pixel - param_KeyLevel) <= tolerance) {
				mask_rgba.at<cv::Vec4b>(i, j) = cv::Vec4b(pixel, pixel, pixel, 255);
			}
			else {
				mask_rgba.at<cv::Vec4b>(i, j) = cv::Vec4b(0, 0, 0, 0);
			}
		}
	}
	return mask_rgba;
}

/**
 * @brief Applies a glow effect to an input image using a grayscale mask.
 *
 * This function combines various operations including generating a highlighted effect,
 * applying a mipmap operation, and blending multiple images to produce a final result.
 *
 * @param image_nm Path to the source image file.
 * @param grayscale_mask Input grayscale mask image (CV_8UC1).
 */
////////////////////////////////////////////////////////////////////////////////
// Function: glow_effect_image
////////////////////////////////////////////////////////////////////////////////
void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask) {
	cv::Mat src_img = cv::imread(image_nm);
	if (src_img.empty()) {
		std::cerr << "Error: Could not load source image." << std::endl;
		return;
	}

	cv::Mat dst_rgba;
	glow_blow(grayscale_mask, dst_rgba, param_KeyLevel, 10);

	cv::Mat mipmap_result;
	apply_mipmap(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

	cv::Mat final_result;
	mix_images(src_img, dst_rgba, mipmap_result, final_result, param_KeyScale);

	cv::imshow("Final Result", final_result);
	cv::waitKey(0);
}


/**
* Old pipeline.
 * @brief Applies a glow effect to each frame of a video using segmentation and image processing.
 *
 * This function processes a video frame by frame, applying a glow effect based on a segmentation mask.
 * Each frame is processed using TensorRT for inference, mipmap operations, and blending to create the final result.
 * The processed frames are saved and combined into a new video.
 *
 * @param video_nm Path to the input video file.
 */
void glow_effect_video(const char* video_nm) {
	auto start_time = std::chrono::high_resolution_clock::now();
	// Print OpenCV build information for debugging.
	cv::String info = cv::getBuildInformation();
	std::cout << info << std::endl;

	cv::VideoCapture video;
	bool pause = false;

	// Open the video file.
	if (!video.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY)) {
		std::cerr << "Error: Could not open video file: " << video_nm << std::endl;
		return;
	}

	// Retrieve video parameters.
	int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps = static_cast<int>(video.get(cv::CAP_PROP_FPS));

	// Create an output folder for processed frames.
	std::string output_folder = "./VideoOutput";
	std::filesystem::create_directory(output_folder);

	// Define the TensorRT plan file path for segmentation.
	std::string planFilePath = "D:/csi4900/TRT-Plans/mobileone_s4.lw.plan";          // user config

	cv::Mat src_img, dst_img;
	int frame_count = 0; // Counter for saved frames.

	while (video.isOpened()) {
		// Prepare a batch of frames.
		std::vector<torch::Tensor> batch_frames;
		std::vector<cv::Mat> original_frames;

		for (int i = 0; i < 4; ++i) {
			if (!video.read(src_img) || src_img.empty()) {
				if (batch_frames.empty()) break; // End of video.
				batch_frames.push_back(batch_frames.back()); // Pad batch with last frame.
				original_frames.push_back(original_frames.back().clone());
				continue;
			}

			original_frames.push_back(src_img.clone());

			// Resize the frame to 384x384 for inference.
			cv::Mat resized_img;
			cv::resize(src_img, resized_img, cv::Size(384, 384));

			// Save temporary resized image.
			std::string temp_img_path = "./temp_video_frame_" + std::to_string(i) + ".png";
			cv::imwrite(temp_img_path, resized_img);

			// Convert the image to a tensor.
			torch::Tensor frame_tensor = ImageProcessingUtil::process_img(temp_img_path, false);
			frame_tensor = frame_tensor.to(torch::kFloat);
			batch_frames.push_back(frame_tensor);

			// Remove the temporary file.
			std::filesystem::remove(temp_img_path);
		}

		if (batch_frames.empty()) break; // All frames processed.

		// Pad batch if it contains fewer than 4 frames.
		while (batch_frames.size() < 4) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}

		// Stack tensors into a batch.
		torch::Tensor batch_tensor = torch::stack(batch_frames, 0); // Shape: [4, 3, 384, 384]

		// Perform TensorRT inference to obtain segmentation masks.
		std::vector<cv::Mat> grayscale_masks = TRTInference::measure_segmentation_trt_performance_mul(planFilePath, batch_tensor, 1);

		/*for (int i = 0; i < grayscale_masks.size(); i++) {
			cv::imshow("Segmentation Mask", grayscale_masks[i]);
			cv::waitKey(1);
		}*/


		if (!grayscale_masks.empty()) {
			for (int i = 0; i < 4; ++i) {
				cv::Mat grayscale_mask;
				cv::resize(grayscale_masks[i], grayscale_mask, original_frames[i].size());

				// Generate a glow effect using the mask.
				cv::Mat dst_rgba;
				glow_blow(grayscale_mask, dst_rgba, param_KeyLevel, 10);

				if (dst_rgba.channels() != 4) {
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				}

				// Apply mipmap operations.
				cv::Mat mipmap_result;
				apply_mipmap(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

				// Blend frames using the glow effect and mipmap results.
				cv::Mat final_result;
				mix_images(original_frames[i], dst_rgba, mipmap_result, final_result, param_KeyScale);

				// Display the processed frame.
				cv::imshow("Processed Frame", final_result);
				int key = cv::waitKey(30);
				if (key == 'q') {
					video.release();
					cv::destroyAllWindows();
					return;
				}

				// Save the processed frame.
				std::string frame_output_path = output_folder + "/frame_" + std::to_string(frame_count++) + ".png";
				cv::imwrite(frame_output_path, final_result);
			}
		}
		else {
			std::cerr << "Warning: No grayscale mask generated for this batch." << std::endl;
		}
	}

	// Release video resources and close display windows.
	video.release();
	cv::destroyAllWindows();

	// Create the output video from processed frames.
	std::string output_video_path = output_folder + "/processed_video.avi";
	cv::VideoWriter output_video(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
		cv::Size(frame_width, frame_height));

	if (!output_video.isOpened()) {
		std::cerr << "Error: Could not open the output video file for writing: " << output_video_path << std::endl;
		return;
	}

	for (int i = 0; i < frame_count; ++i) {
		std::string frame_path = output_folder + "/frame_" + std::to_string(i) + ".png";
		cv::Mat frame = cv::imread(frame_path);
		if (frame.empty()) {
			std::cerr << "Warning: Could not read frame: " << frame_path << std::endl;
			continue;
		}
		output_video.write(frame);
	}

	output_video.release();
	std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double>elapsed = end_time - start_time;

	double avg_frame_process = elapsed.count() / frame_count;
	double frames_per_second = 1.0 / avg_frame_process;

	std::cout << "Glow effect total processing time: " << elapsed.count() << " seconds." << std::endl;
	std::cout << "Total number of frames processed: " << frame_count << " frames." << std::endl;
	std::cout << "Average frame process time: " << avg_frame_process << " seconds/frame." << std::endl;
	std::cout << "Frames per second (fps): " << frames_per_second << " fps." << std::endl;
}

/*
* Preload engine serial pipline
*/
void glow_effect_video_OPT(const char* video_nm) {
	param_KeyLevel = 56;  // Use 56 as the target segmentation value.
	param_KeyScale = 600; // Set scale to 600 to avoid overexposure.
	default_scale = 10;

    // Print OpenCV build information for debugging.
    cv::String info = cv::getBuildInformation();
    std::cout << info << std::endl;

    cv::VideoCapture video;
    if (!video.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY)) {
        std::cerr << "Error: Could not open video file: " << video_nm << std::endl;
        return;
    }

    // Retrieve video parameters.
    int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(video.get(cv::CAP_PROP_FPS));

    std::string output_video_path = "./VideoOutput/processed_video.avi";
    cv::VideoWriter output_video(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps, cv::Size(frame_width, frame_height));

    if (!output_video.isOpened()) {
        std::cerr << "Error: Could not open the output video file for writing: " << output_video_path << std::endl;
        return;
    }

    // Define the TensorRT plan file path for segmentation and initialize the engine ONCE
    std::string planFilePath = "D:/csi4900/TRT-Plans/mobileones4_1.lwfixed.plan";  // user config
    
    // Initialize TensorRT engine once before processing frames
    if (!TRTInference::initializeTRTEngine(planFilePath)) {
        std::cerr << "Failed to initialize TensorRT engine. Exiting." << std::endl;
        return;
    }

    float* h_frame;
    size_t frameSize = frame_width * frame_height * 3;
    cudaMallocHost((void**)&h_frame, frameSize * sizeof(float));

    cv::cuda::GpuMat gpu_frame;
    int frame_count = 0;

    while (true) {
        cv::Mat frame;
        if (!video.read(frame) || frame.empty()) {
            break;  // End of video.
        }
        frame_count++;

        // Keep a copy of the original frame.
        cv::Mat original_frame = frame.clone();

        // Upload and resize the frame for processing.
        gpu_frame.upload(frame);
        cv::cuda::GpuMat resized_gpu_frame;
        cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));

        // Convert the resized GPU image into a Torch tensor.
        torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
        frame_tensor = frame_tensor.to(torch::kFloat);
        // Ensure the tensor has a batch dimension of 1.
        if (frame_tensor.dim() == 3) {
            frame_tensor = frame_tensor.unsqueeze(0);
        }

        // Perform TRT inference using the pre-loaded engine
        std::vector<cv::Mat> grayscale_masks = TRTInference::performSegmentationInference(frame_tensor, 1);
        
        /*for (int i = 0; i < grayscale_masks.size(); i++) {
            cv::imshow("Seg mask", grayscale_masks[i]);
            cv::waitKey(1);
        }*/

        if (!grayscale_masks.empty()) {
            // Resize the segmentation mask to match the original frame.
            cv::Mat grayscale_mask;
            cv::resize(grayscale_masks[0], grayscale_mask, original_frame.size());

			const int EXACT_DETECTION_DELTA = 10;  // Use a tolerance delta, as in the working version.
			cv::Mat processed_mask = grayscale_mask.clone();
			for (int y = 0; y < processed_mask.rows; ++y) {
				for (int x = 0; x < processed_mask.cols; ++x) {
					unsigned char pixel_value = processed_mask.at<uchar>(y, x);
					if (std::abs(pixel_value - param_KeyLevel) <= EXACT_DETECTION_DELTA) {
						processed_mask.at<uchar>(y, x) = param_KeyLevel;
					}
					else {
						processed_mask.at<uchar>(y, x) = 0;
					}
				}
			}

            int tolerance = 0; // Adjust tolerance as needed.
            cv::Mat mask_rgba = threshold_mask_to_rgba(processed_mask, param_KeyLevel, tolerance);

            // Generate the glow effect.
            cv::Mat dst_rgba;
            glow_blow(processed_mask, dst_rgba, param_KeyLevel, 10);

            // Ensure the glow overlay has 4 channels.
            if (dst_rgba.channels() != 4) {
                cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
            }

            // Convert the original frame to BGRA if needed.
            cv::Mat src_rgba;
            if (original_frame.channels() != 4) {
                cv::cvtColor(original_frame, src_rgba, cv::COLOR_BGR2BGRA);
            }
            else {
                src_rgba = original_frame.clone();
            }

            // Prepare the output image.
            cv::Mat final_result;
            final_result.create(src_rgba.size(), CV_8UC4);

            // Blend the original image, the glow overlay, and the mask.
            filter_and_blend(src_rgba.cols, src_rgba.rows, default_scale, param_KeyScale,
                reinterpret_cast<uchar4*>(mask_rgba.data),
                reinterpret_cast<uchar4*>(dst_rgba.data),
                reinterpret_cast<uchar4*>(src_rgba.data),
                reinterpret_cast<uchar4*>(final_result.data));

            // Display and write the processed frame.
            cv::imshow("Processed Frame", final_result);
            int key = cv::waitKey(30);
            if (key == 'q') {
                break;
            }
            output_video.write(final_result);
        }
        else {
            std::cerr << "Warning: No grayscale mask generated for frame " << frame_count << std::endl;
        }
    }

    // Release resources.
    video.release();
    output_video.release();
    cudaFreeHost(h_frame);
    
    // Clean up the TensorRT engine resources
    TRTInference::cleanupTRTEngine();
    
    cv::destroyAllWindows();

    std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;
}


void glow_effect_video_triple_buffer(const char* video_nm, std::string planFilePath) {
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

	cv::Size defaultSize((frame_width > 0) ? frame_width : 640, (frame_height > 0) ? frame_height : 360);

	if (!fs::exists("./VideoOutput/")) {
		if (fs::create_directory("./VideoOutput/"))
			std::cout << "Video Output Directory successfully created." << std::endl;
		else {
			std::cerr << "Failed to create video output folder." << std::endl;
			return;
		}
	}
	else {
		std::cout << "Video Output Directory already exists." << std::endl;
	}

	std::string output_video_path = "./VideoOutput/processed_video.avi";
	cv::VideoWriter output_video(output_video_path,
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps, cv::Size((frame_width > 0) ? frame_width : defaultSize.width,
			(frame_height > 0) ? frame_height : defaultSize.height));
	if (!output_video.isOpened()) {
		std::cerr << "Error: Could not open the output video for writing: " << output_video_path << std::endl;
		return;
	}

	cv::cuda::GpuMat gpu_frame;
	// Containers for 8 frames per iteration.
	std::vector<torch::Tensor> batch_frames;
	std::vector<cv::Mat> original_frames;

	// Use two futures to run segmentation concurrently on sub-batches.
	std::future<std::vector<cv::Mat>> segFuture1, segFuture2;
	bool segFutureValid1 = false, segFutureValid2 = false;

	while (video.isOpened()) {
		batch_frames.clear();
		original_frames.clear();

		// Read a batch of 8 frames.
		for (int i = 0; i < 8; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty())
					break;
				// Duplicate the last valid frame if we run out.
				batch_frames.push_back(batch_frames.back());
				original_frames.push_back(original_frames.back().clone());
				continue;
			}
			if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
				std::cerr << "Warning: Read frame " << i << " is invalid. Using default blank image." << std::endl;
				frame = cv::Mat(defaultSize, CV_8UC3, cv::Scalar(0, 0, 0));
			}
			original_frames.push_back(frame.clone());

			gpu_frame.upload(frame);
			cv::cuda::GpuMat resized_gpu_frame;
			try {
				cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during GPU resize: " << e.what() << ". Using blank image instead." << std::endl;
				cv::Mat blank(384, 384, frame.type(), cv::Scalar(0, 0, 0));
				resized_gpu_frame.upload(blank);
			}
			torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
			frame_tensor = frame_tensor.to(torch::kFloat);
			batch_frames.push_back(frame_tensor);
		}
		if (batch_frames.empty())
			break;
		while (batch_frames.size() < 8) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}

		// Process segmentation results from the previous iteration, if available.
		if (segFutureValid1) {
			std::vector<cv::Mat> grayscale_masks = segFuture1.get();
			segFutureValid1 = false;
			// Process first sub-batch (frames 0 to 3)
			std::vector<cv::Mat> resized_masks_batch;
			for (int i = 0; i < 4; ++i) {
				cv::Mat resized_mask;
				cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
					? defaultSize : original_frames[i].size();
				try {
					cv::resize(grayscale_masks[i], resized_mask, targetSize);
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during segmentation mask resize for sub-batch 1 frame " << i
						<< ": " << e.what() << ". Using blank mask." << std::endl;
					resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
				}
				resized_masks_batch.push_back(resized_mask);
			}

			std::vector<cv::Mat> glow_blow_results(4);
			for (int i = 0; i < 4; ++i) {
				cv::Mat dst_rgba;
				glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
				if (dst_rgba.channels() != 4)
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				glow_blow_results[i] = dst_rgba;
			}
			std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
				resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
			);
			for (int i = 0; i < 4; ++i) {
				cv::Mat final_result;
				mix_images(original_frames[i], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
				if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
					std::cerr << "Warning: Final blended image is empty for sub-batch 1 frame " << i
						<< ". Creating blank output." << std::endl;
					final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
				}
				cv::imshow("Processed Frame", final_result);
				int key = cv::waitKey(30);
				if (key == 'q') {
					video.release();
					cv::destroyAllWindows();
					goto cleanup;
				}
				output_video.write(final_result);
			}
		}
		if (segFutureValid2) {
			std::vector<cv::Mat> grayscale_masks = segFuture2.get();
			segFutureValid2 = false;
			// Process second sub-batch (frames 4 to 7)
			std::vector<cv::Mat> resized_masks_batch;
			for (int i = 4; i < 8; ++i) {
				cv::Mat resized_mask;
				cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
					? defaultSize : original_frames[i].size();
				try {
					cv::resize(grayscale_masks[i - 4], resized_mask, targetSize);
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during segmentation mask resize for sub-batch 2 frame " << i - 4
						<< ": " << e.what() << ". Using blank mask." << std::endl;
					resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
				}
				resized_masks_batch.push_back(resized_mask);
			}
			std::vector<cv::Mat> glow_blow_results(4);
			for (int i = 0; i < 4; ++i) {
				cv::Mat dst_rgba;
				glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
				if (dst_rgba.channels() != 4)
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				glow_blow_results[i] = dst_rgba;
			}
			std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
				resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
			);
			for (int i = 0; i < 4; ++i) {
				cv::Mat final_result;
				mix_images(original_frames[i + 4], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
				if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
					std::cerr << "Warning: Final blended image is empty for sub-batch 2 frame " << i
						<< ". Creating blank output." << std::endl;
					final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
				}
				cv::imshow("Processed Frame", final_result);
				int key = cv::waitKey(30);
				if (key == 'q') {
					video.release();
					cv::destroyAllWindows();
					goto cleanup;
				}
				output_video.write(final_result);
			}
		}

		// Prepare segmentation input for the current batch:
		// Create two sub-batch tensors (each containing 4 frames).
		torch::Tensor sub_batch_tensor1 = torch::stack(
			std::vector<torch::Tensor>(batch_frames.begin(), batch_frames.begin() + 4), 0);
		torch::Tensor sub_batch_tensor2 = torch::stack(
			std::vector<torch::Tensor>(batch_frames.begin() + 4, batch_frames.end()), 0);

		// Launch segmentation concurrently for both sub-batches.
		segFuture1 = std::async(std::launch::async,
			TRTInference::measure_segmentation_trt_performance_mul_concurrent,
			planFilePath, sub_batch_tensor1, 1);
		segFutureValid1 = true;
		segFuture2 = std::async(std::launch::async,
			TRTInference::measure_segmentation_trt_performance_mul_concurrent,
			planFilePath, sub_batch_tensor2, 1);
		segFutureValid2 = true;
	}

	// If any segmentation result is still pending, process it.
	if (segFutureValid1) {
		std::vector<cv::Mat> grayscale_masks = segFuture1.get();
		std::vector<cv::Mat> resized_masks_batch;
		for (int i = 0; i < 4; ++i) {
			cv::Mat resized_mask;
			cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
				? defaultSize : original_frames[i].size();
			try {
				cv::resize(grayscale_masks[i], resized_mask, targetSize);
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during final segmentation mask resize for sub-batch 1 frame " << i
					<< ": " << e.what() << ". Using blank mask." << std::endl;
				resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
			}
			resized_masks_batch.push_back(resized_mask);
		}
		std::vector<cv::Mat> glow_blow_results(4);
		for (int i = 0; i < 4; ++i) {
			cv::Mat dst_rgba;
			glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
			if (dst_rgba.channels() != 4)
				cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
			glow_blow_results[i] = dst_rgba;
		}
		std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
			resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
		);
		for (int i = 0; i < 4; ++i) {
			cv::Mat final_result;
			mix_images(original_frames[i], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
			if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
				std::cerr << "Warning: Final blended image is empty for final sub-batch 1 frame " << i
					<< ". Creating blank output." << std::endl;
				final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
			}
			cv::imshow("Processed Frame", final_result);
			int key = cv::waitKey(30);
			if (key == 'q') {
				video.release();
				cv::destroyAllWindows();
				goto cleanup;
			}
			output_video.write(final_result);
		}
	}
	if (segFutureValid2) {
		std::vector<cv::Mat> grayscale_masks = segFuture2.get();
		std::vector<cv::Mat> resized_masks_batch;
		for (int i = 4; i < 8; ++i) {
			cv::Mat resized_mask;
			cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
				? defaultSize : original_frames[i].size();
			try {
				cv::resize(grayscale_masks[i - 4], resized_mask, targetSize);
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during final segmentation mask resize for sub-batch 2 frame " << i - 4
					<< ": " << e.what() << ". Using blank mask." << std::endl;
				resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
			}
			resized_masks_batch.push_back(resized_mask);
		}
		std::vector<cv::Mat> glow_blow_results(4);
		for (int i = 0; i < 4; ++i) {
			cv::Mat dst_rgba;
			glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
			if (dst_rgba.channels() != 4)
				cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
			glow_blow_results[i] = dst_rgba;
		}
		std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
			resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
		);
		for (int i = 0; i < 4; ++i) {
			cv::Mat final_result;
			mix_images(original_frames[i + 4], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
			if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
				std::cerr << "Warning: Final blended image is empty for final sub-batch 2 frame " << i
					<< ". Creating blank output." << std::endl;
				final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
			}
			cv::imshow("Processed Frame", final_result);
			int key = cv::waitKey(30);
			if (key == 'q') {
				video.release();
				cv::destroyAllWindows();
				goto cleanup;
			}
			output_video.write(final_result);
		}
	}

cleanup:
	video.release();
	output_video.release();
	cv::destroyAllWindows();
	std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Function: glow_effect_video_graph
// Description: CUDA Graph accelerated version of glow_effect_video
////////////////////////////////////////////////////////////////////////////////
void glow_effect_video_graph(const char* video_nm, std::string planFilePath) {
	// Performance measurement
	auto total_start = std::chrono::high_resolution_clock::now();

	cv::String info = cv::getBuildInformation();
	std::cout << info << std::endl;

	// Open video
	cv::VideoCapture video;
	if (!video.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY)) {
		std::cerr << "Error: Could not open video file: " << video_nm << std::endl;
		return;
	}

	int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps = static_cast<int>(video.get(cv::CAP_PROP_FPS));

	cv::Size defaultSize((frame_width > 0) ? frame_width : 640, (frame_height > 0) ? frame_height : 360);

	// Create output directory if needed
	if (!fs::exists("./VideoOutput/")) {
		if (fs::create_directory("./VideoOutput/"))
			std::cout << "Video Output Directory successfully created." << std::endl;
		else {
			std::cerr << "Failed to create video output folder." << std::endl;
			return;
		}
	}
	else {
		std::cout << "Video Output Directory already exists." << std::endl;
	}

	// Create output video writer
	std::string output_video_path = "./VideoOutput/processed_video_graph.avi";
	cv::VideoWriter output_video(output_video_path,
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps, cv::Size((frame_width > 0) ? frame_width : defaultSize.width,
			(frame_height > 0) ? frame_height : defaultSize.height));
	if (!output_video.isOpened()) {
		std::cerr << "Error: Could not open the output video for writing: " << output_video_path << std::endl;
		return;
	}

	// Performance metrics
	int total_frames = 0;
	double segmentation_time = 0.0;
	double post_processing_time = 0.0;

	// Main video processing - follows same structure as original function
	cv::cuda::GpuMat gpu_frame;
	std::vector<torch::Tensor> batch_frames;
	std::vector<cv::Mat> original_frames;

	// Use two futures to run segmentation concurrently on sub-batches
	std::future<std::vector<cv::Mat>> segFuture1, segFuture2;
	bool segFutureValid1 = false, segFutureValid2 = false;

	while (video.isOpened()) {
		batch_frames.clear();
		original_frames.clear();

		// Read a batch of 8 frames - same as original function
		for (int i = 0; i < 8; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty())
					break;
				batch_frames.push_back(batch_frames.back());
				original_frames.push_back(original_frames.back().clone());
				continue;
			}

			total_frames++; // Count frames for performance metrics

			if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
				std::cerr << "Warning: Read frame " << i << " is invalid. Using default blank image." << std::endl;
				frame = cv::Mat(defaultSize, CV_8UC3, cv::Scalar(0, 0, 0));
			}
			original_frames.push_back(frame.clone());

			gpu_frame.upload(frame);
			cv::cuda::GpuMat resized_gpu_frame;
			try {
				cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during GPU resize: " << e.what() << ". Using blank image instead." << std::endl;
				cv::Mat blank(384, 384, frame.type(), cv::Scalar(0, 0, 0));
				resized_gpu_frame.upload(blank);
			}
			torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
			frame_tensor = frame_tensor.to(torch::kFloat);
			batch_frames.push_back(frame_tensor);
		}
		if (batch_frames.empty())
			break;

		while (batch_frames.size() < 8) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}

		// Process segmentation results from the previous iteration, if available
		if (segFutureValid1) {
			auto pp_start = std::chrono::high_resolution_clock::now();

			std::vector<cv::Mat> grayscale_masks = segFuture1.get();
			segFutureValid1 = false;

			// Process first sub-batch (frames 0 to 3) - same as original function
			std::vector<cv::Mat> resized_masks_batch;
			for (int i = 0; i < 4; ++i) {
				cv::Mat resized_mask;
				cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
					? defaultSize : original_frames[i].size();
				try {
					cv::resize(grayscale_masks[i], resized_mask, targetSize);
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during segmentation mask resize for sub-batch 1 frame " << i
						<< ": " << e.what() << ". Using blank mask." << std::endl;
					resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
				}
				resized_masks_batch.push_back(resized_mask);
			}

			std::vector<cv::Mat> glow_blow_results(4);
			for (int i = 0; i < 4; ++i) {
				cv::Mat dst_rgba;
				glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
				if (dst_rgba.channels() != 4)
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				glow_blow_results[i] = dst_rgba;
			}

			std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
				resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
			);

			for (int i = 0; i < 4; ++i) {
				cv::Mat final_result;
				mix_images(original_frames[i], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
				if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
					std::cerr << "Warning: Final blended image is empty for sub-batch 1 frame " << i
						<< ". Creating blank output." << std::endl;
					final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
				}
				cv::imshow("Processed Frame (CUDA Graph)", final_result);
				int key = cv::waitKey(30);
				if (key == 'q') {
					video.release();
					cv::destroyAllWindows();
					goto cleanup;
				}
				output_video.write(final_result);
			}

			auto pp_end = std::chrono::high_resolution_clock::now();
			post_processing_time += std::chrono::duration<double>(pp_end - pp_start).count();
		}

		if (segFutureValid2) {
			auto pp_start = std::chrono::high_resolution_clock::now();

			std::vector<cv::Mat> grayscale_masks = segFuture2.get();
			segFutureValid2 = false;

			// Process second sub-batch (frames 4 to 7) - same as original function
			std::vector<cv::Mat> resized_masks_batch;
			for (int i = 4; i < 8; ++i) {
				cv::Mat resized_mask;
				cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
					? defaultSize : original_frames[i].size();
				try {
					cv::resize(grayscale_masks[i - 4], resized_mask, targetSize);
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during segmentation mask resize for sub-batch 2 frame " << i - 4
						<< ": " << e.what() << ". Using blank mask." << std::endl;
					resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
				}
				resized_masks_batch.push_back(resized_mask);
			}

			std::vector<cv::Mat> glow_blow_results(4);
			for (int i = 0; i < 4; ++i) {
				cv::Mat dst_rgba;
				glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
				if (dst_rgba.channels() != 4)
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				glow_blow_results[i] = dst_rgba;
			}

			std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
				resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
			);

			for (int i = 0; i < 4; ++i) {
				cv::Mat final_result;
				mix_images(original_frames[i + 4], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
				if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
					std::cerr << "Warning: Final blended image is empty for sub-batch 2 frame " << i
						<< ". Creating blank output." << std::endl;
					final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
				}
				cv::imshow("Processed Frame (CUDA Graph)", final_result);
				int key = cv::waitKey(30);
				if (key == 'q') {
					video.release();
					cv::destroyAllWindows();
					goto cleanup;
				}
				output_video.write(final_result);
			}

			auto pp_end = std::chrono::high_resolution_clock::now();
			post_processing_time += std::chrono::duration<double>(pp_end - pp_start).count();
		}

		// Prepare segmentation for the current batch
		// Create two sub-batch tensors (each containing 4 frames)
		torch::Tensor sub_batch_tensor1 = torch::stack(
			std::vector<torch::Tensor>(batch_frames.begin(), batch_frames.begin() + 4), 0);
		torch::Tensor sub_batch_tensor2 = torch::stack(
			std::vector<torch::Tensor>(batch_frames.begin() + 4, batch_frames.end()), 0);

		// Launch segmentation concurrently for both sub-batches - using CUDA Graph version
		auto seg_start = std::chrono::high_resolution_clock::now();

		segFuture1 = std::async(std::launch::async,
			TRTInference::measure_segmentation_trt_performance_mul_concurrent_graph, // Use the graph version
			planFilePath, sub_batch_tensor1, 1);
		segFutureValid1 = true;

		segFuture2 = std::async(std::launch::async,
			TRTInference::measure_segmentation_trt_performance_mul_concurrent_graph, // Use the graph version 
			planFilePath, sub_batch_tensor2, 1);
		segFutureValid2 = true;

		auto seg_end = std::chrono::high_resolution_clock::now();
		segmentation_time += std::chrono::duration<double>(seg_end - seg_start).count();
	}

	// Process any remaining segmentation results - same as original function
	if (segFutureValid1) {
		auto pp_start = std::chrono::high_resolution_clock::now();

		std::vector<cv::Mat> grayscale_masks = segFuture1.get();
		std::vector<cv::Mat> resized_masks_batch;
		for (int i = 0; i < 4; ++i) {
			cv::Mat resized_mask;
			cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
				? defaultSize : original_frames[i].size();
			try {
				cv::resize(grayscale_masks[i], resized_mask, targetSize);
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during final segmentation mask resize for sub-batch 1 frame " << i
					<< ": " << e.what() << ". Using blank mask." << std::endl;
				resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
			}
			resized_masks_batch.push_back(resized_mask);
		}

		std::vector<cv::Mat> glow_blow_results(4);
		for (int i = 0; i < 4; ++i) {
			cv::Mat dst_rgba;
			glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
			if (dst_rgba.channels() != 4)
				cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
			glow_blow_results[i] = dst_rgba;
		}

		std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
			resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
		);

		for (int i = 0; i < 4; ++i) {
			cv::Mat final_result;
			mix_images(original_frames[i], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
			if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
				std::cerr << "Warning: Final blended image is empty for final sub-batch 1 frame " << i
					<< ". Creating blank output." << std::endl;
				final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
			}
			cv::imshow("Processed Frame (CUDA Graph)", final_result);
			int key = cv::waitKey(30);
			if (key == 'q') {
				video.release();
				cv::destroyAllWindows();
				goto cleanup;
			}
			output_video.write(final_result);
		}

		auto pp_end = std::chrono::high_resolution_clock::now();
		post_processing_time += std::chrono::duration<double>(pp_end - pp_start).count();
	}

	if (segFutureValid2) {
		auto pp_start = std::chrono::high_resolution_clock::now();

		std::vector<cv::Mat> grayscale_masks = segFuture2.get();
		std::vector<cv::Mat> resized_masks_batch;
		for (int i = 4; i < 8; ++i) {
			cv::Mat resized_mask;
			cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
				? defaultSize : original_frames[i].size();
			try {
				cv::resize(grayscale_masks[i - 4], resized_mask, targetSize);
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during final segmentation mask resize for sub-batch 2 frame " << i - 4
					<< ": " << e.what() << ". Using blank mask." << std::endl;
				resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
			}
			resized_masks_batch.push_back(resized_mask);
		}

		std::vector<cv::Mat> glow_blow_results(4);
		for (int i = 0; i < 4; ++i) {
			cv::Mat dst_rgba;
			glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
			if (dst_rgba.channels() != 4)
				cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
			glow_blow_results[i] = dst_rgba;
		}

		std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
			resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
		);

		for (int i = 0; i < 4; ++i) {
			cv::Mat final_result;
			mix_images(original_frames[i + 4], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
			if (final_result.empty() || final_result.size().width <= 0 || final_result.size().height <= 0) {
				std::cerr << "Warning: Final blended image is empty for final sub-batch 2 frame " << i
					<< ". Creating blank output." << std::endl;
				final_result = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
			}
			cv::imshow("Processed Frame (CUDA Graph)", final_result);
			int key = cv::waitKey(30);
			if (key == 'q') {
				video.release();
				cv::destroyAllWindows();
				goto cleanup;
			}
			output_video.write(final_result);
		}

		auto pp_end = std::chrono::high_resolution_clock::now();
		post_processing_time += std::chrono::duration<double>(pp_end - pp_start).count();
	}

cleanup:
	auto total_end = std::chrono::high_resolution_clock::now();
	double total_time = std::chrono::duration<double>(total_end - total_start).count();

	video.release();
	output_video.release();
	cv::destroyAllWindows();

	// Output performance metrics
	std::cout << "---------------------------------------------------" << std::endl;
	std::cout << "CUDA Graph Video Processing Performance" << std::endl;
	std::cout << "---------------------------------------------------" << std::endl;
	std::cout << "Total frames processed: " << total_frames << std::endl;
	std::cout << "Total processing time: " << total_time << " seconds" << std::endl;
	if (total_frames > 0) {
		std::cout << "Average time per frame: " << (total_time * 1000.0) / total_frames << " ms" << std::endl;
		std::cout << "Effective frame rate: " << total_frames / total_time << " fps" << std::endl;
	}
	std::cout << "Segmentation time: " << segmentation_time << " seconds ("
		<< (segmentation_time / total_time) * 100.0 << "%)" << std::endl;
	std::cout << "Post-processing time: " << post_processing_time << " seconds ("
		<< (post_processing_time / total_time) * 100.0 << "%)" << std::endl;
	std::cout << "Video processing completed with CUDA Graph acceleration." << std::endl;
	std::cout << "Saved to: " << output_video_path << std::endl;
	std::cout << "---------------------------------------------------" << std::endl;
}

/**
 * @brief Applies a glow effect to video using parallel processing of single-batch TRT model
 *
 * This function processes video frames in parallel using multiple streams and the
 * single-batch TensorRT model. Fixed version addresses glow effect visibility issues.
 *
 * @param video_nm Path to the input video file
 * @param planFilePath Path to the single-batch TensorRT plan file
 */
void glow_effect_video_single_batch_parallel(const char* video_nm, std::string planFilePath) {
	std::cout << "Starting optimized glow effect video processing with reduced latency" << std::endl;

	// *** SET YOUR SPECIFIC TARGET VALUE HERE ***
	param_KeyLevel = 56;  // Change this to your desired segmentation class value
	param_KeyScale = 600; // Ensure we have proper intensity set for blending

	// Set to 0 for exact matching only, or a small value (1-2) for minimal tolerance; here we use 20
	const int EXACT_DETECTION_DELTA = 20;

	// Performance timing
	auto total_start = std::chrono::high_resolution_clock::now();

	// Open video
	cv::VideoCapture video;
	if (!video.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY)) {
		std::cerr << "Error: Could not open video file: " << video_nm << std::endl;
		return;
	}

	// Get video properties
	int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
	int fps = static_cast<int>(video.get(cv::CAP_PROP_FPS));
	cv::Size defaultSize((frame_width > 0) ? frame_width : 640, (frame_height > 0) ? frame_height : 360);

	// Create output directory if needed
	if (!fs::exists("./VideoOutput/")) {
		if (!fs::create_directory("./VideoOutput/")) {
			std::cerr << "Failed to create video output folder." << std::endl;
			return;
		}
	}

	// Create output video writer
	std::string output_video_path = "./VideoOutput/processed_video_optimized.avi";
	cv::VideoWriter output_video(output_video_path,
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps, cv::Size((frame_width > 0) ? frame_width : defaultSize.width,
			(frame_height > 0) ? frame_height : defaultSize.height));

	if (!output_video.isOpened()) {
		std::cerr << "Error: Could not open the output video for writing: " << output_video_path << std::endl;
		return;
	}

	// --- OPTIMIZATION: Load TensorRT engine once at the beginning ---
	TRTGeneration::CustomLogger myLogger;
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(myLogger);

	// Load plan file
	std::ifstream planFile(planFilePath, std::ios::binary);
	if (!planFile.is_open()) {
		std::cerr << "Error: Could not open plan file: " << planFilePath << std::endl;
		return;
	}

	std::vector<char> plan((std::istreambuf_iterator<char>(planFile)),
		std::istreambuf_iterator<char>());
	std::cout << "Loaded TensorRT plan: " << plan.size() / (1024 * 1024) << " MiB" << std::endl;

	// Deserialize the engine
	auto engine_load_start = std::chrono::high_resolution_clock::now();
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan.data(), plan.size());
	auto engine_load_end = std::chrono::high_resolution_clock::now();

	if (!engine) {
		std::cerr << "Error: Failed to deserialize CUDA engine" << std::endl;
		runtime->destroy();
		return;
	}

	double engine_load_time = std::chrono::duration<double>(engine_load_end - engine_load_start).count();
	std::cout << "Engine deserialization time: " << engine_load_time << " seconds" << std::endl;

	// Performance metrics
	int total_frames = 0;
	double segmentation_time = 0.0;
	double post_processing_time = 0.0;
	// Note: The previous mipmap_time variable and triple buffering allocations are removed.

	// Number of parallel streams to use - REDUCED FROM 4 TO 2
	const int NUM_PARALLEL_STREAMS = 2;

	// Create the final result window - ONLY ONE WINDOW
	cv::namedWindow("Final Result", cv::WINDOW_NORMAL);

	// Main processing loop
	cv::cuda::GpuMat gpu_frame;
	std::vector<cv::Mat> original_frames;
	std::vector<torch::Tensor> frame_tensors;
	bool processing = true;
	int batch_count = 0;

	std::cout << "TARGET VALUE: " << param_KeyLevel << " (using delta: " << EXACT_DETECTION_DELTA << ")" << std::endl;

	while (processing) {
		batch_count++;

		// Clear containers for this batch
		original_frames.clear();
		frame_tensors.clear();

		// Read a batch of frames (reduced from 4 to 2)
		for (int i = 0; i < NUM_PARALLEL_STREAMS; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (i == 0) {
					// No more frames to process
					processing = false;
					break;
				}
				// If we have some frames but not enough to fill all streams,
				// duplicate the last valid frame
				if (!original_frames.empty()) {
					frame_tensors.push_back(frame_tensors.back().clone());
					original_frames.push_back(original_frames.back().clone());
				}
				continue;
			}

			total_frames++;

			// Handle invalid frames
			if (frame.empty() || frame.cols <= 0 || frame.rows <= 0) {
				frame = cv::Mat(defaultSize, CV_8UC3, cv::Scalar(0, 0, 0));
			}

			original_frames.push_back(frame.clone());

			try {
				// Preprocess frame for TensorRT - OPTIMIZED PREPROCESSING
				gpu_frame.upload(frame);
				cv::cuda::GpuMat resized_gpu_frame;

				try {
					cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during GPU resize: " << e.what() << ". Using blank image instead." << std::endl;
					cv::Mat blank(384, 384, frame.type(), cv::Scalar(0, 0, 0));
					resized_gpu_frame.upload(blank);
				}

				torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
				frame_tensor = frame_tensor.to(torch::kFloat);

				// Ensure tensor has batch dimension of 1
				if (frame_tensor.dim() == 3) {
					frame_tensor = frame_tensor.unsqueeze(0);
				}

				frame_tensors.push_back(frame_tensor);
			}
			catch (const std::exception& e) {
				std::cerr << "Error preprocessing frame " << i << ": " << e.what() << std::endl;
				// Create a dummy tensor with the right shape
				torch::Tensor dummy_tensor = torch::zeros({ 1, 3, 384, 384 }, torch::kFloat);
				frame_tensors.push_back(dummy_tensor);
			}
		}

		if (!processing || frame_tensors.empty()) {
			break;
		}

		// --- Run Segmentation Inference ---
		auto seg_start = std::chrono::high_resolution_clock::now();
		std::vector<cv::Mat> segmentation_masks;
		try {
			// Use the preloaded engine instead of loading from the plan file
			segmentation_masks = TRTInference::measure_segmentation_trt_performance_single_batch_parallel_preloaded(
				engine, frame_tensors, NUM_PARALLEL_STREAMS);
		}
		catch (const std::exception& e) {
			std::cerr << "Error in segmentation inference: " << e.what() << std::endl;
			// Create empty masks to continue processing
			segmentation_masks.resize(frame_tensors.size());
			for (size_t i = 0; i < frame_tensors.size(); ++i) {
				segmentation_masks[i] = cv::Mat(384, 384, CV_8UC1, cv::Scalar(0));
			}
		}
		auto seg_end = std::chrono::high_resolution_clock::now();
		segmentation_time += std::chrono::duration<double>(seg_end - seg_start).count();

		// --- Post-Processing and Blending ---
		auto pp_start = std::chrono::high_resolution_clock::now();
		// For each frame in the batch, do per-frame mipmapping/blending (no triple buffering)
		for (size_t i = 0; i < original_frames.size() && i < segmentation_masks.size(); ++i) {
			try {
				// Determine target size (use original frame size if valid)
				cv::Size targetSize = (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0)
					? defaultSize : original_frames[i].size();

				// Resize the segmentation mask to match the original frame.
				cv::Mat resized_mask;
				try {
					if (segmentation_masks[i].empty() || segmentation_masks[i].cols <= 0 || segmentation_masks[i].rows <= 0) {
						resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
					}
					else {
						cv::resize(segmentation_masks[i], resized_mask, targetSize);
					}
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during segmentation mask resize: " << e.what() << ". Using blank mask." << std::endl;
					resized_mask = cv::Mat(targetSize, CV_8UC1, cv::Scalar(0));
				}

				// Process the mask: enforce exact matching within tolerance.
				cv::Mat processed_mask = resized_mask.clone();
				for (int y = 0; y < processed_mask.rows; ++y) {
					for (int x = 0; x < processed_mask.cols; ++x) {
						unsigned char pixel_value = processed_mask.at<uchar>(y, x);
						if (std::abs(pixel_value - param_KeyLevel) <= EXACT_DETECTION_DELTA)
							processed_mask.at<uchar>(y, x) = param_KeyLevel;
						else
							processed_mask.at<uchar>(y, x) = 0;
					}
				}

				// Convert the processed mask to an RGBA image.
				int tolerance = 0; // Adjust as needed
				cv::Mat mask_rgba = threshold_mask_to_rgba(processed_mask, param_KeyLevel, tolerance);

				// Generate the glow effect overlay.
				cv::Mat dst_rgba;
				glow_blow(processed_mask, dst_rgba, param_KeyLevel, EXACT_DETECTION_DELTA);
				if (dst_rgba.channels() != 4) {
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				}

				// Convert the original frame to BGRA if needed.
				cv::Mat src_rgba;
				if (original_frames[i].channels() != 4) {
					cv::cvtColor(original_frames[i], src_rgba, cv::COLOR_BGR2BGRA);
				}
				else {
					src_rgba = original_frames[i].clone();
				}

				// Prepare the output image.
				cv::Mat final_result;
				final_result.create(src_rgba.size(), CV_8UC4);

				// Blend using filter_and_blend (same as the OPT pipeline).
				filter_and_blend(src_rgba.cols, src_rgba.rows, default_scale, param_KeyScale,
					reinterpret_cast<uchar4*>(mask_rgba.data),
					reinterpret_cast<uchar4*>(dst_rgba.data),
					reinterpret_cast<uchar4*>(src_rgba.data),
					reinterpret_cast<uchar4*>(final_result.data));

				// Display and write the blended frame.
				cv::imshow("Final Result", final_result);
				int key = cv::waitKey(1);
				if (key == 'q') {
					processing = false;
					break;
				}
				output_video.write(final_result);
			}
			catch (const std::exception& e) {
				std::cerr << "Error in final frame composition for frame " << i << ": " << e.what() << std::endl;
				cv::Mat blank_output = cv::Mat(defaultSize, CV_8UC4, cv::Scalar(0, 0, 0, 255));
				output_video.write(blank_output);
				cv::imshow("Final Result", blank_output);
				cv::waitKey(1);
			}
		}
		auto pp_end = std::chrono::high_resolution_clock::now();
		post_processing_time += std::chrono::duration<double>(pp_end - pp_start).count();

		// Report progress every 10 batches.
		if (batch_count % 10 == 0) {
			std::cout << "Completed batch " << batch_count << " (" << total_frames
				<< " frames, " << (total_frames / (std::chrono::duration<double>(
					std::chrono::high_resolution_clock::now() - total_start).count())) << " fps)" << std::endl;
		}
	}

	// --- Cleanup ---

	// Previously allocated triple-buffering resources have been removed.
	if (engine) {
		engine->destroy();
	}
	if (runtime) {
		runtime->destroy();
	}

	auto total_end = std::chrono::high_resolution_clock::now();
	double total_time = std::chrono::duration<double>(total_end - total_start).count();
	double avg_time_per_batch = total_time / batch_count;
	video.release();
	output_video.release();
	cv::destroyAllWindows();

	std::cout << "---------------------------------------------------" << std::endl;
	std::cout << "Optimized Processing Performance" << std::endl;
	std::cout << "---------------------------------------------------" << std::endl;
	std::cout << "Total frames processed: " << total_frames << std::endl;
	std::cout << "Total processing time: " << total_time << " seconds" << std::endl;
	std::cout << "Engine loading time: " << engine_load_time << " seconds ("
		<< (engine_load_time / total_time * 100.0) << "% of total)" << std::endl;
	std::cout << "Average time per batch: " << avg_time_per_batch << " seconds" << std::endl;
	if (total_frames > 0) {
		std::cout << "Effective frame rate: " << total_frames / total_time << " fps" << std::endl;
		std::cout << "Segmentation time: " << segmentation_time << " seconds ("
			<< (segmentation_time / total_time * 100.0) << "% of total)" << std::endl;
		std::cout << "Post-processing time: " << post_processing_time << " seconds ("
			<< (post_processing_time / total_time * 100.0) << "% of total)" << std::endl;
	}
	std::cout << "Video saved to: " << output_video_path << std::endl;
	std::cout << "---------------------------------------------------" << std::endl;
}
