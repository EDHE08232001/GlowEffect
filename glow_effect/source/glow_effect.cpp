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

 // Global Variable Array: button_State
bool button_State[5] = { false, false, false, false, false };


#define FILTER_MORPH 1
#define FILTER_GAUSS 1



/**
 * @brief Applies a glowing effect to an image based on a segmentation mask.
 *
 * This function checks if the input segmentation mask contains regions that satisfy a specific condition.
 * If such regions are found, it creates an RGBA image with a pink overlay applied to all pixels.
 *
 * @param mask Input single-channel segmentation mask (CV_8UC1).
 * @param dst_rgba Output RGBA image (CV_8UC4) with the overlay applied.
 * @param param_KeyLevel Threshold level to determine target regions in the mask.
 * @param Delta Tolerance range around param_KeyLevel for determining target regions.
 */
void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta) {
	// Check if the input mask is empty.
	if (mask.empty()) {
		std::cerr << "Error: Segmentation mask is empty." << std::endl;
		return;
	}

	// Ensure the mask is of type CV_8UC1 (single-channel, 8-bit).
	if (mask.type() != CV_8UC1) {
		std::cerr << "Error: Mask is not of type CV_8UC1." << std::endl;
		return;
	}

	// Initialize the output image as an empty RGBA image with a transparent background.
	dst_rgba = cv::Mat::zeros(mask.size(), CV_8UC4);

	// Define the overlay color (pink) in RGBA format.
	cv::Vec4b overlay_color = { 199, 170, 255, 255 }; // B, G, R, A

	// Flag to indicate if a target region exists in the mask.
	bool has_target_region = false;

	// Iterate through each pixel in the mask to find regions satisfying the condition.
	for (int i = 0; i < mask.rows; ++i) {
		for (int j = 0; j < mask.cols; ++j) {
			// Get the pixel value from the mask.
			int mask_pixel = mask.at<uchar>(i, j);

			// Check if the pixel value is within the specified range around param_KeyLevel.
			if (std::abs(mask_pixel - param_KeyLevel) < Delta) {
				has_target_region = true;
				break; // Exit the inner loop if a target region is found.
			}
		}
		if (has_target_region) break; // Exit the outer loop if a target region is found.
	}

	// If a target region is found, fill the entire output image with the overlay color.
	if (has_target_region) {
		for (int i = 0; i < dst_rgba.rows; ++i) {
			for (int j = 0; j < dst_rgba.cols; ++j) {
				dst_rgba.at<cv::Vec4b>(i, j) = overlay_color;
			}
		}
	}

	// Print the result of the operation.
	std::cout << "glow_blow completed. Target region " << (has_target_region ? "found and applied." : "not found.") << std::endl;
}





/**
 * @brief Applies a mipmap-like operation to a grayscale image, producing an RGBA output image.
 *
 * This function filters an input grayscale image based on a specified key level. Pixels matching the key level
 * are preserved with full opacity in an RGBA output image, while others are set to transparent. It also applies
 * a custom filter operation (filter_mipmap) on the processed data.
 *
 * @param input_gray Input single-channel grayscale image (CV_8UC1).
 * @param output_image Output RGBA image (CV_8UC4) with the mipmap operation applied.
 * @param scale Scaling factor used in the mipmap filter operation.
 * @param param_KeyLevel Grayscale value to determine target pixels in the input image.
 */
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
void mix_images(const cv::Mat& src_img, const cv::Mat& dst_rgba, const cv::Mat& mipmap_result, cv::Mat& output_image, float param_KeyScale) {
	// Check if the input images are valid.
	if (src_img.empty() || dst_rgba.empty() || mipmap_result.empty()) {
		std::cerr << "Error: One or more input images are empty." << std::endl;
		return;
	}

	// Ensure all input images have the same dimensions.
	if (src_img.size() != dst_rgba.size() || src_img.size() != mipmap_result.size()) {
		std::cerr << "Error: Images must have the same dimensions." << std::endl;
		return;
	}

	// Ensure src_img is 4-channel RGBA.
	cv::Mat src_rgba;
	if (src_img.channels() != 4) {
		cv::cvtColor(src_img, src_rgba, cv::COLOR_BGR2BGRA);
	}
	else {
		src_rgba = src_img.clone();
	}

	// Ensure dst_rgba is 4-channel RGBA.
	cv::Mat high_lighted_rgba;
	if (dst_rgba.channels() != 4) {
		cv::cvtColor(dst_rgba, high_lighted_rgba, cv::COLOR_BGR2BGRA);
	}
	else {
		high_lighted_rgba = dst_rgba.clone();
	}

	// Ensure mipmap_result is a single-channel grayscale image.
	cv::Mat mipmap_gray;
	if (mipmap_result.channels() != 1) {
		cv::cvtColor(mipmap_result, mipmap_gray, cv::COLOR_BGR2GRAY);
	}
	else {
		mipmap_gray = mipmap_result.clone();
	}

	// Initialize the output image with the source image.
	output_image = src_rgba.clone();

	// Iterate through each pixel and blend based on alpha values.
	for (int i = 0; i < src_rgba.rows; ++i) {
		for (int j = 0; j < src_rgba.cols; ++j) {
			// Retrieve the alpha value from mipmap_result and scale it using param_KeyScale.
			uchar original_alpha = mipmap_gray.at<uchar>(i, j);
			uchar alpha = (original_alpha * static_cast<int>(param_KeyScale)) >> 8;

			// Get the source and highlighted pixels.
			cv::Vec4b src_pixel = src_rgba.at<cv::Vec4b>(i, j);
			cv::Vec4b dst_pixel = high_lighted_rgba.at<cv::Vec4b>(i, j);

			// Blend the pixels using the formula:
			// Output = Src * (255 - alpha) + Dst * alpha
			cv::Vec4b& output_pixel = output_image.at<cv::Vec4b>(i, j);
			for (int k = 0; k < 4; ++k) {
				int temp_pixel = (src_pixel[k] * (255 - alpha) + dst_pixel[k] * alpha) >> 8;
				output_pixel[k] = static_cast<uchar>(std::min(255, std::max(0, temp_pixel)));
			}
		}
	}

	std::cout << "Image mixing completed successfully using scaled alpha." << std::endl;
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
void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask) {
	// Load the source image from the given file path.
	cv::Mat src_img = cv::imread(image_nm);
	if (src_img.empty()) {
		std::cerr << "Error: Could not load source image." << std::endl;
		return;
	}

	// Generate a highlighted effect using the glow_blow function.
	// Uses the global variable `param_KeyLevel` and a delta of 10.
	cv::Mat dst_rgba;
	glow_blow(grayscale_mask, dst_rgba, param_KeyLevel, 10);

	// Save the highlighted effect image for debugging purposes.
	cv::imwrite("./pngOutput/dst_rgba.png", dst_rgba);

	// Ensure the grayscale mask is a single-channel image.
	cv::Mat mipmap_result;

	// Apply the mipmap operation to the grayscale mask.
	// Uses the global variable `default_scale` and `param_KeyLevel`.
	apply_mipmap(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

	// Blend the source image, highlighted image, and mipmap result.
	cv::Mat final_result;
	mix_images(src_img, dst_rgba, mipmap_result, final_result, param_KeyScale);

	// Display the final result in a window.
	cv::imshow("Final Result", final_result);

	// Save the final blended result to a file.
	cv::imwrite("./results/final_result.png", final_result);
}





/**
 * @brief Applies a glow effect to each frame of a video using segmentation and image processing.
 *
 * This function processes a video frame by frame, applying a glow effect based on a segmentation mask.
 * Each frame is processed using TensorRT for inference, mipmap operations, and blending to create the final result.
 * The processed frames are saved and combined into a new video.
 *
 * @param video_nm Path to the input video file.
 */
void glow_effect_video(const char* video_nm) {
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

	// Create output video writer directy.
	std::string output_video_path = "./VideoOutput/processed_video.avi";
	cv::VideoWriter output_video(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, 
		cv::Size(frame_width, frame_height));

	if (!output_video.isOpened()) {
		std::cerr << "Error: COuld not open the ouput video for writing: " << output_video_path << std::endl;
		return;
	}

	/*std::string output_folder = "./VideoOutput";
	std::filesystem::create_directory(output_folder);*/

	// Define the TensorRT plan file path for segmentation.
	std::string planFilePath = "D:/csi4900/TRT-Plans/mobileone_s4.lw.plan";          // user config

	// Pinned memory allocation for CPU-GPU transfer.
	float* h_frame;
	size_t frameSize = frame_width * frame_height * 3;
	cudaMallocHost((void**)&h_frame, frameSize * sizeof(float));

	cv::cuda::GpuMat gpu_frame;

	std::vector<torch::Tensor> batch_frames;

	// cv::Mat src_img, dst_img;
	int frame_count = 0; // Counter for saved frames.

	while (video.isOpened()) {
		// Prepare a batch of frames.
		
		// std::vector<torch::Tensor> batch_frames;
		std::vector<cv::Mat> original_frames;
		batch_frames.clear();

		for (int i = 0; i < 4; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty()) break; // End of video.
				batch_frames.push_back(batch_frames.back()); // Pad batch with last frame.
				original_frames.push_back(original_frames.back().clone());
				continue;
			}

			original_frames.push_back(frame.clone());

			gpu_frame.upload(frame);

			cv::cuda::GpuMat resized_gpu_frame;
			cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));

			// Resize the frame to 384x384 for inference.
			/*cv::Mat resized_img;
			cv::resize(src_img, resized_img, cv::Size(384, 384));*/

			// Save temporary resized image.
			/*std::string temp_img_path = "./temp_video_frame_" + std::to_string(i) + ".png";
			cv::imwrite(temp_img_path, resized_img);*/

			// Convert the image to a tensor.
			torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
			frame_tensor = frame_tensor.to(torch::kFloat);
			batch_frames.push_back(frame_tensor);

			// Remove the temporary file.
			// std::filesystem::remove(temp_img_path);
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

				output_video.write(final_result);

				// Save the processed frame.
				/*std::string frame_output_path = output_folder + "/frame_" + std::to_string(frame_count++) + ".png";
				cv::imwrite(frame_output_path, final_result);*/
			}
		}
		else {
			std::cerr << "Warning: No grayscale mask generated for this batch." << std::endl;
		}
	}

	// Release video resources and close display windows.
	video.release();
	

	// Create the output video from processed frames.
	/*std::string output_video_path = output_folder + "/processed_video.avi";
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
	}*/

	output_video.release();
	cudaFreeHost(h_frame);
	cv::destroyAllWindows();
	std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;
}