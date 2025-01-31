/*******************************************************************************************************************
 * FILE NAME   :    glow_effect.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    This file implements various glow-related effects such as "blow" highlighting,
 *                  mipmapping, and alpha blending to create bloom/glow effects on images and video frames.
 *                  It integrates CUDA kernels, OpenCV, and TensorRT (for segmentation in the video pipeline).
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

 /**
  * @brief Global boolean array indicating button states in GUI (for demonstration/testing).
  *
  * This array is used internally to simulate or track various button states that may
  * control different aspects of the glow or mipmap effects.
  */
bool button_State[5] = { false, false, false, false, false };

// Macros controlling which filters are active (used in other modules, e.g. dilate/erode).
#define FILTER_MORPH 1
#define FILTER_GAUSS 1

/**
 * -------------------------------------------------------------------------------------------------
 * glow_blow
 * -------------------------------------------------------------------------------------------------
 *
 * @details
 * Applies a simple "blow" or highlight effect to an output RGBA image, based on whether the
 * input grayscale mask contains pixels within a specified range around `param_KeyLevel`.
 *
 * - If the mask contains any pixel such that |mask_pixel - param_KeyLevel| < Delta, then the
 *   entire output image (`dst_rgba`) is filled with a pink overlay (fully opaque).
 * - If no such pixel is found, `dst_rgba` remains black/transparent.
 *
 * **Core Steps:**
 * 1. Verify the mask is valid and single-channel.
 * 2. Search for pixels that meet the highlight condition (`param_KeyLevel ± Delta`).
 * 3. If found, fill the output with an RGBA overlay color; otherwise leave it transparent.
 *
 * **Usage Example:**
 * @code
 * cv::Mat mask = cv::imread("segmentation_mask.png", cv::IMREAD_GRAYSCALE);
 * cv::Mat dst;
 * glow_blow(mask, dst, 128, 10);
 * // If any pixel in 'mask' is within [118..138], dst is fully pink/opaque
 * // otherwise dst is transparent.
 * @endcode
 *
 * @param[in]  mask          A single-channel (CV_8UC1) mask.
 * @param[out] dst_rgba      Output RGBA image (CV_8UC4); created/overwritten here.
 * @param[in]  param_KeyLevel Key level parameter controlling the highlight trigger.
 * @param[in]  Delta         The tolerance range around `param_KeyLevel`.
 *
 * @pre  `mask.type() == CV_8UC1` (grayscale) and `mask` is not empty.
 * @post `dst_rgba` is either transparent (no region found) or fully pink (region found).
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
	std::cout << "glow_blow completed. Target region "
		<< (has_target_region ? "found and applied." : "not found.") << std::endl;
}

/**
 * -------------------------------------------------------------------------------------------------
 * apply_mipmap
 * -------------------------------------------------------------------------------------------------
 *
 * @details
 * This function takes a single-channel grayscale image (`input_gray`), creates an RGBA representation
 * where only pixels equal to `param_KeyLevel` are opaque, and then calls `filter_mipmap` to perform
 * a CUDA-based downscale or blur. The result is stored in `output_image`.
 *
 * **Core Steps:**
 * 1. Validate input image (grayscale, non-empty).
 * 2. Convert grayscale to an RGBA buffer (`src_img`) with selective opacity (only `param_KeyLevel` is opaque).
 * 3. Invoke `filter_mipmap` on the CUDA device.
 * 4. Reconstruct the RGBA output in `output_image`.
 *
 * **Usage Example:**
 * @code
 * cv::Mat gray = cv::imread("gray_mask.png", cv::IMREAD_GRAYSCALE);
 * cv::Mat out;
 * apply_mipmap(gray, out, 0.5f, 128);
 * // 'out' is an RGBA image with filtered result from the CUDA mipmap.
 * @endcode
 *
 * @param[in]  input_gray     The source single-channel (CV_8UC1) grayscale image.
 * @param[out] output_image   The destination RGBA (CV_8UC4) image after mipmap filtering.
 * @param[in]  scale          The scale factor used by `filter_mipmap` (e.g., 0.5f).
 * @param[in]  param_KeyLevel A grayscale value determining which pixels become opaque before filtering.
 *
 * @pre  `input_gray` is valid, non-empty, and has `1` channel (8-bit).
 * @post `output_image` is generated and saved (for debugging) as "output_image_after_mipmap.png".
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

	// Convert uchar4 array to an OpenCV RGBA image for debugging purposes.
	cv::Mat uchar4_image_before(height, width, CV_8UC4);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar4 value = src_img[i * width + j];
			uchar4_image_before.at<cv::Vec4b>(i, j) = cv::Vec4b(value.x, value.y, value.z, value.w);
		}
	}
	cv::imwrite("./pngOutput/converted_uchar4_before_mipmap.png", uchar4_image_before);
	std::cout << "Converted uchar4 image saved as converted_uchar4_before_mipmap.png" << std::endl;

	// Apply the mipmap filter operation (CUDA).
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
 * -------------------------------------------------------------------------------------------------
 * mix_images
 * -------------------------------------------------------------------------------------------------
 *
 * @details
 * Blends two images (`img1` and `img2`) using a grayscale mask (`mask`) and a global alpha factor (`alpha`).
 * Internally, this function:
 *
 * 1. Ensures both images are RGBA (4-channel). If not, converts them.
 * 2. Interprets the mask's pixel values as alpha multipliers (optionally scaled by `param_KeyScale` or `alpha`).
 * 3. Performs per-pixel blending:
 *    \f$ \text{output} = \text{img1} \times (255 - \alpha) + \text{img2} \times \alpha \f$
 *
 * **Usage Example:**
 * @code
 * cv::Mat src = cv::imread("source.png", cv::IMREAD_COLOR); // BGR
 * cv::Mat highlight = cv::imread("highlight.png", cv::IMREAD_COLOR);
 * cv::Mat mask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
 * cv::Mat output;
 * mix_images(src, highlight, mask, output, 0.5f);
 * // 'output' is now a blended RGBA image.
 * @endcode
 *
 * @param[in]  img1   The first source image (can be 3 or 4 channels).
 * @param[in]  img2   The second source image (can be 3 or 4 channels).
 * @param[in]  mask   A single-channel (CV_8UC1) mask that influences blending.
 * @param[out] dst    The final blended image (CV_8UC4).
 * @param[in]  alpha  A floating-point factor scaling the mask-based alpha.
 *
 * @pre  `img1.size() == img2.size() == mask.size()`, and none are empty.
 * @post `dst` (RGBA) contains the blended result.
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
 * -------------------------------------------------------------------------------------------------
 * glow_effect_image
 * -------------------------------------------------------------------------------------------------
 *
 * @details
 * Provides a high-level pipeline for applying glow effects to a single image:
 * 1. Loads the source image from `image_nm`.
 * 2. Generates a highlight overlay using `glow_blow`.
 * 3. Performs a mipmap transformation on the input mask (`apply_mipmap`).
 * 4. Blends the original image, highlight overlay, and mipmap result (`mix_images`).
 * 5. Displays and saves the final result as "final_result.png".
 *
 * **Usage Example:**
 * @code
 * cv::Mat mask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
 * glow_effect_image("input.png", mask);
 * @endcode
 *
 * @param[in] image_nm        Path to the source image file.
 * @param[in] grayscale_mask  Single-channel mask image guiding the glow effect.
 *
 * @pre  The file at `image_nm` should be a valid image readable by OpenCV.
 * @pre  `grayscale_mask.size()` should match the loaded image’s size for best alignment.
 * @post Final result is shown in a window and saved to `./results/final_result.png`.
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
 * -------------------------------------------------------------------------------------------------
 * glow_effect_video
 * -------------------------------------------------------------------------------------------------
 *
 * @details
 * Processes an entire video, applying a glow effect to each frame:
 *
 * 1. Opens the specified video file.
 * 2. Reads frames in batches of four, performing TensorRT inference to generate segmentation masks.
 * 3. For each mask in the batch, runs `glow_blow`, `apply_mipmap`, and `mix_images` to create the final processed frame.
 * 4. Saves each processed frame to disk.
 * 5. Compiles all processed frames into `processed_video.avi`.
 *
 * **Usage Example:**
 * @code
 * glow_effect_video("input_video.mp4");
 * @endcode
 *
 * @param[in] video_nm			Path to the input video file.
 * @param[in] planFilePath		Path to trt plan
 *
 * @pre  `video_nm` must point to a valid video file (supported by OpenCV).
 * @post Outputs processed frames into `./VideoOutput` and saves a compiled video `processed_video.avi`.
 */
void glow_effect_video(const char* video_nm) {
	// For debugging and informational purposes, print out the OpenCV build info.
	cv::String info = cv::getBuildInformation();
	std::cout << info << std::endl;

	// Create a VideoCapture object to handle reading frames from the specified video file.
	cv::VideoCapture video;
	bool pause = false;

	// Attempt to open the video using any available backend. If this fails, print an error and return.
	if (!video.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY)) {
		std::cerr << "Error: Could not open video file: " << video_nm << std::endl;
		return;
	}

	// Retrieve various properties of the video, such as its frame width, height, and frames per second (FPS).
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
		std::vector<cv::Mat> original_frames;
		batch_frames.clear();

		// Attempt to read four frames from the video source.
		for (int i = 0; i < 4; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty()) break; // End of video.
				batch_frames.push_back(batch_frames.back()); // Pad batch with last frame.
				batch_frames.push_back(batch_frames.back());
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

			// Add the tensor to our batch.
			batch_frames.push_back(frame_tensor);


			// Remove the temporary file.
			// std::filesystem::remove(temp_img_path);
		}

		// If we didn't read any frames at all (i.e., the video is finished), break the main loop.
		if (batch_frames.empty()) break;

		// If we have fewer than four frames in this batch (e.g., end of video), 
		// duplicate the last frame/tensor to pad up to a full batch.
		while (batch_frames.size() < 4) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}

		// Combine the 4 individual frame tensors into a single batched tensor of shape [4, 3, 384, 384].
		torch::Tensor batch_tensor = torch::stack(batch_frames, 0);

		// Perform TensorRT inference on the batched tensor to obtain segmentation masks for each frame.
		// This function returns a vector of single-channel cv::Mat objects (grayscale masks).
		std::vector<cv::Mat> grayscale_masks = TRTInference::measure_segmentation_trt_performance_mul(planFilePath, batch_tensor, 1);

		// If we successfully retrieved masks, process them.
		if (!grayscale_masks.empty()) {
			// Each mask in `grayscale_masks` corresponds to a frame in the current batch.
			for (int i = 0; i < 4; ++i) {
				cv::Mat grayscale_mask;
				// Resize the returned mask to match the original frame's size (before we had resized to 384x384).
				cv::resize(grayscale_masks[i], grayscale_mask, original_frames[i].size());

				// Use the grayscale mask to produce a "blow" or highlight effect.
				// glow_blow will fill an RGBA image (`dst_rgba`) if the mask contains pixels 
				// near our key-level threshold, otherwise it remains transparent.
				cv::Mat dst_rgba;
				glow_blow(grayscale_mask, dst_rgba, param_KeyLevel, 10);

				// Ensure we are working in RGBA format, which is needed for subsequent blending.
				if (dst_rgba.channels() != 4) {
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				}

				// Perform a mipmap-like operation on the mask to generate an additional overlay or effect.
				cv::Mat mipmap_result;
				apply_mipmap(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

				// Combine the original frame, the blow effect, and the mipmap result into a final RGBA image.
				// `mix_images` handles per-pixel alpha blending logic.
				cv::Mat final_result;
				mix_images(original_frames[i], dst_rgba, mipmap_result, final_result, param_KeyScale);

				// Show the processed frame in a window for real-time preview.
				// Pressing 'q' will stop processing immediately.
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
			// If segmentation fails or no masks are generated, print a warning and continue.
			std::cerr << "Warning: No grayscale mask generated for this batch." << std::endl;
		}
	}

	// When we reach here, we've either processed all frames or encountered a break condition.



	// Create the output video from processed frames.
	/*std::string output_video_path = output_folder + "/processed_video.avi";
	cv::VideoWriter output_video(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
		cv::Size(frame_width, frame_height));

	// If the output video file fails to open, log an error and return.
	if (!output_video.isOpened()) {
		std::cerr << "Error: Could not open the output video file for writing: "
			<< output_video_path << std::endl;
		return;
	}

	// Loop through all the frames we saved as PNG files in the output folder.
	for (int i = 0; i < frame_count; ++i) {
		std::string frame_path = output_folder + "/frame_" + std::to_string(i) + ".png";
		cv::Mat frame = cv::imread(frame_path);
		if (frame.empty()) {
			std::cerr << "Warning: Could not read frame: " << frame_path << std::endl;
			continue;
		}
		// Write this frame into the final video file.
		output_video.write(frame);
	}*/


	video.release();
	output_video.release();
	cudaFreeHost(h_frame);
	cv::destroyAllWindows();

	// Indicate that processing has completed successfully.
	std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;
}