/*******************************************************************************************************************
 * FILE NAME   : glow_effect.cpp
 * PROJECT NAME: Cuda Learning
 * DESCRIPTION : Implements various glow-related effects such as "blow" highlighting,
 *               mipmapping, and alpha blending to create bloom/glow effects on images and video frames.
 *               Integrates CUDA kernels, OpenCV, and TensorRT (for segmentation in the video pipeline).
 *               Uses triple buffering to accelerate asynchronous mipmap filtering.
 * VERSION     : Updated 2025 FEB 04
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

////////////////////////////////////////////////////////////////////////////////
// Helper Function: convert_mask_to_rgba_buffer
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Converts a grayscale mask to an RGBA buffer (uchar4 array) based on a key level.
 *
 * Only pixels equal to param_KeyLevel become opaque (alpha 255); all others are set transparent.
 *
 * @param mask           The input grayscale mask (CV_8UC1).
 * @param dst            Preallocated destination buffer (array of uchar4) of size (frame_width * frame_height).
 * @param frame_width    The width of the image.
 * @param frame_height   The height of the image.
 * @param param_KeyLevel The grayscale value to preserve (others become transparent).
 */
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
/**
 * @brief Processes a batch of grayscale masks through an asynchronous mipmap filter using triple buffering.
 *
 * This function converts each resized grayscale mask into an RGBA buffer, launches asynchronous
 * mipmap filtering on each using triple buffering, and collects the processed output images.
 *
 * This version uses pinned (page-locked) memory for the triple buffers to accelerate host-device transfers.
 *
 * @param resized_masks  The input vector of resized grayscale masks (CV_8UC1) for each frame.
 * @param frame_width    The width of the frames.
 * @param frame_height   The height of the frames.
 * @param default_scale  The scale factor used by the mipmap filter.
 * @param param_KeyLevel The key level used to determine opacity in the mask.
 * @return A vector of cv::Mat objects (CV_8UC4) containing the filtered mipmap output for each frame.
 */
std::vector<cv::Mat> triple_buffered_mipmap_pipeline(const std::vector<cv::Mat>& resized_masks,
	int frame_width, int frame_height,
	float default_scale, int param_KeyLevel) {
	int N = resized_masks.size();
	const int numBuffers = 3;
	std::vector<cv::Mat> outputImages(N);

	// Allocate triple buffers for source and destination using pinned memory.
	std::vector<uchar4*> tripleSrc(numBuffers, nullptr);
	std::vector<uchar4*> tripleDst(numBuffers, nullptr);
	std::vector<cudaStream_t> mipmapStreams(numBuffers);
	std::vector<cudaEvent_t> mipmapDone(numBuffers);

	for (int i = 0; i < numBuffers; ++i) {
		checkCudaErrors(cudaStreamCreate(&mipmapStreams[i]));
		checkCudaErrors(cudaEventCreate(&mipmapDone[i]));
		// Allocate pinned host memory for each buffer.
		checkCudaErrors(cudaMallocHost((void**)&tripleSrc[i], frame_width * frame_height * sizeof(uchar4)));
		checkCudaErrors(cudaMallocHost((void**)&tripleDst[i], frame_width * frame_height * sizeof(uchar4)));
	}

	// Pipeline loop: run for (N + 2) iterations to flush the pipeline.
	for (int i = 0; i < N + 2; ++i) {
		// Stage 1: Launch asynchronous filtering for frame i if available.
		if (i < N) {
			int bufIdx = i % numBuffers;
			// Convert the resized grayscale mask to an RGBA buffer and store in tripleSrc.
			convert_mask_to_rgba_buffer(resized_masks[i], tripleSrc[bufIdx], frame_width, frame_height, param_KeyLevel);
			// Launch asynchronous mipmap filtering using the modified apply_mipmap_async.
			// The output is captured into a temporary cv::Mat.
			cv::Mat tempOutput;
			apply_mipmap_async(resized_masks[i], tempOutput, default_scale, param_KeyLevel, mipmapStreams[bufIdx]);
			// Copy the resulting tempOutput into our tripleDst buffer.
			for (int r = 0; r < frame_height; ++r) {
				for (int c = 0; c < frame_width; ++c) {
					cv::Vec4b pixel = tempOutput.at<cv::Vec4b>(r, c);
					tripleDst[bufIdx][r * frame_width + c] = { pixel[0], pixel[1], pixel[2], pixel[3] };
				}
			}
			// Record an event for this buffer.
			checkCudaErrors(cudaEventRecord(mipmapDone[bufIdx], mipmapStreams[bufIdx]));
		}

		// Stage 2: For frame (i - 2), check if the asynchronous filtering is complete.
		if (i - 2 >= 0 && (i - 2) < N) {
			int bufIdx = (i - 2) % numBuffers;
			cudaError_t query = cudaEventQuery(mipmapDone[bufIdx]);
			if (query == cudaSuccess) {
				// Copy the contents of tripleDst into a cv::Mat.
				cv::Mat mipmapResult(frame_height, frame_width, CV_8UC4);
				for (int r = 0; r < frame_height; ++r) {
					for (int c = 0; c < frame_width; ++c) {
						uchar4 val = tripleDst[bufIdx][r * frame_width + c];
						mipmapResult.at<cv::Vec4b>(r, c) = cv::Vec4b(val.x, val.y, val.z, val.w);
					}
				}
				outputImages[i - 2] = mipmapResult;
			}
			else if (query != cudaErrorNotReady) {
				checkCudaErrors(query);
			}
		}
	}

	// Cleanup: free pinned memory and destroy streams and events.
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

	// Check if any pixel is within the tolerance Delta of param_KeyLevel.
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
		// Fill the entire image with the overlay color.
		for (int i = 0; i < dst_rgba.rows; ++i) {
			for (int j = 0; j < dst_rgba.cols; ++j) {
				dst_rgba.at<cv::Vec4b>(i, j) = overlay_color;
			}
		}
	}

	std::cout << "glow_blow completed. Target region "
		<< (has_target_region ? "found and applied." : "not found.") << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Function: apply_mipmap (Synchronous Version)
////////////////////////////////////////////////////////////////////////////////
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

	// Validate input image.
	if (input_gray.channels() != 1 || input_gray.type() != CV_8UC1) {
		std::cerr << "Error: Input image must be a single-channel grayscale image." << std::endl;
		return;
	}

	// Allocate host memory for RGBA buffers.
	uchar4* src_img = new uchar4[width * height];
	uchar4* dst_img = new uchar4[width * height];

	// Convert grayscale image to an RGBA buffer.
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

	// Convert the output buffer back into an OpenCV image.
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

////////////////////////////////////////////////////////////////////////////////
// Function: apply_mipmap_async (Asynchronous Version)
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Asynchronously applies a CUDA-based mipmap filter to a grayscale image and outputs an RGBA image.
 *
 * Converts the input grayscale image to an RGBA buffer (keeping only pixels equal to param_KeyLevel as opaque),
 * then uses the asynchronous filter_mipmap_async function to apply the mipmap filter on the provided CUDA stream.
 * The function returns immediately so that the caller can overlap operations.
 *
 * @param input_gray    The source single-channel (CV_8UC1) grayscale image.
 * @param output_image  The destination RGBA image (CV_8UC4) after mipmap filtering.
 * @param scale         The scale factor used by the mipmap filter.
 * @param param_KeyLevel Grayscale value determining which pixels become opaque.
 * @param stream        The CUDA stream on which to perform asynchronous mipmap filtering.
 */
void apply_mipmap_async(const cv::Mat& input_gray, cv::Mat& output_image, float scale, int param_KeyLevel, cudaStream_t stream) {
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

	// Launch the asynchronous mipmap filter on the provided stream.
	// This call uses your low-level asynchronous function filter_mipmap_async.
	filter_mipmap_async(width, height, scale, src_img, dst_img, stream);

	// Do NOT synchronize here; the caller is responsible for synchronization.
	// Convert the output buffer back into an OpenCV image.
	output_image.create(height, width, CV_8UC4);
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			uchar4 value = dst_img[i * width + j];
			output_image.at<cv::Vec4b>(i, j) = cv::Vec4b(value.x, value.y, value.z, value.w);
		}
	}

	std::cout << "apply_mipmap_async: Launched asynchronous mipmap filtering on provided stream." << std::endl;

	delete[] src_img;
	delete[] dst_img;
}

////////////////////////////////////////////////////////////////////////////////
// Function: mix_images
////////////////////////////////////////////////////////////////////////////////
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

	// Blend each pixel based on the alpha value computed from the grayscale mask.
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

////////////////////////////////////////////////////////////////////////////////
// Function: glow_effect_image
////////////////////////////////////////////////////////////////////////////////
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
	// For single-image processing, use synchronous mipmap filtering.
	apply_mipmap(grayscale_mask, mipmap_result, static_cast<float>(default_scale), param_KeyLevel);

	cv::Mat final_result;
	mix_images(src_img, dst_rgba, mipmap_result, final_result, param_KeyScale);

	cv::imshow("Final Result", final_result);
	cv::waitKey(0);
}

////////////////////////////////////////////////////////////////////////////////
// Function: glow_effect_video
////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Applies a glow effect to a video file.
 *
 * Processes each frame of the input video by performing TensorRT segmentation, applying a glow highlight,
 * asynchronous mipmap filtering using triple buffering, and blending. The processed frames are written to an output video file.
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

	// Create output directory if it does not exist.
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

	cv::cuda::GpuMat gpu_frame;
	std::vector<torch::Tensor> batch_frames;

	// Main video processing loop: process frames in batches of 4.
	while (video.isOpened()) {
		std::vector<cv::Mat> original_frames;
		batch_frames.clear();
		std::vector<cv::Mat> resized_masks;  // Will store resized grayscale masks.

		// Read 4 frames for batch processing.
		for (int i = 0; i < 4; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty())
					break;
				// If fewer than 4 frames are read, duplicate the last frame.
				batch_frames.push_back(batch_frames.back());
				original_frames.push_back(original_frames.back().clone());
				continue;
			}
			original_frames.push_back(frame.clone());

			// Upload frame to GPU and resize for segmentation.
			gpu_frame.upload(frame);
			cv::cuda::GpuMat resized_gpu_frame;
			cv::cuda::resize(gpu_frame, resized_gpu_frame, cv::Size(384, 384));

			torch::Tensor frame_tensor = ImageProcessingUtil::process_img(resized_gpu_frame, false);
			frame_tensor = frame_tensor.to(torch::kFloat);
			batch_frames.push_back(frame_tensor);
		}
		if (batch_frames.empty())
			break;
		// Ensure we have 4 frames by duplicating the last frame if necessary.
		while (batch_frames.size() < 4) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}
		torch::Tensor batch_tensor = torch::stack(batch_frames, 0);
		// Perform segmentation using TensorRT.
		std::vector<cv::Mat> grayscale_masks = TRTInference::measure_segmentation_trt_performance_mul(planFilePath, batch_tensor, 1);

		if (!grayscale_masks.empty()) {
			// Resize each grayscale mask to match the corresponding original frame.
			std::vector<cv::Mat> resized_masks_batch;
			for (int i = 0; i < 4; ++i) {
				cv::Mat resized_mask;
				cv::resize(grayscale_masks[i], resized_mask, original_frames[i].size());
				resized_masks_batch.push_back(resized_mask);
			}

			// Generate glow highlights using glow_blow for each frame.
			std::vector<cv::Mat> glow_blow_results(4);
			for (int i = 0; i < 4; ++i) {
				cv::Mat dst_rgba;
				glow_blow(resized_masks_batch[i], dst_rgba, param_KeyLevel, 10);
				if (dst_rgba.channels() != 4) {
					cv::cvtColor(dst_rgba, dst_rgba, cv::COLOR_BGR2RGBA);
				}
				glow_blow_results[i] = dst_rgba;
			}

			// Process the resized masks with the triple-buffered mipmap pipeline.
			std::vector<cv::Mat> mipmap_results = triple_buffered_mipmap_pipeline(
				resized_masks_batch, frame_width, frame_height, static_cast<float>(default_scale), param_KeyLevel
			);

			// Blend each frame using the glow highlight and the processed mipmap result.
			for (int i = 0; i < 4; ++i) {
				cv::Mat final_result;
				mix_images(original_frames[i], glow_blow_results[i], mipmap_results[i], final_result, param_KeyScale);
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
		else {
			std::cerr << "Warning: No grayscale mask generated for this batch." << std::endl;
		}
	}

cleanup:
	video.release();
	output_video.release();
	cv::destroyAllWindows();

	std::cout << "Video processing completed. Saved to: " << output_video_path << std::endl;
}
