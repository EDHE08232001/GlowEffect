/*******************************************************************************************************************
 * FILE NAME   : glow_effect.cpp
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
#include "helper_cuda.h"  // For checkCudaErrors
#include <future>         // For std::async, std::future
#include <exception>

namespace fs = std::filesystem;

// Global boolean array indicating button states (for demonstration/testing).
bool button_State[5] = { false, false, false, false, false };

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

	dst_rgba = cv::Mat::zeros(mask.size(), CV_8UC4);
	cv::Vec4b overlay_color = { 199, 170, 255, 255 };
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
		<< (has_target_region ? "found and applied." : "not found.") << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
// Function: apply_mipmap (Synchronous Version)
////////////////////////////////////////////////////////////////////////////////
void apply_mipmap(const cv::Mat& input_gray, cv::Mat& output_image, float scale, int param_KeyLevel) {
	int width = input_gray.cols;
	int height = input_gray.rows;

	if (input_gray.channels() != 1 || input_gray.type() != CV_8UC1) {
		std::cerr << "Error: Input image must be a single-channel grayscale image." << std::endl;
		return;
	}

	uchar4* src_img = new uchar4[width * height];
	uchar4* dst_img = new uchar4[width * height];

	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			unsigned char gray_value = input_gray.at<uchar>(i, j);
			if (gray_value == param_KeyLevel)
				src_img[i * width + j] = { gray_value, gray_value, gray_value, 255 };
			else
				src_img[i * width + j] = { 0, 0, 0, 0 };
		}
	}

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

////////////////////////////////////////////////////////////////////////////////
// Function: glow_effect_video
////////////////////////////////////////////////////////////////////////////////
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

	// Use a default size if video frame size is invalid.
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
	std::vector<torch::Tensor> batch_frames;
	std::vector<cv::Mat> original_frames;

	// Use a future to run segmentation concurrently.
	std::future<std::vector<cv::Mat>> segFuture;
	bool segFutureValid = false;

	while (video.isOpened()) {
		batch_frames.clear();
		original_frames.clear();

		// Read a batch of 4 frames.
		for (int i = 0; i < 4; ++i) {
			cv::Mat frame;
			if (!video.read(frame) || frame.empty()) {
				if (batch_frames.empty())
					break;
				// Duplicate the last valid frame.
				batch_frames.push_back(batch_frames.back());
				original_frames.push_back(original_frames.back().clone());
				continue;
			}
			// If the frame is invalid, replace with a blank image.
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
		while (batch_frames.size() < 4) {
			batch_frames.push_back(batch_frames.back());
			original_frames.push_back(original_frames.back().clone());
		}
		torch::Tensor batch_tensor = torch::stack(batch_frames, 0);
		if (batch_tensor.dim() == 5 && batch_tensor.size(1) == 1)
			batch_tensor = batch_tensor.squeeze(1);
		if (batch_tensor.size(0) < 4) {
			int pad = 4 - batch_tensor.size(0);
			torch::Tensor last_frame = batch_tensor[batch_tensor.size(0) - 1].unsqueeze(0);
			torch::Tensor padTensor = last_frame.repeat({ pad, 1, 1, 1 });
			batch_tensor = torch::cat({ batch_tensor, padTensor }, 0);
		}

		// Process segmentation result from the previous batch, if available.
		if (segFutureValid) {
			std::vector<cv::Mat> grayscale_masks = segFuture.get();
			segFutureValid = false;

			std::vector<cv::Mat> resized_masks_batch;
			for (int i = 0; i < 4; ++i) {
				cv::Mat resized_mask;
				cv::Size targetSize;
				if (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0) {
					std::cerr << "Warning: Original frame " << i << " has invalid size. Using default size." << std::endl;
					targetSize = defaultSize;
				}
				else {
					targetSize = original_frames[i].size();
				}
				try {
					cv::resize(grayscale_masks[i], resized_mask, targetSize);
				}
				catch (cv::Exception& e) {
					std::cerr << "Error during segmentation mask resize: " << e.what() << ". Using blank mask." << std::endl;
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
					std::cerr << "Warning: Final blended image is empty. Creating blank output." << std::endl;
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

		// Launch concurrent segmentation on the current batch.
		segFuture = std::async(std::launch::async,
			TRTInference::measure_segmentation_trt_performance_mul_concurrent,
			planFilePath, batch_tensor, 1);
		segFutureValid = true;
	}

	if (segFutureValid) {
		std::vector<cv::Mat> grayscale_masks = segFuture.get();
		std::vector<cv::Mat> resized_masks_batch;
		for (int i = 0; i < 4; ++i) {
			cv::Mat resized_mask;
			cv::Size targetSize;
			if (original_frames[i].empty() || original_frames[i].cols <= 0 || original_frames[i].rows <= 0) {
				std::cerr << "Warning: Original frame " << i << " has invalid size. Using default size." << std::endl;
				targetSize = defaultSize;
			}
			else {
				targetSize = original_frames[i].size();
			}
			try {
				cv::resize(grayscale_masks[i], resized_mask, targetSize);
			}
			catch (cv::Exception& e) {
				std::cerr << "Error during final segmentation mask resize: " << e.what() << ". Using blank mask." << std::endl;
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
				std::cerr << "Warning: Final blended image is empty. Creating blank output." << std::endl;
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
