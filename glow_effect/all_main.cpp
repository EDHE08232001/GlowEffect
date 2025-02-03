/*******************************************************************************************************************
 * FILE NAME   : all_main.cpp
 * PROJECT NAME: Cuda Learning
 * DESCRIPTION : Top entry point to apply a glow effect using CUDA, TensorRT, and OpenCV.
 * VERSION     : 2022 DEC 11 - Yu Liu - Creation
 *******************************************************************************************************************/

#include "all_common.h"
#include <torch/torch.h>
#include "source/imageprocessingutil.hpp"
#include "source/trtinference.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "glow_effect.hpp"
#include <exception>
#include <filesystem>
#include <thread>
#include <mutex>

 // Forward declaration for the GUI control thread function.
void set_control(void);

// Global state variables
cv::Mat         current_original_img;
cv::Mat         current_grayscale_mask;
std::string     current_image_path;

// Mutex for protecting state updates.
std::mutex state_mutex;

/**
 * @brief Applies the glow effect to the current image using the loaded grayscale mask.
 */
void updateImage() {
	try {
		if (!current_original_img.empty() && !current_grayscale_mask.empty()) {
			glow_effect_image(current_image_path.c_str(), current_grayscale_mask);
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error updating image: " << e.what() << std::endl;
	}
}

/**
 * @brief Callback for the Key Level slider updates.
 *
 * Updates the key level parameter and applies the glow effect.
 *
 * @param newValue The new key level value.
 */
void bar_key_level_cb(int newValue) {
	std::lock_guard<std::mutex> lock(state_mutex);
	param_KeyLevel = newValue;
	std::cout << "Key Level updated to: " << param_KeyLevel << std::endl;
	updateImage();
}

/**
 * @brief Callback for the Key Scale slider updates.
 *
 * Updates the key scale parameter and applies the glow effect.
 *
 * @param newValue The new key scale value.
 */
void bar_key_scale_cb(int newValue) {
	std::lock_guard<std::mutex> lock(state_mutex);
	param_KeyScale = newValue;
	std::cout << "Key Scale updated to: " << param_KeyScale << std::endl;
	updateImage();
}

/**
 * @brief Callback for the Default Scale slider updates.
 *
 * Updates the default scale parameter and applies the glow effect.
 *
 * @param newValue The new default scale value.
 */
void bar_default_scale_cb(int newValue) {
	std::lock_guard<std::mutex> lock(state_mutex);
	default_scale = newValue;
	std::cout << "Default Scale updated to: " << default_scale << std::endl;
	updateImage();
}

/**
 * @brief Main entry point for the glow effect application.
 *
 * Processes a single image, an image directory, or a video file, and applies
 * the glow effect using CUDA, TensorRT, and OpenCV.
 *
 * @return int Exit status.
 */
int main() {
	try {
		auto usage = []() {
			printf("Usage:\n");
			printf("   This program processes single images, directories, or video files.\n");
			printf("Key usage:\n");
			printf("   +: display delay increases by 30ms, max to 300ms\n");
			printf("   -: display delay decreases by 30ms, min to 30ms\n");
			printf("   p: display pauses\n");
			printf("   q: program exits\n");
			printf("   click bottom buttons on the control GUI to switch effect modes\n\n");
		};

		usage();

		// Launch the GUI control thread.
		std::thread guiThread(set_control);

		std::string planFilePath = "D:/csi4900/TRT-Plans/mobileone_s4.edhe.plan";
		std::string userInput;

		printf("Do you want to input a single image, an image directory, or a video file? (single/directory/video): ");
		std::cin >> userInput;

		if (userInput == "single" || userInput == "s") {
			printf("Enter the full path of the input image: ");
			std::cin >> current_image_path;

			current_original_img = cv::imread(current_image_path);
			if (current_original_img.empty()) {
				std::cerr << "Error: Could not load input image." << std::endl;
				return -1;
			}

			cv::Mat resized_img;
			cv::resize(current_original_img, resized_img, cv::Size(384, 384));
			std::string temp_path = "./temp_resized_image.png";
			cv::imwrite(temp_path, resized_img);

			torch::Tensor img_tensor = ImageProcessingUtil::process_img(temp_path, false);
			std::vector<cv::Mat> grayscale_images;

			try {
				grayscale_images = TRTInference::measure_segmentation_trt_performance_mul(planFilePath, img_tensor, 20);
			}
			catch (const std::exception& e) {
				std::cerr << "Error in segmentation inference: " << e.what() << std::endl;
				return -1;
			}

			if (!grayscale_images.empty()) {
				current_grayscale_mask = grayscale_images[0];
				cv::resize(current_grayscale_mask, current_grayscale_mask, current_original_img.size());
				updateImage();
			}

			std::filesystem::remove(temp_path);

			// Wait for user to exit.
			while (true) {
				char key = cv::waitKey(30);
				if (key == 'q')
					break;
			}
		}
		else if (userInput == "directory" || userInput == "d") {
			printf("Enter the full path of the image directory: ");
			std::cin >> userInput;

			std::vector<std::string> img_paths;
			try {
				for (const auto& entry : std::filesystem::directory_iterator(userInput)) {
					if (entry.is_regular_file() &&
						(entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
						img_paths.push_back(entry.path().string());
					}
				}
			}
			catch (const std::exception& e) {
				std::cerr << "Error accessing directory: " << e.what() << std::endl;
				return -1;
			}

			std::vector<cv::Mat> original_images;
			std::vector<std::string> resized_image_paths;

			for (size_t i = 0; i < img_paths.size(); ++i) {
				cv::Mat img = cv::imread(img_paths[i]);
				if (img.empty()) {
					std::cerr << "Error: Could not load image at path: " << img_paths[i] << std::endl;
					continue;
				}
				original_images.push_back(img);

				cv::Mat resized_img;
				cv::resize(img, resized_img, cv::Size(384, 384));

				std::string temp_path = "./temp_resized_image_" + std::to_string(i) + ".png";
				cv::imwrite(temp_path, resized_img);
				resized_image_paths.push_back(temp_path);
			}

			torch::Tensor img_tensor_batch;
			try {
				img_tensor_batch = ImageProcessingUtil::process_img_batch(resized_image_paths, false);
			}
			catch (const std::exception& e) {
				std::cerr << "Error processing batch images: " << e.what() << std::endl;
				return -1;
			}

			std::vector<cv::Mat> grayscale_images;
			try {
				grayscale_images = TRTInference::measure_segmentation_trt_performance_mul(planFilePath, img_tensor_batch, 20);
			}
			catch (const std::exception& e) {
				std::cerr << "Error in segmentation inference: " << e.what() << std::endl;
				return -1;
			}

			size_t current_index = 0;
			while (true) {
				current_image_path = img_paths[current_index];
				current_original_img = original_images[current_index];

				if (!grayscale_images.empty() && current_index < grayscale_images.size()) {
					current_grayscale_mask = grayscale_images[current_index];
					cv::resize(current_grayscale_mask, current_grayscale_mask, current_original_img.size());
					updateImage();
				}

				char key = cv::waitKey(30);
				if (key == 'q')
					break;
				if (key == 13) { // Enter key
					current_index = (current_index + 1) % img_paths.size();
				}
			}

			for (const auto& temp_path : resized_image_paths) {
				std::filesystem::remove(temp_path);
			}
		}
		else if (userInput == "video" || userInput == "v") {
			std::string videoPath;
			std::string videoInputOption;

			printf("Do you want to use default input video path? (y/n): ");
			std::cin >> videoInputOption;

			// Choose default path or a customized path.
			if (videoInputOption == "y") {
				videoPath = "D:/csi4900/VideoInputs/racing_cars.sd.mp4"; // Use your own default path.
			}
			else {
				printf("Enter the full path of the video file: ");
				std::cin >> videoPath;
			}

			if (videoPath.empty()) {
				std::cout << "videoPath cannot be empty" << std::endl;
				return 1;
			}

			if (!std::filesystem::exists(videoPath)) {
				std::cout << "The specified file path does not exist." << std::endl;
				return 1;
			}

			if (!std::filesystem::is_regular_file(videoPath)) {
				std::cout << "The specified path is not a valid file." << std::endl;
				return 1;
			}

			try {
				glow_effect_video(videoPath.c_str(), planFilePath);
			}
			catch (const std::exception& e) {
				std::cerr << "Error processing video: " << e.what() << std::endl;
				return -1;
			}
		}
		else {
			printf("Invalid input. Terminating the program.\n");
			return 0;
		}

		// Wait for the GUI thread to finish before exiting.
		guiThread.join();
	}
	catch (const std::exception& e) {
		std::cerr << "Unexpected error occurred: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}
