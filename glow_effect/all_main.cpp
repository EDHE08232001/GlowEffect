/*******************************************************************************************************************
 * FILE NAME   :    all_main.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    Top entry point to apply a glow effect using CUDA, TensorRT, and OpenCV.
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 DEC 11      Yu Liu          Creation
 *
 ********************************************************************************************************************/
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

void set_control(void);

// Global state variables
cv::Mat current_original_img;
cv::Mat current_grayscale_mask;
std::string current_image_path;

// Mutex Setup
std::mutex state_mutex;

// Callback Functions For GUI
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Updates the current image with the glow effect using the loaded mask.
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
 * Callback for Key Level slider updates.
 */
void bar_key_level_cb(int newValue) {
	std::lock_guard<std::mutex> lock(state_mutex);
	param_KeyLevel = newValue;
	std::cout << "Key Level updated to: " << param_KeyLevel << std::endl;
	updateImage();
}

/**
 * Callback for Key Scale slider updates.
 */
void bar_key_scale_cb(int newValue) {
	std::lock_guard<std::mutex> lock(state_mutex);
	param_KeyScale = newValue;
	std::cout << "Key Scale updated to: " << param_KeyScale << std::endl;
	updateImage();
}
/**
 * Callback for Default Scale slider updates.
 */
void bar_default_scale_cb(int newValue) {
	std::lock_guard<std::mutex> lock(state_mutex);
	default_scale = newValue;
	std::cout << "Default Scale updated to: " << default_scale << std::endl;
	updateImage();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
	try {
		auto usage = []() {
			printf("Usage:\n");
			printf("   This program processes single images, directories, or video files.\n");
			printf("Key usage:\n");
			printf("   +: display delay increases by 30ms, max to 300ms\n");
			printf("   -: display delay decreases by 30ms, min to 30ms\n");
			printf("   p: display pauses \n");
			printf("   q: program exits \n");
			printf("   click bottom buttons on the control GUI to switch effect modes\n\n");
		};

		usage();

		// Launch GUI in a separate thread
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

			// Wait for user to exit
			while (true) {
				char key = cv::waitKey(30);
				if (key == 'q') break;
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
				if (key == 'q') break;
				if (key == 13) {
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

			// choose default path or customized path
			if (videoInputOption == "y") {
				videoPath = "D:/csi4900/GlowEffect/glow_effect/resource/racing_cars.sd.mp4"; // use your own path
			}
			else {
				printf("Enter the full path of the video file: ");
				std::cin >> videoPath;
			}

			// make sure videoPath is not empty
			if (videoPath.empty()) {
				std::cout << "videoPath cannot be empty" << std::endl;
				return 1;
			}

			// Validate if the videoPath exists and is a regular file
			if (!std::filesystem::exists(videoPath)) {
				std::cout << "The specified file path does not exist." << std::endl;
				return 1;
			}

			// if the file is valid
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

		// Wait for GUI thread to finish before existing
		guiThread.join();
	}
	catch (const std::exception& e) {
		std::cerr << "Unexpected error occurred: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}