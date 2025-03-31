#ifndef IMAGE_PROCESSING_UTIL_HPP
#define IMAGE_PROCESSING_UTIL_HPP

#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>

/**
 * @brief Utility class for common image processing tasks.
 *
 * Provides static methods for retrieving image paths, extracting image shapes,
 * comparing images, and converting images (or batches) to Torch tensors.
 */
class ImageProcessingUtil {
public:
	/**
	 * @brief Retrieves all valid image file paths (jpg, jpeg, png, bmp) from a folder.
	 *
	 * The function scans the folder recursively.
	 *
	 * @param folderPath The path to the folder to scan.
	 * @return A vector of full file paths for valid image files.
	 * @throws std::filesystem::filesystem_error if directory access fails.
	 */
	static std::vector<std::string> getImagePaths(const std::string& folderPath);

	/**
	 * @brief Extracts the shape of an image as a 4D vector (1, channels, rows, cols).
	 *
	 * @param img_path The file path of the image.
	 * @return A cv::Vec4f containing (1, channels, rows, cols).
	 * @throws std::runtime_error if the image fails to load.
	 */
	static cv::Vec4f get_input_shape_from_image(const std::string& img_path);

	/**
	 * @brief Compares two images using PSNR and SSIM metrics.
	 *
	 * @param generated_img The generated image (expected as a float/double Mat, range typically [0,1]).
	 * @param gray_original The reference grayscale image.
	 * @note The results are logged to std::cout.
	 */
	static void compareImages(const cv::Mat& generated_img, const cv::Mat& gray_original);

	/**
	 * @brief Processes an image from a file and returns it as a Torch tensor.
	 *
	 * If @p grayscale is true, the image is loaded in grayscale and returned with shape [1, H, W];
	 * otherwise, the image is loaded in color, converted from BGR to RGB, normalized, and returned with shape [1, 3, H, W].
	 *
	 * @param img_path The path to the image file.
	 * @param grayscale Whether to load the image as grayscale.
	 * @return A Torch tensor representing the processed image.
	 * @throws std::invalid_argument if the image fails to load.
	 */
	static torch::Tensor process_img(const std::string& img_path, bool grayscale = false);

	/**
	 * @brief Processes an image from a GPU Mat and returns it as a Torch tensor.
	 *
	 * If @p grayscale is true, the image is converted to grayscale and returned with shape [1, H, W];
	 * otherwise, it is processed in color (converted from BGR to RGB, normalized) and returned with shape [1, 3, H, W].
	 *
	 * @param process_img The input cv::cuda::GpuMat image.
	 * @param grayscale Whether to convert the image to grayscale.
	 * @return A Torch tensor representing the processed image.
	 */
	static torch::Tensor process_img(const cv::cuda::GpuMat& process_img, bool grayscale = false);

	/**
	 * @brief Processes a batch of images and concatenates them into a single batched Torch tensor.
	 *
	 * Each image is processed individually, and the resulting tensors are concatenated along the batch dimension.
	 *
	 * @param img_paths A vector of image file paths.
	 * @param grayscale Whether to load images as grayscale.
	 * @return A 4D Torch tensor with shape [batch_size, channels, height, width].
	 * @throws std::invalid_argument if any image fails to load.
	 */
	static torch::Tensor process_img_batch(const std::vector<std::string>& img_paths, bool grayscale = false);
// >>>>>>> dc03d1e01975d937278105b157d6e05d46516332
};

#endif // IMAGE_PROCESSING_UTIL_HPP
