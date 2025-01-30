#ifndef GLOW_EFFECT_HPP
#define GLOW_EFFECT_HPP

/**
 * @file glow_effect.hpp
 * @brief Declarations for glow-related image and video processing functions using CUDA, TensorRT, and OpenCV.
 *
 * @details
 * This header provides function declarations and external variable definitions for various glow and blending effects.
 * It leverages CUDA kernels (for mipmapping) and integrates with OpenCV for image manipulation.
 *
 * The typical workflow involves:
 * 1. Generating or receiving a mask (e.g., a segmentation mask).
 * 2. Optionally converting and blending images using the mask and/or custom parameters.
 * 3. Combining results to achieve glow/bloom effects in either images or video frames.
 *
 * **Dependencies:**
 * - CUDA Runtime
 * - OpenCV (for Mat and color structures)
 * - External libraries for inference (TensorRT) are expected to be utilized in corresponding .cpp implementations
 *   but do not need direct references here.
 *
 * **Important:**
 * - Avoid adding `using namespace std;` in headers to reduce the risk of name collisions.
 * - Ensure that `uchar4` is available from the CUDA headers or defined appropriately in your build environment.
 */

 // ------------------------------------------------------------------------------------------------
 // Includes
 // ------------------------------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>

// ------------------------------------------------------------------------------------------------
// External variables (controlled by GUI)
// ------------------------------------------------------------------------------------------------

/**
 * @var button_id
 * Global integer indicating which button is selected in the GUI.
 */
extern int button_id;

/**
 * @var param_KeyScale
 * Parameter that scales the blend factor (e.g., for alpha) or otherwise influences intensity in image mixing.
 * Controlled by a "Key Scale" slider in the GUI.
 */
extern int param_KeyScale;

/**
 * @var param_KeyLevel
 * Parameter that defines a threshold or key level (often for grayscale values).
 * Controlled by a "Key Level" slider in the GUI.
 */
extern int param_KeyLevel;

/**
 * @var default_scale
 * Parameter that represents a default scale factor (e.g., used in mipmap operations).
 * Controlled by a "Default Scale" slider in the GUI.
 */
extern int default_scale;

/**
 * @var param_KeyColor
 * Parameter that defines a key color (BGR format) if needed for color-based segmentation or highlighting.
 * This is a 3-channel vector (Blue, Green, Red).
 */
extern cv::Vec3b param_KeyColor;

// ------------------------------------------------------------------------------------------------
// Function Declarations
// ------------------------------------------------------------------------------------------------

/**
 * @brief Applies a CUDA-based mipmapping filter to an RGBA image.
 *
 * @details
 * This function uses a CUDA kernel (implemented in the .cpp) to downscale or otherwise filter
 * an input RGBA image. The result is written back to a separate RGBA buffer.
 *
 * **Usage Example:**
 * @code
 * uchar4* deviceSrc = ...; // device memory pointer to source image
 * uchar4* deviceDst = ...; // device memory pointer to destination image
 * filter_mipmap(width, height, 0.5f, deviceSrc, deviceDst);
 * // deviceDst now holds the filtered image
 * @endcode
 *
 * @param width       The width of the source image in pixels.
 * @param height      The height of the source image in pixels.
 * @param scale       The scale factor for mipmapping (e.g., 0.5 for half size).
 * @param src_img     Pointer to the source image data in RGBA (uchar4) format.
 * @param dst_img     Pointer to the destination image data in RGBA (uchar4) format.
 *
 * @pre `src_img` and `dst_img` must be allocated and must contain at least `width * height` elements.
 * @post The destination buffer contains the scaled (or otherwise filtered) image.
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);

/**
 * @brief Applies a glow effect to a single image, using a provided grayscale mask.
 *
 * @details
 * This function:
 * 1. Loads the input image from the specified path.
 * 2. Uses the grayscale mask to determine which regions receive the glow effect.
 * 3. May rely on multiple operations (e.g., glow_blow, mipmap, alpha blending) to generate the final result.
 *
 * **Usage Example:**
 * @code
 * cv::Mat mask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
 * glow_effect_image("input.png", mask);
 * @endcode
 *
 * @param image_nm        The file path of the input image (supports formats compatible with OpenCV).
 * @param grayscale_mask  A single-channel (CV_8UC1) mask that influences the glow effect regions.
 *
 * @pre The file at `image_nm` should be a valid image file readable by OpenCV.
 * @pre `grayscale_mask` should have the same width and height as the input image for best results.
 * @post Displays the final result in a window and saves it to `./results/final_result.png`.
 */
void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask);

/**
 * @brief Applies a glow effect to a video file, processing each frame based on a segmentation or grayscale mask.
 *
 * @details
 * This function:
 * 1. Opens the specified video file.
 * 2. For each frame, performs TensorRT-based inference to obtain a mask (in the .cpp implementation),
 *    or uses a provided mask to determine glow regions.
 * 3. Uses mipmap and other blending techniques to enhance glow/bloom in target regions.
 * 4. Saves each processed frame to disk and then compiles them into a new video.
 *
 * **Usage Example:**
 * @code
 * glow_effect_video("input_video.mp4");
 * // Output is saved as ./VideoOutput/processed_video.avi
 * @endcode
 *
 * @param video_nm			The file path of the input video (any format supported by OpenCV).
 * @param planFilePath		Path to trt plan
 *
 * @pre `video_nm` must refer to a valid video file.
 * @post Creates a directory `./VideoOutput` for the processed frames and assembles them into `processed_video.avi`.
 */
void glow_effect_video(const char* video_nm, std::string planFilePath);

/**
 * @brief Applies a "blow" (highlight) effect to an image based on a mask and thresholding.
 *
 * @details
 * This function scans the mask looking for pixels within a certain range around `param_KeyLevel`.
 * If found, it applies an overlay color (e.g., pink) across the entire output image.
 *
 * **Usage Example:**
 * @code
 * cv::Mat mask = cv::imread("segmentation_mask.png", cv::IMREAD_GRAYSCALE);
 * cv::Mat dst;
 * glow_blow(mask, dst, 128, 10); // highlight where mask pixels are ~128 ± 10
 * @endcode
 *
 * @param mask           A single-channel (CV_8UC1) mask indicating areas to highlight.
 * @param dst_rgba       Destination RGBA image. Will be created/overwritten with the overlay if conditions are met.
 * @param param_KeyLevel The key level parameter controlling the highlight trigger (e.g., mask value).
 * @param Delta          The tolerance range around `param_KeyLevel`. If `(mask_pixel - param_KeyLevel) < Delta`,
 *                       the highlight is applied.
 *
 * @pre `mask` must be a valid, non-empty single-channel image.
 * @post `dst_rgba` will either remain transparent if no target region is found or be fully overlaid with a highlight color if found.
 */
void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta);

/**
 * @brief Applies a mipmap operation (downscale or custom filtering) to a single-channel image, then outputs RGBA.
 *
 * @details
 * This function converts the input grayscale image into an RGBA buffer based on `param_KeyLevel`.
 * Pixels matching `param_KeyLevel` become fully opaque in the output; others become transparent.
 * Then it calls CUDA-based mipmap filtering (`filter_mipmap`) and writes the result to `dst`.
 *
 * **Usage Example:**
 * @code
 * cv::Mat gray = cv::imread("gray_mask.png", cv::IMREAD_GRAYSCALE);
 * cv::Mat mipmap_rgba;
 * apply_mipmap(gray, mipmap_rgba, 0.5f, 128);
 * // mipmap_rgba is now a CV_8UC4 image
 * @endcode
 *
 * @param src            The source single-channel (CV_8UC1) grayscale image.
 * @param dst            The destination RGBA image (CV_8UC4) after mipmap filtering.
 * @param scale          The scale factor for mipmapping (e.g., 0.5 for half size).
 * @param param_KeyLevel A grayscale value dictating which pixels become opaque before filtering.
 *
 * @pre `src` must be a valid grayscale image (1 channel, 8-bit).
 * @post `dst` is populated with the CUDA-filtered RGBA image.
 */
void apply_mipmap(const cv::Mat& src, cv::Mat& dst, float scale, int param_KeyLevel);

/**
 * @brief Blends two images based on a mask and alpha value, producing an output image.
 *
 * @details
 * This function uses alpha blending to mix `img1` and `img2`, guided by the single-channel `mask`.
 * An additional scale factor (`alpha`) can further adjust blending intensity.
 *
 * **Usage Example:**
 * @code
 * cv::Mat source = cv::imread("source.png", cv::IMREAD_COLOR);
 * cv::Mat highlight = cv::imread("highlight.png", cv::IMREAD_COLOR);
 * cv::Mat mask = cv::imread("mask.png", cv::IMREAD_GRAYSCALE);
 * cv::Mat blended;
 * mix_images(source, highlight, mask, blended, 0.7f);
 * // blended now has a combination of source and highlight
 * @endcode
 *
 * @param img1   The first source image (BGR or RGBA).
 * @param img2   The second source image (BGR or RGBA).
 * @param mask   A single-channel (CV_8UC1) mask that influences blending.
 * @param dst    The destination image (CV_8UC4) after blending.
 * @param alpha  The blending factor (0.0 - 1.0 recommended, though the function
 *               may internally scale or clamp values).
 *
 * @pre `img1`, `img2`, and `mask` should be the same resolution.
 *      `mask` should be single-channel if used as a direct alpha reference.
 * @post `dst` contains the blended RGBA image.
 */
void mix_images(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask, cv::Mat& dst, float alpha);

#endif // GLOW_EFFECT_HPP
