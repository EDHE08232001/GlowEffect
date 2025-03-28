#ifndef GLOW_EFFECT_HPP
#define GLOW_EFFECT_HPP

/**
 * @file glow_effect.hpp
 * @brief Declares functions and external variables for applying glow effects using CUDA, TensorRT, and OpenCV.
 *
 * @details
 * This header provides declarations for functions that apply glow effects to images and videos.
 * It also declares external variables that are controlled by the GUI (wxWidgets).
 */

 // Required CUDA type definitions
#include <cuda_runtime.h>

// Include OpenCV core module for cv::Mat and cv::Vec3b
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>

// If uchar4 is not defined in cuda_runtime.h, you might need to define it or include appropriate CUDA headers
// Typically, cuda_runtime.h should define uchar4, but ensure it's available

// ** Avoid using 'using namespace std;' in headers to prevent name collisions **
// No 'using namespace std;' here

// Declare external variables controlled by the GUI
extern int button_id;                  ///< Tracks which button is selected in the GUI
extern int param_KeyScale;             ///< Parameter for key scaling (controlled by Key Scale slider)
extern int param_KeyLevel;             ///< Parameter for key level (controlled by Key Level slider)
extern int default_scale;              ///< Parameter for default scaling (controlled by Default Scale slider)
extern cv::Vec3b param_KeyColor;       ///< Parameter for key color (if used)

// Declare functions for applying glow effects

/**
 * @brief Applies mipmapping to an image.
 *
 * @param width The width of the source image.
 * @param height The height of the source image.
 * @param scale The scale factor for mipmapping.
 * @param src_img Pointer to the source image data (uchar4 format).
 * @param dst_img Pointer to the destination image data (uchar4 format).
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);

void filter_and_blend(const int width, const int height, const float scale, const float key_scale,
    const uchar4* mask_img,   // 4-channel segmentation mask (for mipmap/alpha generation)
    const uchar4* glow_img,   // Glow (highlighted) image
    const uchar4* base_img,   // Original frame (base image for blending)
    uchar4* output_img);      // Blended output image (RGBA)

cv::Mat threshold_mask_to_rgba(const cv::Mat& grayscale_mask, int param_KeyLevel, int tolerance = 0);
/**
 * @brief Applies a glow effect to a single image using a grayscale mask.
 *
 * @param image_nm The name/path of the input image.
 * @param grayscale_mask The grayscale mask to apply the glow effect.
 */
void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask);

/**
 * @brief Applies a glow effect to a video file.
 *
 * @param video_nm The name/path of the input video file.
 */
void glow_effect_video(const char* video_nm);
void glow_effect_video_OPT(const char* video_nm);

/**
 * @brief Applies a "blow" effect to an image based on a mask.
 *
 * @param mask The mask determining where to apply the blow effect.
 * @param dst_rgba The destination image with the blow effect applied.
 * @param param_KeyLevel The key level parameter controlling the intensity.
 * @param Delta Additional parameter for the blow effect (purpose defined in implementation).
 */
void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta);

/**
 * @brief Applies mipmapping and other transformations to an image.
 *
 * @param src The source image.
 * @param dst The destination image after applying mipmapping.
 * @param scale The scale factor for mipmapping.
 * @param param_KeyLevel The key level parameter controlling the transformation.
 */
void apply_mipmap(const cv::Mat& src, cv::Mat& dst, float scale, int param_KeyLevel);

/**
 * @brief Mixes two images based on a mask and an alpha value.
 *
 * @param img1 The first source image.
 * @param img2 The second source image.
 * @param mask The mask determining where to mix the images.
 * @param dst The destination image after mixing.
 * @param alpha The blending factor.
 */
void mix_images(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask, cv::Mat& dst, float alpha);

#endif // GLOW_EFFECT_HPP

