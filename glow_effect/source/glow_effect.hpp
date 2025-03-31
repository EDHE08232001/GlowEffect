﻿#ifndef GLOW_EFFECT_HPP
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

/**
 * @brief Applies mipmapping to an image.
 *
 * @param width The width of the source image.
 * @param height The height of the source image.
 * @param scale The scale factor for mipmapping.
 * @param src_img Pointer to the source image data (uchar4 format).
 * @param dst_img Pointer to the destination image data (uchar4 format).

#include <string>
#include <opencv2/imgproc.hpp>

/**
 * External GUI-controlled variables.
 */
extern int button_id;
extern int param_KeyScale;
extern int param_KeyLevel;
extern int default_scale;
extern cv::Vec3b param_KeyColor;

/**
 * @brief Applies a CUDA-based mipmapping filter to an RGBA image.
 *
 * @param width    Width of the source image.
 * @param height   Height of the source image.
 * @param scale    Scale factor for mipmapping.
 * @param src_img  Pointer to the source image data (uchar4).
 * @param dst_img  Pointer to the destination image data (uchar4).
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);

void filter_and_blend(const int width, const int height, const float scale, const float key_scale,
    const uchar4* mask_img,   // 4-channel segmentation mask (for mipmap/alpha generation)
    const uchar4* glow_img,   // Glow (highlighted) image
    const uchar4* base_img,   // Original frame (base image for blending)
    uchar4* output_img);      // Blended output image (RGBA)

cv::Mat threshold_mask_to_rgba(const cv::Mat& grayscale_mask, int param_KeyLevel, int tolerance = 0);
/**
 * @brief Applies a glow effect to an image using a provided grayscale mask.
 *
 * @param image_nm       Path to the input image file.
 * @param grayscale_mask A single-channel mask guiding the glow effect.
 */
void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask);

/**
 * @brief Applies a glow effect to a video file.
 *
 * @param video_nm     Path to the input video file.
 * @param planFilePath Path to the TRT plan file.
 */
void glow_effect_video(const char* video_nm);
void glow_effect_video_OPT(const char* video_nm);
void glow_effect_video_triple_buffer(const char* video_nm, std::string planFilePath);

/**
 * @brief Applies a glow effect to a video file using CUDA Graph acceleration.
 *
 * This function maintains the same parallel processing approach as glow_effect_video,
 * but enhances the segmentation phase with CUDA Graph technology to reduce kernel
 * launch overhead and improve overall performance. The function processes frames in
 * batches of 8, dividing them into two sub-batches of 4 frames each for parallel processing.
 *
 * @param video_nm     Path to the input video file.
 * @param planFilePath Path to the TRT plan file.
 */
void glow_effect_video_graph(const char* video_nm, std::string planFilePath);

/**
 * @brief Applies a glow effect to video using parallel processing of single-batch TRT model
 *
 * This function processes video frames in parallel using multiple streams and the
 * single-batch TensorRT model. It employs the updated TRTInference that correctly
 * handles CUDA Graph capture for post-processing operations.
 *
 * The function maintains the same glow/bloom effect pipeline but organizes the processing
 * for optimal parallel execution with proper error handling.
 *
 * @param video_nm Path to the input video file
 * @param planFilePath Path to the single-batch TensorRT plan file
 */
void glow_effect_video_single_batch_parallel(const char* video_nm, std::string planFilePath);

/**
 * @brief Applies a "blow" (highlight) effect based on a grayscale mask.
 *
 * If any pixel in the input mask is within the specified tolerance (Delta) of
 * the key level (param_KeyLevel), the entire output image is filled with a pink overlay.
 *
 * @param mask          A single-channel (CV_8UC1) mask.
 * @param dst_rgba      Destination RGBA image (CV_8UC4); will be created/overwritten.
 * @param param_KeyLevel Key level parameter controlling the highlight trigger.
 * @param Delta         Tolerance range around param_KeyLevel.
 */
void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta);

/**
<<<<<<< HEAD
 * @brief Applies mipmapping and other transformations to an image.
 *
 * @param src The source image.
 * @param dst The destination image after applying mipmapping.
 * @param scale The scale factor for mipmapping.
 * @param param_KeyLevel The key level parameter controlling the transformation.
 * @brief Applies a mipmap filtering operation on a grayscale image and outputs an RGBA image.
 *
 * Converts the input grayscale image to an RGBA image where only pixels equal to
 * param_KeyLevel are made opaque, applies the CUDA mipmap filter, and then converts
 * the filtered result back into an OpenCV image.
 *
 * @param src            Source grayscale image (CV_8UC1).
 * @param dst            Destination RGBA image (CV_8UC4) after filtering.
 * @param scale          Scale factor used by the mipmap filter.
 * @param param_KeyLevel Grayscale value that determines which pixels are made opaque.
 */
void apply_mipmap(const cv::Mat& src, cv::Mat& dst, float scale, int param_KeyLevel);

/**
 * @brief Asynchronously applies a CUDA-based mipmap filter to a grayscale image and outputs an RGBA image.
 *
 * Converts the input grayscale image to an RGBA buffer (keeping only the pixels
 * equal to param_KeyLevel as opaque), then uses the asynchronous filter_mipmap_async function
 * on the provided non-blocking CUDA stream. The result is written directly into the caller-provided
 * pinned destination buffer.
 *
 * @param input_gray    The source single-channel (CV_8UC1) grayscale image.
 * @param dst_img       Pointer to the preallocated pinned host memory for the output RGBA image.
 * @param scale         The scale factor used by the mipmap filter.
 * @param param_KeyLevel Grayscale value determining which pixels become opaque.
 * @param stream        The non-blocking CUDA stream on which to perform asynchronous mipmap filtering.
 */
void apply_mipmap_async(const cv::Mat& input_gray, uchar4* dst_img, float scale, int param_KeyLevel, cudaStream_t stream);

/**
 * @brief Blends two images using a mask and per-pixel alpha blending.
 *
 * The function blends a source image with a highlighted image using a grayscale mask
 * (interpreted as alpha values scaled by param_KeyScale), producing a final blended RGBA output.
 *
 * @param src_img        First source image.
 * @param dst_rgba       Second source image (highlighted).
 * @param mipmap_result  Grayscale mask image used for alpha blending.
 * @param output_image   Destination blended image.
 * @param param_KeyScale Blending factor scaling.
 */
void mix_images(const cv::Mat& src_img, const cv::Mat& dst_rgba, const cv::Mat& mipmap_result, cv::Mat& output_image, float param_KeyScale);

#endif // GLOW_EFFECT_HPP
