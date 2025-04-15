/*******************************************************************************************************************
 * FILE NAME   : mipmap.h
 *
 * DESCRIPTION : Header file declaring CUDA functions for mipmapping operations including 
 *               image blurring and glow effects using GPU-accelerated texture processing.
 *
 * VERSION HISTORY
 * 2022 OCT 10      Yu Liu          Initial version
 * 2025 APR 15      Modified        Improved documentation and formatting
 *
 ********************************************************************************************************************/

 #ifndef MIPMAP_H
 #define MIPMAP_H
 
 #include <cuda_runtime.h>
 
 /**
  * @brief Applies a mipmap-based blur filter to an image (synchronous version).
  *
  * Creates a complete mipmap chain from the input image, applies a blur effect via mipmapping,
  * and retrieves the resulting image into the output buffer. This function blocks until
  * all GPU operations are complete.
  *
  * @param width   Width of the input image in pixels.
  * @param height  Height of the input image in pixels.
  * @param scale   Scale factor controlling blur intensity (larger values = more blur).
  * @param src_img Input image data in RGBA format (uchar4).
  * @param dst_img Output buffer for the processed image (must be pre-allocated).
  */
 void filter_mipmap(const int width, const int height, const float scale, 
                   const uchar4* src_img, uchar4* dst_img);
 
 /**
  * @brief Filters and blends images using mipmap-based glow effect.
  * 
  * Creates a mipmap chain from the mask image, uses it to compute per-pixel alpha values,
  * then blends the base and glow images together. This produces a controllable glow effect
  * where the mask has higher values.
  *
  * @param width      Width of the images in pixels.
  * @param height     Height of the images in pixels.
  * @param scale      Scale factor controlling blur spread of the glow.
  * @param key_scale  Multiplier for the alpha values (boosts glow intensity).
  * @param mask_img   Mask image determining glow regions (used to create mipmaps).
  * @param glow_img   Glow/highlight image to blend over the base image.
  * @param base_img   Original base image to receive the glow effect.
  * @param output_img Output buffer for the blended result (must be pre-allocated).
  */
 void filter_and_blend(const int width, const int height, const float scale, const float key_scale,
                      const uchar4* mask_img, const uchar4* glow_img, 
                      const uchar4* base_img, uchar4* output_img);
 
 /**
  * @brief Asynchronously applies a mipmap filter to an image.
  *
  * Performs the same operation as filter_mipmap() but asynchronously using the provided
  * CUDA stream. This allows overlapping data transfers with kernel execution for better
  * performance when processing multiple images.
  *
  * @param width   Width of the input image in pixels.
  * @param height  Height of the input image in pixels.
  * @param scale   Scale factor controlling blur intensity.
  * @param src_img Input image data in RGBA format (uchar4).
  * @param dst_img Output buffer for the processed image (must be pre-allocated).
  * @param stream  CUDA stream to use for asynchronous operations.
  */
 void filter_mipmap_async(const int width, const int height, const float scale, 
                         const uchar4* src_img, uchar4* dst_img, cudaStream_t stream);
 
 #endif // MIPMAP_H