#ifndef MIPMAP_H
#define MIPMAP_H

#include <cuda_runtime.h>
#include "old_movies.cuh"  // This header should include any common CUDA error checking utilities (e.g., checkCudaErrors)

/**
 * @brief Synchronously filters an image by generating mipmap levels and retrieving a blurred version.
 *
 * This function creates a complete mipmap chain from the input image stored in host memory, applies a blur
 * effect via mipmapping, and retrieves the resulting image into a host output buffer. The operation is performed
 * synchronously, meaning the function does not return until all GPU operations are complete.
 *
 * @param width   The width of the input image.
 * @param height  The height of the input image.
 * @param scale   The scale factor used for mipmap sampling (used to compute the uniform level-of-detail).
 * @param src_img Pointer to the input image data in host memory. The image is expected to be in RGBA format (stored as uchar4).
 * @param dst_img Pointer to the output image data in host memory where the processed image will be stored.
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);

/**
 * @brief Asynchronously applies a mipmap filter to an image.
 *
 * This function performs the mipmap filtering operation asynchronously. It allocates a CUDA mipmapped array,
 * copies the source image into the array, generates the mipmap levels, and retrieves the filtered output into
 * a host buffer�all while using the provided CUDA stream. This allows overlapping data transfers with kernel execution.
 *
 * @param width   The width of the input image.
 * @param height  The height of the input image.
 * @param scale   The scale factor used for mipmap sampling (used to compute the uniform level-of-detail).
 * @param src_img Pointer to the source image data in host memory. The image is expected to be in RGBA format (stored as uchar4).
 * @param dst_img Pointer to the output image data in host memory where the processed image will be stored.
 * @param stream  The CUDA stream to use for asynchronous operations.
 */
void filter_mipmap_async(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img, cudaStream_t stream);

#endif // MIPMAP_H
