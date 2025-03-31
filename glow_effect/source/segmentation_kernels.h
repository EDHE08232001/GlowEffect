/**
 * @file segmentation_kernels.h
 * @brief Header file for CUDA-accelerated segmentation post-processing functions
 */

#ifndef SEGMENTATION_KERNELS_H
#define SEGMENTATION_KERNELS_H

#include <cuda_runtime.h>

 /**
  * @brief Launches a CUDA kernel to find the maximum class index across channels
  *
  * This function finds the argmax along the channel dimension (dim=1) of the segmentation output
  * and scales the result to fit within the 8-bit range. It replaces the PyTorch torch::max operation
  * with a custom CUDA kernel that can be captured in CUDA Graphs.
  *
  * @param input       Device pointer to input tensor with shape [batch, num_classes, height, width]
  * @param output      Device pointer to output tensor with shape [batch, height, width]
  * @param batch_size  Number of images in the batch
  * @param num_classes Number of segmentation classes
  * @param height      Height of each image
  * @param width       Width of each image
  * @param stream      CUDA stream to use for the kernel launch
  */
void launchArgmaxKernel(const float* input, unsigned char* output,
	int batch_size, int num_classes, int height, int width,
	cudaStream_t stream);

/**
 * @brief Launches a CUDA kernel to apply class-to-color mapping for visualization
 *
 * This function maps class indices to specific colors based on a predefined color map.
 * It's useful for creating visualization outputs from segmentation masks.
 *
 * @param input       Device pointer to input tensor with shape [batch, height, width]
 * @param output      Device pointer to output tensor with shape [batch, height, width, 4] (RGBA)
 * @param batch_size  Number of images in the batch
 * @param height      Height of each image
 * @param width       Width of each image
 * @param stream      CUDA stream to use for the kernel launch
 */
void launchColorMapKernel(const unsigned char* input, unsigned char* output,
	int batch_size, int height, int width,
	cudaStream_t stream);

#endif // SEGMENTATION_KERNELS_H