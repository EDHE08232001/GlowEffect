/**
 * @file segmentation_kernels.cu
 * @brief CUDA kernel implementations for segmentation post-processing
 */

#include "segmentation_kernels.h"
#include <cfloat>  // For FLT_MAX
#include <stdio.h>

 /**
  * @brief CUDA kernel to find the maximum class index across channels for segmentation output
  *
  * This kernel finds the argmax along the channel dimension (dim=1) of the segmentation output
  * and scales the result according to a scaling factor.
  */
__global__ void argmaxKernel(const float* input, unsigned char* output,
	int batch_size, int num_classes, int height, int width,
	int scaleFactor) {
	// Calculate global thread position
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int b = blockIdx.z;

	// Check bounds
	if (x < width && y < height && b < batch_size) {
		float maxVal = -FLT_MAX;
		int maxIdx = 0;

		// Find max class
		for (int c = 0; c < num_classes; c++) {
			// Index into 4D tensor [batch, channels, height, width]
			const int idx = ((b * num_classes + c) * height + y) * width + x;
			const float val = input[idx];
			if (val > maxVal) {
				maxVal = val;
				maxIdx = c;
			}
		}

		// Scale the class index to 8-bit range
		const int outputIdx = (b * height + y) * width + x;
		output[outputIdx] = (unsigned char)(maxIdx * scaleFactor);
	}
}

/**
 * @brief Wrapper function to launch the argmax kernel
 */
void launchArgmaxKernel(const float* input, unsigned char* output,
	int batch_size, int num_classes, int height, int width,
	cudaStream_t stream) {
	// Block size can be tuned for your specific GPU architecture
	dim3 threads(16, 16);

	// Grid size based on image dimensions and batch size
	dim3 blocks((width + threads.x - 1) / threads.x,
		(height + threads.y - 1) / threads.y,
		batch_size);

	// Scale factor for class indices (assuming 21 classes for semantic segmentation)
	int scaleFactor = 255 / 21;  // Adjust this based on your model's output

	// Launch kernel - fix the spacing in the kernel launch syntax
	argmaxKernel << <blocks, threads, 0, stream >> > (
		input, output, batch_size, num_classes, height, width, scaleFactor);

	// The stream synchronization should be done by the caller if needed
	// Don't synchronize here if this function is called during graph capture
}

// Predefined colormap for visualization (example with 21 classes)
__constant__ unsigned char d_colormap[21][4] = {
	{0, 0, 0, 255},       // Background (black)
	{128, 0, 0, 255},     // Class 1 (maroon)
	{0, 128, 0, 255},     // Class 2 (green)
	{128, 128, 0, 255},   // Class 3 (olive)
	{0, 0, 128, 255},     // Class 4 (navy)
	{128, 0, 128, 255},   // Class 5 (purple)
	{0, 128, 128, 255},   // Class 6 (teal)
	{128, 128, 128, 255}, // Class 7 (gray)
	{64, 0, 0, 255},      // Class 8 (dark red)
	{192, 0, 0, 255},     // Class 9 (red)
	{64, 128, 0, 255},    // Class 10 (dark lime)
	{192, 128, 0, 255},   // Class 11 (orange)
	{64, 0, 128, 255},    // Class 12 (dark blue)
	{192, 0, 128, 255},   // Class 13 (pink)
	{64, 128, 128, 255},  // Class 14 (dark cyan)
	{192, 128, 128, 255}, // Class 15 (light pink)
	{0, 64, 0, 255},      // Class 16 (dark green)
	{128, 64, 0, 255},    // Class 17 (brown)
	{0, 192, 0, 255},     // Class 18 (lime)
	{128, 192, 0, 255},   // Class 19 (yellow)
	{0, 64, 128, 255}     // Class 20 (dark cyan)
};

/**
 * @brief CUDA kernel to apply color mapping for visualization
 */
__global__ void colorMapKernel(const unsigned char* input, unsigned char* output,
	int batch_size, int height, int width, int scaleFactor) {
	// Calculate global thread position
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int b = blockIdx.z;

	// Check bounds
	if (x < width && y < height && b < batch_size) {
		// Index into input (class indices)
		const int inputIdx = (b * height + y) * width + x;
		unsigned char classIdx = input[inputIdx] / scaleFactor;

		// Clamp class index to valid range
		if (classIdx >= 21) classIdx = 0;

		// Index into output (RGBA)
		const int outputIdx = ((b * height + y) * width + x) * 4;

		// Copy color from colormap
		output[outputIdx] = d_colormap[classIdx][0]; // R
		output[outputIdx + 1] = d_colormap[classIdx][1]; // G
		output[outputIdx + 2] = d_colormap[classIdx][2]; // B
		output[outputIdx + 3] = d_colormap[classIdx][3]; // A
	}
}

/**
 * @brief Wrapper function to launch the color mapping kernel
 */
void launchColorMapKernel(const unsigned char* input, unsigned char* output,
	int batch_size, int height, int width,
	cudaStream_t stream) {
	// Block size can be tuned for your specific GPU architecture
	dim3 threads(16, 16);

	// Grid size based on image dimensions and batch size
	dim3 blocks((width + threads.x - 1) / threads.x,
		(height + threads.y - 1) / threads.y,
		batch_size);

	// Scale factor (same as used in argmaxKernel)
	int scaleFactor = 255 / 21;

	// Launch kernel - fix the spacing in the kernel launch syntax
	colorMapKernel << <blocks, threads, 0, stream >> > (
		input, output, batch_size, height, width, scaleFactor);

	// The stream synchronization should be done by the caller if needed
}