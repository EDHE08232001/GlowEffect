/*******************************************************************************************************************
 * FILE NAME   : mipmap.cu
 *
 * PROJECT NAME: Cuda Learning
 *
 * DESCRIPTION : Implements CUDA kernels and host functions for generating mipmap levels,
 *               retrieving mipmapped data, and applying a blur effect via mipmapping.
 *
 * VERSION HISTORY
 * 2022 OCT 10      Yu Liu          Creation
 * 2022 OCT 26      Yu Liu          Moved V-shaped curve into CUDA
 * 2022 OCT 27      Yu Liu          Required texReadMode = cudaReadModeNormalizedFloat for linear filtering;
 *                                  corrected phase shift by using (x+1.f)/(y+1.f) rather than (x+0.5)/(y+0.5)
 *
 ********************************************************************************************************************/

#include "old_movies.cuh"
#include "mipmap.h"
extern bool button_State[5];


/**
 * @brief CUDA kernel to generate a mipmap level by downscaling an input texture.
 *
 * This kernel performs 2x2 averaging of texels from the input texture to create a
 * lower resolution mipmap level. The resulting color is converted to the [0,255] range,
 * clamped, and written to an output surface.
 *
 * @param mipOutput  CUDA surface object for writing the output mipmap level.
 * @param mipInput   CUDA texture object for reading the input image.
 * @param imageW     Width (in pixels) of the output mipmap level.
 * @param imageH     Height (in pixels) of the output mipmap level.
 */

__global__ void d_gen_mipmap(
	cudaSurfaceObject_t mipOutput,
	cudaTextureObject_t mipInput,
	uint imageW,
	uint imageH
) {
	// Compute the thread's coordinates in the output image.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// Compute normalized pixel size.
	float px = 1.0f / static_cast<float>(imageW);
	float py = 1.0f / static_cast<float>(imageH);

	// Ensure the thread operates only within valid image dimensions.
	if ((x < imageW) && (y < imageH)) {
		// Take the average of 4 neighboring texels using normalized texture access.
		float4 color =
			tex2D<float4>(mipInput, (x + 0.0f) * px, (y + 0.0f) * py) +
			tex2D<float4>(mipInput, (x + 1.0f) * px, (y + 0.0f) * py) +
			tex2D<float4>(mipInput, (x + 1.0f) * px, (y + 1.0f) * py) +
			tex2D<float4>(mipInput, (x + 0.0f) * px, (y + 1.0f) * py);

		// Compute the average color and scale it to the [0, 255] range.
		color /= 4.0f;
		color *= 255.0f;

		// Clamp the color values to a maximum of 255.
		color = fminf(color, make_float4(255.0f));

		// Write the resulting color to the output surface.
		surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
	}
}


/**
 * @brief Generates a complete mipmap chain from a given CUDA mipmapped array.
 *
 * This host function creates successive mipmap levels by halving the dimensions
 * of the current level until the image is reduced to 1x1. For each level, a texture
 * object is created for reading and a surface object is created for writing.
 *
 * @param mipmapArray Reference to the CUDA mipmapped array.
 * @param size        Extent (width and height) of the highest resolution mipmap level.
 */
static void gen_mipmap(cudaMipmappedArray_t& mipmapArray, cudaExtent size) {
	// Initialize current dimensions.
	size_t width = size.width;
	size_t height = size.height;
	uint level = 0;  // Mipmap level counter

	// Continue generating levels until dimensions reduce to 1x1.
	while (width != 1 || height != 1) {
		// Compute dimensions for the next mipmap level.
		width = MAX((size_t)1, width / 2);
		height = MAX((size_t)1, height / 2);

		// Retrieve current (levelFrom) and next (levelTo) mipmap levels.
		cudaArray_t levelFrom;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		cudaArray_t levelTo;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

		// Verify that the next level has the correct dimensions.
		cudaExtent levelToSize;
		checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		assert(levelToSize.width == width);
		assert(levelToSize.height == height);
		assert(levelToSize.depth == 0);

		// Create texture object for reading from the current level.
		cudaTextureObject_t texInput;
		cudaResourceDesc texResrc = {};
		texResrc.resType = cudaResourceTypeArray;
		texResrc.res.array.array = levelFrom;

		cudaTextureDesc texDescr = {};
		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeNormalizedFloat;

		checkCudaErrors(cudaCreateTextureObject(&texInput, &texResrc, &texDescr, NULL));

		// Create surface object for writing to the next level.
		cudaSurfaceObject_t surfOutput;
		cudaResourceDesc surfRes = {};
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = levelTo;

		checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

		// Configure kernel launch dimensions.
		dim3 blockSize(16, 16, 1);
		dim3 gridSize((uint(width) + blockSize.x - 1) / blockSize.x,
			(uint(height) + blockSize.y - 1) / blockSize.y,
			1);

		// Launch the kernel to generate the current mipmap level.
		d_gen_mipmap << <gridSize, blockSize >> > (surfOutput, texInput, (uint)width, (uint)height);

		// Synchronize and check for kernel errors.
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		// Clean up texture and surface objects.
		checkCudaErrors(cudaDestroySurfaceObject(surfOutput));
		checkCudaErrors(cudaDestroyTextureObject(texInput));

		// Proceed to the next mipmap level.
		level++;
	}
}

/**
 * @brief CUDA kernel to sample a mipmapped texture with per-pixel level-of-detail (LOD).
 *
 * For each pixel, this kernel uses a corresponding LOD value from the provided array,
 * samples the texture accordingly, and writes the resulting color (converted to uchar4)
 * into the output buffer.
 *
 * @param texEngine CUDA texture object for the mipmapped texture.
 * @param width     Width of the output image.
 * @param height    Height of the output image.
 * @param lod       Pointer to an array of per-pixel LOD values.
 * @param dout      Output buffer for storing the resulting uchar4 color values.
 */
__global__ void d_get_mipmap(
	cudaTextureObject_t texEngine,
	const int width,
	const int height,
	const float* lod,
	uchar4* dout
) {
	int xi = blockIdx.x * blockDim.x + threadIdx.x;
	int yi = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = yi * width + xi;

	float u = static_cast<float>(xi) / static_cast<float>(width);
	float v = static_cast<float>(yi) / static_cast<float>(height);
	bool state;

	if (xi < width && yi < height) {
		// Sample the texture using the per-pixel LOD.
		float4 data = tex2DLod<float4>(texEngine, u, v, lod[idx], &state);
		// Convert the sampled color to uchar4 and write it to the output buffer.
		dout[idx] = to_uchar4(255.0f * data);
	}
}

/**
 * @brief CUDA kernel to sample a mipmapped texture with a uniform LOD.
 *
 * This kernel uses a single, uniform LOD computed from the scale factor for all pixels,
 * samples the texture accordingly, and writes the resulting color (converted to uchar4)
 * into the output buffer.
 *
 * @param texEngine CUDA texture object for the mipmapped texture.
 * @param width     Width of the output image.
 * @param height    Height of the output image.
 * @param scale     Scale factor used to compute the uniform LOD.
 * @param dout      Output buffer for storing the resulting uchar4 color values.
 */
__global__ void d_get_mipmap(
	cudaTextureObject_t texEngine,
	const int width,
	const int height,
	const float scale,
	uchar4* dout
) {
	int xi = blockIdx.x * blockDim.x + threadIdx.x;
	int yi = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = yi * width + xi;

	float u = (xi + 0.5f) / static_cast<float>(width);
	float v = (yi + 0.5f) / static_cast<float>(height);

	// Compute the uniform LOD based on the scale factor.
	float lod = log2(scale);

	if (xi < width && yi < height) {
		// Sample the texture using the uniform LOD.
		float4 data = tex2DLod<float4>(texEngine, u, v, lod);
		// Convert the sampled color to uchar4 and write it to the output buffer.
		dout[idx] = to_uchar4(255.0f * data);
	}
}

// Unified kernel: Samples the mipmapped texture (built from the mask)
// to compute a per-pixel alpha, applies a key scale to boost the glow,
// then blends the base image with the glow image.
__global__ void post_processing_kernel(const uchar4* base_img,
	const uchar4* glow_img,
	cudaTextureObject_t mip_tex,
	uchar4* output,
	int width,
	int height,
	float scale,
	float key_scale)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < width && y < height)
	{
		int idx = y * width + x;

		// Compute normalized texture coordinates.
		float u = (x + 0.5f) / (float)width;
		float v = (y + 0.5f) / (float)height;

		// Compute LOD (level-of-detail). Adjust this if needed.
		float lod = log2(fabs(u - 0.5f) * fabs(v - 0.5f) * scale + 1.0f);

		// Sample the mipmapped texture from the mask.
		float4 mip_val = tex2DLod<float4>(mip_tex, u, v, lod);
		// Assume the red channel of mip_val holds the key; normalize to [0,1].
		float alpha = fminf(mip_val.x / 255.0f, 1.0f);
		// Apply the key scale factor (boosting the glow).
		alpha = fminf(alpha * key_scale, 1.0f);

		// Retrieve the base (original) and glow pixels.
		uchar4 base_pixel = base_img[idx];
		uchar4 glow_pixel = glow_img[idx];

		// Blend: output = alpha * glow + (1 - alpha) * base.
		uchar4 blended;
		blended.x = (unsigned char)(alpha * glow_pixel.x + (1.0f - alpha) * base_pixel.x);
		blended.y = (unsigned char)(alpha * glow_pixel.y + (1.0f - alpha) * base_pixel.y);
		blended.z = (unsigned char)(alpha * glow_pixel.z + (1.0f - alpha) * base_pixel.z);
		blended.w = 255; // Fully opaque.
		output[idx] = blended;
	}
}



/**
 * @brief Retrieves a mipmap image with uniform blur using CUDA texture sampling.
 *
 * This function creates a texture object from a CUDA mipmapped array and launches a kernel
 * to retrieve the mipmapped image. The resulting image is copied from device to host.
 *
 * @param mm_array  CUDA mipmapped array containing the mipmap levels.
 * @param img_size  Dimensions of the image and number of mipmap levels (int3: {width, height, n_level}).
 * @param scale     Scale factor used to compute the LOD for mipmap sampling.
 * @param dout      Host output buffer for storing the resulting uchar4 image.
 */
static void get_mipmap(cudaMipmappedArray_t mm_array, const int3 img_size, const float scale, uchar4* dout) {
	const int width = img_size.x;
	const int height = img_size.y;
	const int n_level = img_size.z;
	const int asize = width * height;

	// Set up the texture resource description for the mipmapped array.
	cudaResourceDesc texResrc = {};
	texResrc.resType = cudaResourceTypeMipmappedArray;
	texResrc.res.mipmap.mipmap = mm_array;

	// Configure texture description using button_State to control filter modes.
	cudaTextureDesc texDescr = {};
	texDescr.normalizedCoords = 1;
	texDescr.filterMode = button_State[0] ? cudaFilterModeLinear : cudaFilterModePoint;
	texDescr.mipmapFilterMode = button_State[1] ? cudaFilterModeLinear : cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.maxMipmapLevelClamp = float(n_level - 1);
	texDescr.readMode = button_State[2] ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
	texDescr.disableTrilinearOptimization = button_State[3];

	// Create the texture object.
	cudaTextureObject_t texEngine;
	checkCudaErrors(cudaCreateTextureObject(&texEngine, &texResrc, &texDescr, NULL));

	// Allocate device memory for the output buffer.
	uchar4* d_out;
	checkCudaErrors(cudaMalloc(&d_out, asize * sizeof(uchar4)));

	// Define kernel launch configuration.
	dim3 blocksize(16, 16, 1);
	dim3 gridsize((width + blocksize.x - 1) / blocksize.x,
		(height + blocksize.y - 1) / blocksize.y,
		1);

	// Launch the kernel to retrieve the mipmap with uniform LOD.
	d_get_mipmap << <gridsize, blocksize >> > (texEngine, width, height, scale, d_out);

	// Copy the result from device memory to the host output buffer.
	checkCudaErrors(cudaMemcpy(dout, d_out, asize * sizeof(uchar4), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// Clean up texture object and device output memory.
	checkCudaErrors(cudaDestroyTextureObject(texEngine));
	checkCudaErrors(cudaFree(d_out));
}

void filter_and_blend(const int width, const int height, const float scale, const float key_scale,
	const uchar4* mask_img,   // 4-channel segmentation mask (for mipmap/alpha generation)
	const uchar4* glow_img,   // Glow (highlighted) image
	const uchar4* base_img,   // Original frame (base image for blending)
	uchar4* output_img)       // Blended output image (RGBA)
{
	size_t numBytes = width * height * sizeof(uchar4);

	// Allocate device memory for mask, glow, base, and output images.
	uchar4* d_mask = nullptr;
	uchar4* d_glow = nullptr;
	uchar4* d_base = nullptr;
	uchar4* d_output = nullptr;
	checkCudaErrors(cudaMalloc(&d_mask, numBytes));
	checkCudaErrors(cudaMalloc(&d_glow, numBytes));
	checkCudaErrors(cudaMalloc(&d_base, numBytes));
	checkCudaErrors(cudaMalloc(&d_output, numBytes));

	// Copy host images to device.
	checkCudaErrors(cudaMemcpy(d_mask, mask_img, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_glow, glow_img, numBytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_base, base_img, numBytes, cudaMemcpyHostToDevice));

	// Determine the number of mipmap levels.
	int n_level = 0, level = max(width, height);
	while (level) {
		level >>= 1;
		n_level++;
	}

	cudaExtent img_size = { static_cast<size_t>(width), static_cast<size_t>(height), 0 };
	cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();
	cudaMipmappedArray_t mm_array;
	checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));

	// Copy d_mask into level 0 of the mipmapped array.
	cudaArray_t level0;
	checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));
	cudaMemcpy3DParms cpy_param = { 0 };
	cpy_param.srcPtr = make_cudaPitchedPtr(d_mask, width * sizeof(uchar4), width, height);
	cpy_param.dstArray = level0;
	cpy_param.extent = img_size;
	cpy_param.extent.depth = 1;
	cpy_param.kind = cudaMemcpyDeviceToDevice;
	checkCudaErrors(cudaMemcpy3D(&cpy_param));

	// Generate all mipmap levels.
	gen_mipmap(mm_array, img_size);

	// Create a texture object from the mipmapped array.
	cudaResourceDesc texResrc;
	memset(&texResrc, 0, sizeof(cudaResourceDesc));
	texResrc.resType = cudaResourceTypeMipmappedArray;
	texResrc.res.mipmap.mipmap = mm_array;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.mipmapFilterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.maxMipmapLevelClamp = float(n_level - 1);
	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaTextureObject_t mip_tex;
	checkCudaErrors(cudaCreateTextureObject(&mip_tex, &texResrc, &texDescr, NULL));

	// Launch the unified post-processing kernel.
	dim3 blockSize(16, 16, 1);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y,
		1);
	post_processing_kernel << <gridSize, blockSize >> > (d_base, d_glow, mip_tex, d_output, width, height, scale, key_scale);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	// Copy the result back to host memory.
	checkCudaErrors(cudaMemcpy(output_img, d_output, numBytes, cudaMemcpyDeviceToHost));

	// Clean up resources.
	checkCudaErrors(cudaDestroyTextureObject(mip_tex));
	checkCudaErrors(cudaFreeMipmappedArray(mm_array));
	cudaFree(d_mask);
	cudaFree(d_glow);
	cudaFree(d_base);
	cudaFree(d_output);
}

/**
 * @brief Filters an image by generating mipmap levels and retrieving a blurred version.
 *
 * This function generates mipmap levels from an input image stored in host memory,
 * applies a blur effect via mipmapping, and retrieves the result into a host output buffer.
 *
 * @param width   Width of the input image.
 * @param height  Height of the input image.
 * @param scale   Scale factor used for the blur effect.
 * @param src_img Pointer to the input image data in host memory.
 * @param dst_img Pointer to the output image data in host memory.
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img) {
	// Calculate the number of mipmap levels based on the maximum dimension.
	int n_level = 0;
	int level = max(height, width);
	while (level) {
		level >>= 1;
		n_level++;
	}

	// Define the image extent and channel format.
	cudaExtent img_size = { static_cast<size_t>(width), static_cast<size_t>(height), 0 };
	cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();

	// Allocate a CUDA mipmapped array.
	cudaMipmappedArray_t mm_array;
	checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));

	// Get the first mipmap level (level 0) and copy the input image data.
	cudaArray_t level0;
	checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));
	cudaMemcpy3DParms cpy_param = {};
	cpy_param.srcPtr = make_cudaPitchedPtr((void*)src_img, width * sizeof(uchar4), width, height);
	cpy_param.dstArray = level0;
	cpy_param.extent = img_size;
	cpy_param.extent.depth = 1;
	cpy_param.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&cpy_param));

	// Generate all subsequent mipmap levels.
	gen_mipmap(mm_array, img_size);

	// Retrieve the filtered (blurred) mipmap image from the mipmapped array.
	get_mipmap(mm_array, make_int3(width, height, n_level), scale, dst_img);

	// Free the allocated CUDA mipmapped array.
	checkCudaErrors(cudaFreeMipmappedArray(mm_array));
}

///////////////////////////////////////////////////////////////////////////
// Asynchronous Functions Are Created Below
///////////////////////////////////////////////////////////////////////////

/**
 * @brief Asynchronously generates all mipmap levels for a given mipmapped array.
 *
 * This function loops through the mipmap levels and launches the mipmap generation kernel
 * on the provided CUDA stream. It reads from the current level and writes to the next level.
 *
 * @param mipmapArray The CUDA mipmapped array allocated for the image.
 * @param size The extent (width and height) of the highest resolution level.
 * @param stream The CUDA stream on which to launch the kernels asynchronously.
 */
static void gen_mipmap_async(cudaMipmappedArray_t& mipmapArray, cudaExtent size, cudaStream_t stream) {
	size_t width = size.width;
	size_t height = size.height;
	uint level = 0;
	while (width != 1 || height != 1) {
		width = std::max((size_t)1, width / 2);
		height = std::max((size_t)1, height / 2);

		// Retrieve current (levelFrom) and next (levelTo) mipmap levels.
		cudaArray_t levelFrom;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		cudaArray_t levelTo;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

		// Sanity check: verify that the next level has the expected dimensions.
		cudaExtent levelToSize;
		checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		assert(levelToSize.width == width);
		assert(levelToSize.height == height);

		// Set up a texture object for reading from the current level.
		cudaResourceDesc texResrc = {};
		texResrc.resType = cudaResourceTypeArray;
		texResrc.res.array.array = levelFrom;
		cudaTextureDesc texDescr = {};
		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeNormalizedFloat;

		cudaTextureObject_t texInput;
		checkCudaErrors(cudaCreateTextureObject(&texInput, &texResrc, &texDescr, NULL));

		// Set up a surface object for writing to the next level.
		cudaResourceDesc surfRes = {};
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = levelTo;
		cudaSurfaceObject_t surfOutput;
		checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

		// Configure kernel launch parameters.
		dim3 blockSize(16, 16, 1);
		dim3 gridSize((uint(width) + blockSize.x - 1) / blockSize.x,
			(uint(height) + blockSize.y - 1) / blockSize.y,
			1);

		// Launch the mipmap generation kernel asynchronously on the provided stream.
		d_gen_mipmap << <gridSize, blockSize, 0, stream >> > (surfOutput, texInput, (uint)width, (uint)height);
		checkCudaErrors(cudaGetLastError());

		// Clean up the texture and surface objects.
		checkCudaErrors(cudaDestroySurfaceObject(surfOutput));
		checkCudaErrors(cudaDestroyTextureObject(texInput));

		level++;
	}
}

/**
 * @brief Asynchronously retrieves a mipmapped image using texture sampling.
 *
 * This function creates a texture object from the given mipmapped array and launches a kernel
 * to sample the texture using a uniform level-of-detail (LOD) computed from the scale factor.
 * The output is copied back to host memory asynchronously.
 *
 * @param mm_array The CUDA mipmapped array containing the image and its mipmap levels.
 * @param img_size An int3 representing the original image width, height, and number of mipmap levels.
 * @param scale The scale factor used to compute the uniform LOD.
 * @param dout Pointer to host memory where the resulting uchar4 image will be copied.
 * @param stream The CUDA stream on which to perform the operations.
 */
static void get_mipmap_async(cudaMipmappedArray_t mm_array, const int3 img_size, const float scale, uchar4* dout, cudaStream_t stream) {
	const int width = img_size.x;
	const int height = img_size.y;
	const int n_level = img_size.z;
	const int asize = width * height;

	// Set up the texture resource description for the mipmapped array.
	cudaResourceDesc texResrc = {};
	texResrc.resType = cudaResourceTypeMipmappedArray;
	texResrc.res.mipmap.mipmap = mm_array;

	// Configure the texture description.
	cudaTextureDesc texDescr = {};
	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.mipmapFilterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.maxMipmapLevelClamp = float(n_level - 1);
	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaTextureObject_t texEngine;
	checkCudaErrors(cudaCreateTextureObject(&texEngine, &texResrc, &texDescr, NULL));

	// Allocate device memory for the output.
	uchar4* d_out;
	checkCudaErrors(cudaMalloc(&d_out, asize * sizeof(uchar4)));

	// Configure kernel launch parameters.
	dim3 blockSize(16, 16, 1);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
		(height + blockSize.y - 1) / blockSize.y,
		1);

	// Launch the mipmap sampling kernel asynchronously.
	d_get_mipmap << <gridSize, blockSize, 0, stream >> > (texEngine, width, height, scale, d_out);
	checkCudaErrors(cudaGetLastError());

	// Asynchronously copy the output from device to host.
	checkCudaErrors(cudaMemcpyAsync(dout, d_out, asize * sizeof(uchar4), cudaMemcpyDeviceToHost, stream));

	// Clean up resources.
	checkCudaErrors(cudaDestroyTextureObject(texEngine));
	checkCudaErrors(cudaFree(d_out));
}

/**
 * @brief Asynchronously applies a mipmap filter to an image.
 *
 * This function combines asynchronous memory transfer, mipmap generation, and retrieval. It first
 * allocates a CUDA mipmapped array, copies the source image (src_img) into level 0 of the array
 * asynchronously, then generates all subsequent mipmap levels, and finally retrieves the filtered
 * result back into host memory (dst_img). All operations are performed on the provided CUDA stream.
 *
 * @param width The width of the input image.
 * @param height The height of the input image.
 * @param scale The scale factor for mipmap sampling.
 * @param src_img Pointer to the source image data in host memory (RGBA format as uchar4).
 * @param dst_img Pointer to the host memory buffer where the output image will be stored.
 * @param stream The CUDA stream to use for asynchronous operations.
 */
void filter_mipmap_async(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img, cudaStream_t stream) {
	// Determine the number of mipmap levels.
	int n_level = 0;
	int level = std::max(height, width);
	while (level) {
		level >>= 1;
		n_level++;
	}

	cudaExtent img_size = { static_cast<size_t>(width), static_cast<size_t>(height), 0 };
	cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();

	// Allocate a CUDA mipmapped array.
	cudaMipmappedArray_t mm_array;
	checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));

	// Copy the source image into level 0 of the mipmapped array asynchronously.
	cudaArray_t level0;
	checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));
	cudaMemcpy3DParms cpy_param = {};
	cpy_param.srcPtr = make_cudaPitchedPtr((void*)src_img, width * sizeof(uchar4), width, height);
	cpy_param.dstArray = level0;
	cpy_param.extent = img_size;
	cpy_param.extent.depth = 1;
	cpy_param.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3DAsync(&cpy_param, stream));

	// Generate mipmap levels asynchronously.
	gen_mipmap_async(mm_array, img_size, stream);

	// Retrieve the final processed mipmap output asynchronously.
	get_mipmap_async(mm_array, make_int3(width, height, n_level), scale, dst_img, stream);

	// Free the allocated mipmapped array.
	checkCudaErrors(cudaFreeMipmappedArray(mm_array));
}