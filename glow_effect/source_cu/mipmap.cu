/*******************************************************************************************************************
 * FILE NAME   :    mipmap.cu
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    mipmap genernator and access
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 10      Yu Liu          Creation
 * 2022 OCT 26      Yu Liu          Moved V-shaped curve into cuda
 * 2022 OCT 27      Yu Liu          Proved texReadMode = cudaReadModeNormalizedFloat to be a must for linear filter
 *                                  also corrected phase shift by using x+1.f/y+1.f rather than x+0.5/y+0.5
 *
 ********************************************************************************************************************/
#include "old_movies.cuh"
extern bool button_State[5];

/**
 * CUDA kernel to generate a mipmap level by downscaling an input texture.
 *
 * This kernel performs 2x2 averaging to create a lower resolution mipmap
 * from the provided input texture and writes the results to an output surface.
 *
 * @param mipOutput  CUDA surface object for the output mipmap.
 * @param mipInput   CUDA texture object for the input image.
 * @param imageW     Width of the output mipmap level.
 * @param imageH     Height of the output mipmap level.
 */
__global__ void d_gen_mipmap(
	cudaSurfaceObject_t mipOutput,
	cudaTextureObject_t mipInput,
	uint imageW,
	uint imageH
) {
	// Compute the thread's x and y coordinates within the grid.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	// Compute the normalized pixel width and height.
	float px = 1.0f / static_cast<float>(imageW);
	float py = 1.0f / static_cast<float>(imageH);

	// Ensure the thread operates only within valid image dimensions.
	if ((x < imageW) && (y < imageH)) {
		// Fetch and average the colors of 4 neighboring texels.
		// tex2D samples texture at normalized coordinates [0,1] range.
		float4 color =
			tex2D<float4>(mipInput, (x + 0.0f) * px, (y + 0.0f) * py) +
			tex2D<float4>(mipInput, (x + 1.0f) * px, (y + 0.0f) * py) +
			tex2D<float4>(mipInput, (x + 1.0f) * px, (y + 1.0f) * py) +
			tex2D<float4>(mipInput, (x + 0.0f) * px, (y + 1.0f) * py);

		// Compute the average color.
		color /= 4.0f;
		color *= 255.0f; // Convert to [0,255] range.

		// Clamp values to ensure they do not exceed 255.
		color = fminf(color, make_float4(255.0f));

		// Convert to uchar4 format and write the final pixel to the output surface.
		surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
	}
}

/**
 * @brief Generates a mipmap chain for a given CUDA mipmapped array.
 *
 * This function generates all levels of a mipmapped array by downsampling the higher-level images.
 * Each mipmap level is created by halving the dimensions of the previous level until the dimensions are reduced to 1x1.
 *
 * @param mipmapArray Reference to the CUDA mipmapped array to process.
 * @param size Initial size (extent) of the highest resolution mipmap level.
 */
static void gen_mipmap(cudaMipmappedArray_t& mipmapArray, cudaExtent size) {
	// Initialize the width and height from the size extent.
	size_t width = size.width;
	size_t height = size.height;

	uint level = 0; // Mipmap level counter.

	// Iterate until the dimensions are reduced to 1x1.
	while (width != 1 || height != 1) {
		// Compute the dimensions of the next mipmap level.
		width = MAX((size_t)1, width / 2);
		height = MAX((size_t)1, height / 2);

		// Retrieve the current and next mipmap levels.
		cudaArray_t levelFrom;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		cudaArray_t levelTo;
		checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

		// Verify the dimensions of the next level.
		cudaExtent levelToSize;
		checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		assert(levelToSize.width == width);
		assert(levelToSize.height == height);
		assert(levelToSize.depth == 0);

		// Create a texture object for reading from the current level.
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

		// Create a surface object for writing to the next level.
		cudaSurfaceObject_t surfOutput;
		cudaResourceDesc surfRes = {};
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = levelTo;

		checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

		// Configure kernel launch parameters.
		dim3 blockSize(16, 16, 1);
		dim3 gridSize((uint(width) + blockSize.x - 1) / blockSize.x, (uint(height) + blockSize.y - 1) / blockSize.y, 1);

		// Launch the mipmap generation kernel.
		d_gen_mipmap << <gridSize, blockSize >> > (surfOutput, texInput, (uint)width, (uint)height);

		// Synchronize and check for errors.
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaGetLastError());

		// Destroy the surface and texture objects.
		checkCudaErrors(cudaDestroySurfaceObject(surfOutput));
		checkCudaErrors(cudaDestroyTextureObject(texInput));

		// Increment the mipmap level.
		level++;
	}
}


/**
 * @brief Kernel to sample a mipmapped texture with varying LOD per pixel.
 *
 * This kernel samples a texture using a specified LOD for each pixel, calculates the color data,
 * and writes it to the output buffer.
 *
 * @param texEngine CUDA texture object for the mipmapped texture.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param lod Pointer to an array of LOD values for each pixel.
 * @param dout Output buffer to store the resulting uchar4 color values.
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
		// Sample the texture with per-pixel LOD.
		float4 data = tex2DLod<float4>(texEngine, u, v, lod[idx], &state);

		// Convert the sampled color to uchar4 and write to the output buffer.
		dout[idx] = to_uchar4(255.0f * data);
	}
}

/**
 * @brief Kernel to sample a mipmapped texture with a uniform LOD.
 *
 * This kernel samples a texture using a single LOD for all pixels, calculates the color data,
 * and writes it to the output buffer.
 *
 * @param texEngine CUDA texture object for the mipmapped texture.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param scale Scale factor used to compute the LOD.
 * @param dout Output buffer to store the resulting uchar4 color values.
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

	// Calculate the uniform LOD based on the scale.
	float lod = log2(scale);

	if (xi < width && yi < height) {
		// Sample the texture with a uniform LOD.
		float4 data = tex2DLod<float4>(texEngine, u, v, lod);

		// Convert the sampled color to uchar4 and write to the output buffer.
		dout[idx] = to_uchar4(255.0f * data);
	}
}

/**
 * @brief Retrieves a mipmap image with a uniform blur applied using CUDA.
 *
 * This function retrieves the mipmapped image from a CUDA mipmapped array with uniform scaling.
 * It uses texture sampling and stores the result in a device buffer, which is later copied to the host.
 *
 * @param mm_array The CUDA mipmapped array containing the mipmap levels.
 * @param img_size Dimensions of the image and number of mipmap levels (int3: {width, height, n_level}).
 * @param scale Scale factor used to compute the LOD for mipmap sampling.
 * @param dout Host output buffer for storing the resulting uchar4 image.
 */
static void get_mipmap(cudaMipmappedArray_t mm_array, const int3 img_size, const float scale, uchar4* dout) {
	const int width = img_size.x;
	const int height = img_size.y;
	const int n_level = img_size.z;
	const int asize = width * height;

	// Initialize texture resource description for mipmapped array.
	cudaResourceDesc texResrc = {};
	texResrc.resType = cudaResourceTypeMipmappedArray;
	texResrc.res.mipmap.mipmap = mm_array;

	// Initialize texture description.
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

	// Create texture object for sampling mipmap levels.
	cudaTextureObject_t texEngine;
	checkCudaErrors(cudaCreateTextureObject(&texEngine, &texResrc, &texDescr, NULL));

	// Allocate memory for the device output buffer.
	uchar4* d_out;
	checkCudaErrors(cudaMalloc(&d_out, asize * sizeof(uchar4)));

	// Define kernel execution parameters.
	dim3 blocksize(16, 16, 1);
	dim3 gridsize((width + blocksize.x - 1) / blocksize.x, (height + blocksize.y - 1) / blocksize.y);

	// Launch the mipmap retrieval kernel.
	d_get_mipmap << <gridsize, blocksize >> > (texEngine, width, height, scale, d_out);

	// Copy the result from the device to the host.
	checkCudaErrors(cudaMemcpy(dout, d_out, asize * sizeof(uchar4), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	// Cleanup resources.
	checkCudaErrors(cudaDestroyTextureObject(texEngine));
	checkCudaErrors(cudaFree(d_out));
}

/**
 * @brief Filters an image by generating mipmap levels and retrieving a blurred version.
 *
 * This function generates mipmap levels for a given input image, applies the filtering process,
 * and retrieves the result as a blurred image at a specific scale.
 *
 * @param width Width of the input image.
 * @param height Height of the input image.
 * @param scale Scale factor used for the blur effect.
 * @param src_img Pointer to the input image on the host.
 * @param dst_img Pointer to the output image on the host.
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img) {
	// Calculate the number of mipmap levels based on the largest dimension.
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

	// Get the first mipmap level (level 0).
	cudaArray_t level0;
	checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));

	// Copy the input image data to the first mipmap level.
	cudaMemcpy3DParms cpy_param = {};
	cpy_param.srcPtr = make_cudaPitchedPtr((void*)src_img, width * sizeof(uchar4), width, height);
	cpy_param.dstArray = level0;
	cpy_param.extent = img_size;
	cpy_param.extent.depth = 1;
	cpy_param.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&cpy_param));

	// Generate mipmap levels from the input image.
	gen_mipmap(mm_array, img_size);

	// Retrieve the filtered mipmap image.
	get_mipmap(mm_array, make_int3(width, height, n_level), scale, dst_img);

	// Free the CUDA mipmapped array.
	checkCudaErrors(cudaFreeMipmappedArray(mm_array));
}
