/*******************************************************************************************************************
 * FILE NAME   : mipmap.cu
 *
 * DESCRIPTION : Implements CUDA kernels and host functions for generating mipmap levels,
 *               retrieving mipmapped data, and applying blur effects via mipmapping.
 *
 * VERSION HISTORY
 * 2022 OCT 10      Yu Liu          Creation
 * 2022 OCT 26      Yu Liu          Moved V-shaped curve into CUDA
 * 2022 OCT 27      Yu Liu          Required texReadMode = cudaReadModeNormalizedFloat for linear filtering;
 *                                  Corrected phase shift in texture coordinates
 * 2025 APR 15      Modified        Improved documentation and formatting
 *
 ********************************************************************************************************************/

 #include "mipmap.h"
 #include "old_movies.cuh"  // For utility functions like checkCudaErrors and to_uchar4
 
 // External state parameters controlling texture filtering modes
 extern bool button_State[5];
 
 //------------------------------------------------------------------------------
 // CUDA Kernel Functions
 //------------------------------------------------------------------------------
 
 /**
  * @brief CUDA kernel to generate a mipmap level by downscaling an input texture.
  *
  * Performs 2x2 averaging of texels from the input texture to create a
  * lower resolution mipmap level, converting results to the [0,255] range.
  *
  * @param mipOutput  CUDA surface object for writing the output mipmap level.
  * @param mipInput   CUDA texture object for reading the input image.
  * @param imageW     Width of the output mipmap level.
  * @param imageH     Height of the output mipmap level.
  */
 __global__ void d_gen_mipmap(
	 cudaSurfaceObject_t mipOutput,
	 cudaTextureObject_t mipInput,
	 uint imageW,
	 uint imageH
 ) {
	 // Compute thread coordinates in output image
	 uint x = blockIdx.x * blockDim.x + threadIdx.x;
	 uint y = blockIdx.y * blockDim.y + threadIdx.y;
 
	 // Compute normalized pixel size
	 float px = 1.0f / static_cast<float>(imageW);
	 float py = 1.0f / static_cast<float>(imageH);
 
	 // Ensure thread operates only within valid image dimensions
	 if ((x < imageW) && (y < imageH)) {
		 // Average 4 neighboring texels using normalized texture access
		 float4 color =
			 tex2D<float4>(mipInput, (x + 0.0f) * px, (y + 0.0f) * py) +
			 tex2D<float4>(mipInput, (x + 1.0f) * px, (y + 0.0f) * py) +
			 tex2D<float4>(mipInput, (x + 1.0f) * px, (y + 1.0f) * py) +
			 tex2D<float4>(mipInput, (x + 0.0f) * px, (y + 1.0f) * py);
 
		 // Average color and scale to [0, 255] range
		 color /= 4.0f;
		 color *= 255.0f;
 
		 // Clamp color values
		 color = fminf(color, make_float4(255.0f));
 
		 // Write resulting color to output surface
		 surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
	 }
 }
 
 /**
  * @brief CUDA kernel to sample a mipmapped texture with per-pixel LOD.
  *
  * @param texEngine CUDA texture object for the mipmapped texture.
  * @param width     Width of the output image.
  * @param height    Height of the output image.
  * @param lod       Pointer to array of per-pixel LOD values.
  * @param dout      Output buffer for resulting uchar4 color values.
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
		 // Sample texture using per-pixel LOD
		 float4 data = tex2DLod<float4>(texEngine, u, v, lod[idx], &state);
		 // Convert to uchar4 and write to output
		 dout[idx] = to_uchar4(255.0f * data);
	 }
 }
 
 /**
  * @brief CUDA kernel to sample a mipmapped texture with uniform LOD.
  *
  * @param texEngine CUDA texture object for the mipmapped texture.
  * @param width     Width of the output image.
  * @param height    Height of the output image.
  * @param scale     Scale factor used to compute the uniform LOD.
  * @param dout      Output buffer for resulting uchar4 color values.
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
 
	 // Compute uniform LOD based on scale factor
	 float lod = log2(scale);
 
	 if (xi < width && yi < height) {
		 // Sample texture using uniform LOD
		 float4 data = tex2DLod<float4>(texEngine, u, v, lod);
		 // Convert to uchar4 and write to output
		 dout[idx] = to_uchar4(255.0f * data);
	 }
 }
 
 /**
  * @brief CUDA kernel for post-processing with glow effect.
  * 
  * Samples the mipmapped texture to compute per-pixel alpha values,
  * applies a key scale factor to boost the glow effect, then blends
  * the base image with the glow image.
  *
  * @param base_img  Original frame (base image for blending).
  * @param glow_img  Glow (highlighted) image to blend.
  * @param mip_tex   CUDA texture object for the mipmapped mask texture.
  * @param output    Output buffer for the blended result.
  * @param width     Width of the images.
  * @param height    Height of the images.
  * @param scale     Scale factor for controlling blur spread.
  * @param key_scale Multiplier for boosting glow intensity.
  */
 __global__ void post_processing_kernel(
	 const uchar4* base_img,
	 const uchar4* glow_img,
	 cudaTextureObject_t mip_tex,
	 uchar4* output,
	 int width,
	 int height,
	 float scale,
	 float key_scale
 ) {
	 int x = blockIdx.x * blockDim.x + threadIdx.x;
	 int y = blockIdx.y * blockDim.y + threadIdx.y;
	 
	 if (x < width && y < height) {
		 int idx = y * width + x;
 
		 // Compute normalized texture coordinates
		 float u = (x + 0.5f) / (float)width;
		 float v = (y + 0.5f) / (float)height;
 
		 // Compute LOD (level-of-detail) based on position and scale
		 float lod = log2(fabs(u - 0.5f) * fabs(v - 0.5f) * scale + 1.0f);
 
		 // Sample the mipmapped texture from the mask
		 float4 mip_val = tex2DLod<float4>(mip_tex, u, v, lod);
		 
		 // Get alpha value from red channel and normalize to [0,1]
		 float alpha = fminf(mip_val.x / 255.0f, 1.0f);
		 
		 // Apply key scale factor (boost glow intensity)
		 alpha = fminf(alpha * key_scale, 1.0f);
 
		 // Retrieve base and glow pixels
		 uchar4 base_pixel = base_img[idx];
		 uchar4 glow_pixel = glow_img[idx];
 
		 // Blend: output = alpha * glow + (1 - alpha) * base
		 uchar4 blended;
		 blended.x = (unsigned char)(alpha * glow_pixel.x + (1.0f - alpha) * base_pixel.x);
		 blended.y = (unsigned char)(alpha * glow_pixel.y + (1.0f - alpha) * base_pixel.y);
		 blended.z = (unsigned char)(alpha * glow_pixel.z + (1.0f - alpha) * base_pixel.z);
		 blended.w = 255; // Fully opaque
		 
		 output[idx] = blended;
	 }
 }
 
 //------------------------------------------------------------------------------
 // Helper Functions (Internal)
 //------------------------------------------------------------------------------
 
 /**
  * @brief Generates a complete mipmap chain from a CUDA mipmapped array.
  *
  * Creates successive mipmap levels by halving dimensions until reaching 1x1.
  *
  * @param mipmapArray Reference to the CUDA mipmapped array.
  * @param size        Extent (width and height) of highest resolution mipmap level.
  */
 static void gen_mipmap(cudaMipmappedArray_t& mipmapArray, cudaExtent size) {
	 // Initialize dimensions
	 size_t width = size.width;
	 size_t height = size.height;
	 uint level = 0;  // Mipmap level counter
 
	 // Continue until reaching 1x1
	 while (width != 1 || height != 1) {
		 // Compute next level dimensions
		 width = MAX((size_t)1, width / 2);
		 height = MAX((size_t)1, height / 2);
 
		 // Get current and next mipmap level arrays
		 cudaArray_t levelFrom;
		 checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		 cudaArray_t levelTo;
		 checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));
 
		 // Verify next level has correct dimensions
		 cudaExtent levelToSize;
		 checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		 assert(levelToSize.width == width);
		 assert(levelToSize.height == height);
		 assert(levelToSize.depth == 0);
 
		 // Create texture object for reading from current level
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
 
		 cudaTextureObject_t texInput;
		 checkCudaErrors(cudaCreateTextureObject(&texInput, &texResrc, &texDescr, NULL));
 
		 // Create surface object for writing to next level
		 cudaResourceDesc surfRes = {};
		 surfRes.resType = cudaResourceTypeArray;
		 surfRes.res.array.array = levelTo;
 
		 cudaSurfaceObject_t surfOutput;
		 checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));
 
		 // Configure kernel launch
		 dim3 blockSize(16, 16, 1);
		 dim3 gridSize(
			 (uint(width) + blockSize.x - 1) / blockSize.x,
			 (uint(height) + blockSize.y - 1) / blockSize.y,
			 1
		 );
 
		 // Launch kernel to generate current mipmap level
		 d_gen_mipmap<<<gridSize, blockSize>>>(
			 surfOutput, texInput, (uint)width, (uint)height
		 );
 
		 // Synchronize and check for errors
		 checkCudaErrors(cudaDeviceSynchronize());
		 checkCudaErrors(cudaGetLastError());
 
		 // Clean up resources
		 checkCudaErrors(cudaDestroySurfaceObject(surfOutput));
		 checkCudaErrors(cudaDestroyTextureObject(texInput));
 
		 level++;
	 }
 }
 
 /**
  * @brief Retrieves a mipmap image with uniform blur.
  *
  * @param mm_array  CUDA mipmapped array containing the mipmap levels.
  * @param img_size  Dimensions and mipmap levels (int3: {width, height, n_level}).
  * @param scale     Scale factor for LOD computation.
  * @param dout      Host output buffer for the result.
  */
 static void get_mipmap(
	 cudaMipmappedArray_t mm_array,
	 const int3 img_size,
	 const float scale,
	 uchar4* dout
 ) {
	 const int width = img_size.x;
	 const int height = img_size.y;
	 const int n_level = img_size.z;
	 const int asize = width * height;
 
	 // Set up texture resource description
	 cudaResourceDesc texResrc = {};
	 texResrc.resType = cudaResourceTypeMipmappedArray;
	 texResrc.res.mipmap.mipmap = mm_array;
 
	 // Configure texture description using button_State to control filter modes
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
 
	 // Create texture object
	 cudaTextureObject_t texEngine;
	 checkCudaErrors(cudaCreateTextureObject(&texEngine, &texResrc, &texDescr, NULL));
 
	 // Allocate device memory for output
	 uchar4* d_out;
	 checkCudaErrors(cudaMalloc(&d_out, asize * sizeof(uchar4)));
 
	 // Define kernel launch configuration
	 dim3 blocksize(16, 16, 1);
	 dim3 gridsize(
		 (width + blocksize.x - 1) / blocksize.x,
		 (height + blocksize.y - 1) / blocksize.y,
		 1
	 );
 
	 // Launch kernel
	 d_get_mipmap<<<gridsize, blocksize>>>(texEngine, width, height, scale, d_out);
 
	 // Copy result to host
	 checkCudaErrors(cudaMemcpy(dout, d_out, asize * sizeof(uchar4), cudaMemcpyDeviceToHost));
	 cudaDeviceSynchronize();
 
	 // Clean up resources
	 checkCudaErrors(cudaDestroyTextureObject(texEngine));
	 checkCudaErrors(cudaFree(d_out));
 }
 
 /**
  * @brief Asynchronously generates mipmap levels.
  *
  * @param mipmapArray The CUDA mipmapped array.
  * @param size        Extent of highest resolution level.
  * @param stream      CUDA stream for asynchronous operations.
  */
 static void gen_mipmap_async(
	 cudaMipmappedArray_t& mipmapArray,
	 cudaExtent size,
	 cudaStream_t stream
 ) {
	 size_t width = size.width;
	 size_t height = size.height;
	 uint level = 0;
	 
	 while (width != 1 || height != 1) {
		 width = std::max((size_t)1, width / 2);
		 height = std::max((size_t)1, height / 2);
 
		 // Get current and next mipmap level arrays
		 cudaArray_t levelFrom;
		 checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
		 cudaArray_t levelTo;
		 checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));
 
		 // Verify dimensions
		 cudaExtent levelToSize;
		 checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
		 assert(levelToSize.width == width);
		 assert(levelToSize.height == height);
 
		 // Create texture object for reading from current level
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
 
		 // Create surface object for writing to next level
		 cudaResourceDesc surfRes = {};
		 surfRes.resType = cudaResourceTypeArray;
		 surfRes.res.array.array = levelTo;
		 
		 cudaSurfaceObject_t surfOutput;
		 checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));
 
		 // Configure kernel launch
		 dim3 blockSize(16, 16, 1);
		 dim3 gridSize(
			 (uint(width) + blockSize.x - 1) / blockSize.x,
			 (uint(height) + blockSize.y - 1) / blockSize.y,
			 1
		 );
 
		 // Launch kernel asynchronously
		 d_gen_mipmap<<<gridSize, blockSize, 0, stream>>>(
			 surfOutput, texInput, (uint)width, (uint)height
		 );
		 checkCudaErrors(cudaGetLastError());
 
		 // Clean up resources
		 checkCudaErrors(cudaDestroySurfaceObject(surfOutput));
		 checkCudaErrors(cudaDestroyTextureObject(texInput));
 
		 level++;
	 }
 }
 
 /**
  * @brief Asynchronously retrieves a mipmapped image.
  *
  * @param mm_array  CUDA mipmapped array containing the mipmap levels.
  * @param img_size  Dimensions and mipmap levels (int3: {width, height, n_level}).
  * @param scale     Scale factor for LOD computation.
  * @param dout      Host output buffer for the result.
  * @param stream    CUDA stream for asynchronous operations.
  */
 static void get_mipmap_async(
	 cudaMipmappedArray_t mm_array,
	 const int3 img_size,
	 const float scale,
	 uchar4* dout,
	 cudaStream_t stream
 ) {
	 const int width = img_size.x;
	 const int height = img_size.y;
	 const int n_level = img_size.z;
	 const int asize = width * height;
 
	 // Set up texture resource description
	 cudaResourceDesc texResrc = {};
	 texResrc.resType = cudaResourceTypeMipmappedArray;
	 texResrc.res.mipmap.mipmap = mm_array;
 
	 // Configure texture description
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
 
	 // Allocate device memory for output
	 uchar4* d_out;
	 checkCudaErrors(cudaMalloc(&d_out, asize * sizeof(uchar4)));
 
	 // Configure kernel launch
	 dim3 blockSize(16, 16, 1);
	 dim3 gridSize(
		 (width + blockSize.x - 1) / blockSize.x,
		 (height + blockSize.y - 1) / blockSize.y,
		 1
	 );
 
	 // Launch kernel asynchronously
	 d_get_mipmap<<<gridSize, blockSize, 0, stream>>>(
		 texEngine, width, height, scale, d_out
	 );
	 checkCudaErrors(cudaGetLastError());
 
	 // Copy results asynchronously
	 checkCudaErrors(cudaMemcpyAsync(
		 dout, d_out, asize * sizeof(uchar4), cudaMemcpyDeviceToHost, stream
	 ));
 
	 // Clean up resources
	 checkCudaErrors(cudaDestroyTextureObject(texEngine));
	 checkCudaErrors(cudaFree(d_out));
 }
 
 //------------------------------------------------------------------------------
 // Public API Functions
 //------------------------------------------------------------------------------
 
 /**
  * @brief Filters and blends images using mipmap-based glow effect.
  *
  * See header file for detailed description.
  */
 void filter_and_blend(
	 const int width,
	 const int height,
	 const float scale,
	 const float key_scale,
	 const uchar4* mask_img,
	 const uchar4* glow_img,
	 const uchar4* base_img,
	 uchar4* output_img
 ) {
	 size_t numBytes = width * height * sizeof(uchar4);
 
	 // Allocate device memory
	 uchar4* d_mask = nullptr;
	 uchar4* d_glow = nullptr;
	 uchar4* d_base = nullptr;
	 uchar4* d_output = nullptr;
	 checkCudaErrors(cudaMalloc(&d_mask, numBytes));
	 checkCudaErrors(cudaMalloc(&d_glow, numBytes));
	 checkCudaErrors(cudaMalloc(&d_base, numBytes));
	 checkCudaErrors(cudaMalloc(&d_output, numBytes));
 
	 // Copy host images to device
	 checkCudaErrors(cudaMemcpy(d_mask, mask_img, numBytes, cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(d_glow, glow_img, numBytes, cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(d_base, base_img, numBytes, cudaMemcpyHostToDevice));
 
	 // Determine number of mipmap levels
	 int n_level = 0, level = max(width, height);
	 while (level) {
		 level >>= 1;
		 n_level++;
	 }
 
	 // Create mipmapped array
	 cudaExtent img_size = { 
		 static_cast<size_t>(width), 
		 static_cast<size_t>(height), 
		 0 
	 };
	 cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();
	 cudaMipmappedArray_t mm_array;
	 checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));
 
	 // Copy mask into level 0 of mipmapped array
	 cudaArray_t level0;
	 checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));
	 cudaMemcpy3DParms cpy_param = { 0 };
	 cpy_param.srcPtr = make_cudaPitchedPtr(d_mask, width * sizeof(uchar4), width, height);
	 cpy_param.dstArray = level0;
	 cpy_param.extent = img_size;
	 cpy_param.extent.depth = 1;
	 cpy_param.kind = cudaMemcpyDeviceToDevice;
	 checkCudaErrors(cudaMemcpy3D(&cpy_param));
 
	 // Generate all mipmap levels
	 gen_mipmap(mm_array, img_size);
 
	 // Create texture object
	 cudaResourceDesc texResrc = {};
	 texResrc.resType = cudaResourceTypeMipmappedArray;
	 texResrc.res.mipmap.mipmap = mm_array;
 
	 cudaTextureDesc texDescr = {};
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
 
	 // Launch post-processing kernel
	 dim3 blockSize(16, 16, 1);
	 dim3 gridSize(
		 (width + blockSize.x - 1) / blockSize.x,
		 (height + blockSize.y - 1) / blockSize.y,
		 1
	 );
	 
	 post_processing_kernel<<<gridSize, blockSize>>>(
		 d_base, d_glow, mip_tex, d_output, width, height, scale, key_scale
	 );
	 checkCudaErrors(cudaDeviceSynchronize());
	 checkCudaErrors(cudaGetLastError());
 
	 // Copy result back to host
	 checkCudaErrors(cudaMemcpy(output_img, d_output, numBytes, cudaMemcpyDeviceToHost));
 
	 // Clean up resources
	 checkCudaErrors(cudaDestroyTextureObject(mip_tex));
	 checkCudaErrors(cudaFreeMipmappedArray(mm_array));
	 cudaFree(d_mask);
	 cudaFree(d_glow);
	 cudaFree(d_base);
	 cudaFree(d_output);
 }
 
 /**
  * @brief Applies a mipmap-based blur filter to an image (synchronous version).
  *
  * See header file for detailed description.
  */
 void filter_mipmap(
	 const int width,
	 const int height,
	 const float scale,
	 const uchar4* src_img,
	 uchar4* dst_img
 ) {
	 // Calculate number of mipmap levels
	 int n_level = 0;
	 int level = max(height, width);
	 while (level) {
		 level >>= 1;
		 n_level++;
	 }
 
	 // Define image extent and channel format
	 cudaExtent img_size = { 
		 static_cast<size_t>(width), 
		 static_cast<size_t>(height), 
		 0 
	 };
	 cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();
 
	 // Allocate mipmapped array
	 cudaMipmappedArray_t mm_array;
	 checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));
 
	 // Get level 0 and copy input image data
	 cudaArray_t level0;
	 checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));
	 cudaMemcpy3DParms cpy_param = {};
	 cpy_param.srcPtr = make_cudaPitchedPtr((void*)src_img, width * sizeof(uchar4), width, height);
	 cpy_param.dstArray = level0;
	 cpy_param.extent = img_size;
	 cpy_param.extent.depth = 1;
	 cpy_param.kind = cudaMemcpyHostToDevice;
	 checkCudaErrors(cudaMemcpy3D(&cpy_param));
 
	 // Generate mipmap levels
	 gen_mipmap(mm_array, img_size);
 
	 // Retrieve filtered image from mipmapped array
	 get_mipmap(mm_array, make_int3(width, height, n_level), scale, dst_img);
 
	 // Free mipmapped array
	 checkCudaErrors(cudaFreeMipmappedArray(mm_array));
 }
 
 /**
  * @brief Asynchronously applies a mipmap filter to an image.
  * 
  * See header file for detailed description.
  */
 void filter_mipmap_async(
	 const int width,
	 const int height,
	 const float scale,
	 const uchar4* src_img,
	 uchar4* dst_img,
	 cudaStream_t stream
 ) {
	 // Calculate number of mipmap levels
	 int n_level = 0;
	 int level = std::max(height, width);
	 while (level) {
		 level >>= 1;
		 n_level++;
	 }
 
	 // Define image extent and channel format
	 cudaExtent img_size = { 
		 static_cast<size_t>(width), 
		 static_cast<size_t>(height), 
		 0 
	 };
	 cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();
 
	 // Allocate mipmapped array
	 cudaMipmappedArray_t mm_array;
	 checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));
 
	 // Get level 0 and copy input image data asynchronously
	 cudaArray_t level0;
	 checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));
	 cudaMemcpy3DParms cpy_param = {};
	 cpy_param.srcPtr = make_cudaPitchedPtr((void*)src_img, width * sizeof(uchar4), width, height);
	 cpy_param.dstArray = level0;
	 cpy_param.extent = img_size;
	 cpy_param.extent.depth = 1;
	 cpy_param.kind = cudaMemcpyHostToDevice;
	 checkCudaErrors(cudaMemcpy3DAsync(&cpy_param, stream));
 
	 // Generate mipmap levels asynchronously
	 gen_mipmap_async(mm_array, img_size, stream);
 
	 // Retrieve filtered image asynchronously
	 get_mipmap_async(mm_array, make_int3(width, height, n_level), scale, dst_img, stream);
 
	 // Free mipmapped array (note: this doesn't wait for stream operations to complete)
	 checkCudaErrors(cudaFreeMipmappedArray(mm_array));
 }