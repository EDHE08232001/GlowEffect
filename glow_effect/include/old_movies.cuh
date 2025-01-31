/*******************************************************************************************************************
 * FILE NAME   :    old_movies.cuh
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    Includes all commonly used libraries and data types for CUDA-based image processing.
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * ^^^^^^^^^^^      ^^^^^^          ^^^^^^^^
 * 2022 OCT 10      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "device_launch_parameters.h"
#include "all_common.h"

 // Define an Image structure to store image data and associated CUDA objects.
#pragma pack(push,4)
struct Image
{
	void* h_data;                   // Pointer to host data
	cudaExtent size;                 // Image dimensions (width, height, depth)
	cudaResourceType type;           // Type of CUDA resource (e.g., array, buffer)
	cudaArray_t dataArray;           // CUDA array for storing image data
	cudaMipmappedArray_t mipmapArray;// CUDA mipmapped array
	cudaTextureObject_t textureObject; // CUDA texture object

	// Constructor initializes all members to zero
	Image()
	{
		memset(this, 0, sizeof(Image));
	}
};
#pragma pack(pop)

/**
 * Encodes a CUDA texture object into a uint2 structure for compact storage.
 *
 * @param obj CUDA texture object to encode.
 * @return Encoded uint2 representation of the texture object.
 */
__host__ __device__ __inline__ uint2 encodeTextureObject(cudaTextureObject_t obj)
{
	return make_uint2((uint)(obj & 0xFFFFFFFF), (uint)(obj >> 32));
}

/**
 * Decodes a uint2 structure back into a CUDA texture object.
 *
 * @param obj Encoded uint2 texture object representation.
 * @return Decoded CUDA texture object.
 */
__host__ __device__ __inline__ cudaTextureObject_t decodeTextureObject(uint2 obj)
{
	return (((cudaTextureObject_t)obj.x) | ((cudaTextureObject_t)obj.y) << 32);
}

/**
 * Converts an uchar4 vector to a float4 vector.
 *
 * @param vec Input uchar4 vector.
 * @return Converted float4 vector.
 */
__device__ __inline__ float4 to_float4(uchar4 vec)
{
	return make_float4(vec.x, vec.y, vec.z, vec.w);
}

/**
 * Converts a short4 vector to a float4 vector.
 *
 * @param vec Input short4 vector.
 * @return Converted float4 vector.
 */
__device__ __inline__ float4 to_float4(short4 vec)
{
	return make_float4(vec.x, vec.y, vec.z, vec.w);
}

/**
 * Converts a float4 vector to an uchar4 vector by truncating values to unsigned char range.
 *
 * @param vec Input float4 vector.
 * @return Converted uchar4 vector.
 */
__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
	typedef unsigned char uchar;
	return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

/**
 * Converts a float4 vector to a short4 vector by truncating values to short integer range.
 *
 * @param vec Input float4 vector.
 * @return Converted short4 vector.
 */
__device__ __inline__ short4 to_short4(float4 vec)
{
	return make_short4((short)vec.x, (short)vec.y, (short)vec.z, (short)vec.w);
}

/**
 * Explanation of CUDA Keywords:
 *
 * - __host__: Specifies that a function should run on the CPU (host) and be callable from host code.
 * - __device__: Specifies that a function should run on the GPU (device) and be callable from device code.
 * - __inline__: Suggests to the compiler that the function should be inlined to improve performance by avoiding function call overhead.
 *
 * Functions marked with both __host__ and __device__ can be executed on both CPU and GPU, making them useful for reusable logic.
 */


