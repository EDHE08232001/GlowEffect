/*******************************************************************************************************************
 * FILE NAME   :    old_movies.cuh
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    includes all common used libs and data type
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


 //
#pragma pack(push,4)
struct Image
{
    void* h_data;
    cudaExtent              size;
    cudaResourceType        type;
    cudaArray_t             dataArray;
    cudaMipmappedArray_t    mipmapArray;
    cudaTextureObject_t     textureObject;

    Image()
    {
        memset(this, 0, sizeof(Image));
    }
};
#pragma pack(pop)


__host__ __device__ __inline__ uint2 encodeTextureObject(cudaTextureObject_t obj)
{
    return make_uint2((uint)(obj & 0xFFFFFFFF), (uint)(obj >> 32));
}

__host__ __device__ __inline__ cudaTextureObject_t decodeTextureObject(uint2 obj)
{
    return (((cudaTextureObject_t)obj.x) | ((cudaTextureObject_t)obj.y) << 32);
}

__device__ __inline__ float4 to_float4(uchar4 vec)
{
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

__device__ __inline__ float4 to_float4(short4 vec)
{
    return make_float4(vec.x, vec.y, vec.z, vec.w);
}

__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    typedef unsigned char uchar;
    return make_uchar4((uchar)vec.x, (uchar)vec.y, (uchar)vec.z, (uchar)vec.w);
}

__device__ __inline__ short4 to_short4(float4 vec)
{
    return make_short4((short)vec.x, (short)vec.y, (short)vec.z, (short)vec.w);
}
