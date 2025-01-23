/*******************************************************************************************************************
 * FILE NAME   :    test_learn.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    top entry calls for old movies
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 07      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#include "old_movies.hpp"
#include "old_movies.cuh"
#include "tools_cuda_dinfo.h"


void test_texture(void)
{
    cudaArray_t array;
    /*
     code=27(cudaErrorInvalidNormSetting) "cudaCreateTextureObject(&texEngine, &resDesc, &texDesc, NULL)"
     This indicates that an attempt was made to read a non-float texture as a normalized float.
     */
    //cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);
    checkCudaErrors(cudaMallocArray(&array, &desc, 1000, 500));
    cudaChannelFormatDesc chk_desc;
    cudaExtent chk_ext;
    uint chk_flag;
    cudaArrayGetInfo(&chk_desc, &chk_ext, &chk_flag, array);

    cudaResourceDesc resDesc;
    //memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.res.array.array = array;
    resDesc.resType = cudaResourceTypeArray;
    cudaTextureDesc texDesc;
    //memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.normalizedCoords = true;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;

    cudaTextureObject_t  texEngine;
    checkCudaErrors(cudaCreateTextureObject(&texEngine, &resDesc, &texDesc, NULL));
}



/*
const int rows = src_bgr.rows;
const int cols = src_bgr.cols;
const int chnl = src_bgr.channels();
const int stride = cols * chnl;
const int sect = cols / 8;
for (int iLoop = 0, m = 0; iLoop < rows; iLoop++, m += stride) {
    for (int jLoop = 0, n = m; jLoop < cols; jLoop++, n += chnl) {
        if (jLoop < sect)
            src_bgr.data[n + 0] = 255, src_bgr.data[n + 1] = 255, src_bgr.data[n + 2] = 255;
        else if (jLoop < 2 * sect)
            src_bgr.data[n + 0] = 0, src_bgr.data[n + 1] = 0, src_bgr.data[n + 2] = 255;
        else if (jLoop < 3 * sect)
            src_bgr.data[n + 0] = 0, src_bgr.data[n + 1] = 255, src_bgr.data[n + 2] = 255;
        else if (jLoop < 4 * sect)
            src_bgr.data[n + 0] = 255, src_bgr.data[n + 1] = 255, src_bgr.data[n + 2] = 0;
        else if (jLoop < 5 * sect)
            src_bgr.data[n + 0] = 0, src_bgr.data[n + 1] = 255, src_bgr.data[n + 2] = 0;
        else if (jLoop < 6 * sect)
            src_bgr.data[n + 0] = 255, src_bgr.data[n + 1] = 0, src_bgr.data[n + 2] = 255;
        else if (jLoop < 7 * sect)
            src_bgr.data[n + 0] = 255, src_bgr.data[n + 1] = 0, src_bgr.data[n + 2] = 0;
        else
            src_bgr.data[n + 0] = 64, src_bgr.data[n + 1] = 64, src_bgr.data[n + 2] = 64;
    }
}
*/

