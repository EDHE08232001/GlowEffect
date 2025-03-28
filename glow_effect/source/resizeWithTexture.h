// resizeWithTexture.h
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void resizeWithTexture(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output,
                       int outWidth, int outHeight, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
