/*******************************************************************************************************************
 * FILE NAME   :    gaussian.cu
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    gaussian filter
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 26      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#include "old_movies.cuh"
#include "old_movies.hpp"
extern float knl_coeff[N_KERNEL][KNL_SIZE];


__device__ __constant__ float d_knl_coeff[N_KERNEL * KNL_SIZE];
__global__
void d_filter_gaussion(
    const bool hNv,
    const int width, const int height,
    const int n_knl, const int n_tap,
    const float* coef, uchar4* src_img, uchar4* dst_img
)
{
    const int range = hNv ? height : width;
    const int n_data = (hNv ? width : height);
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    uchar4 line_buf[4096]; // afford UHD size
    //uchar4* line_buf = new uchar4[n_data];
    if (line_buf == nullptr) {
        printf("heap overflow at %d\n", x);
        return;
    }

    if (x < range)
    {
        for (int i = 0; i < n_data; i++) {
            int idx = hNv ? (x * width + i) : (i * width + x);
            line_buf[i] = src_img[idx];
        }

        for (int i = 0; i < n_data; i++)
        {
            int knl_idx = i * n_knl / n_data;
            if (knl_idx >= n_knl)
                knl_idx = n_knl - 1;
            int bnd_idx = knl_idx * n_tap;

            float4 sum = to_float4(line_buf[i]) * d_knl_coeff[bnd_idx]; // coef[bnd_idx]; // 
            for (int j = 1, p = i - 1, q = i + 1; j < n_tap; j++, p--, q++)
            {
                float4 left_data = to_float4(line_buf[p < 0 ? 0 : p]);
                float4 rght_data = to_float4(line_buf[q >= n_data ? n_data - 1 : q]);
                sum += (left_data + rght_data) * d_knl_coeff[bnd_idx + j]; // coef[bnd_idx + j]; // 
            }
            sum = clamp(sum, 0.f, 255.f);
            int idx = hNv ? (x * width + i) : (i * width + x);
            dst_img[idx] = to_uchar4(sum);
        }
    }
    //delete[] line_buf;
}

void filter_gaussian(const int width, const int height, uchar4* src_img, uchar4* dst_img)
{
    const int coef_sz = sizeof(float) * N_KERNEL * KNL_SIZE;
    const int data_sz = sizeof(uchar4) * width * height;

    //-----------------------------------------------
    //  both global mem and constant mem working
    //  but now still using constant for experiments
    //float* d_coef;
    //checkCudaErrors(cudaMalloc(&d_coef, coef_sz));
    //checkCudaErrors(cudaMemcpy(d_coef, knl_coeff, coef_sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(d_knl_coeff, knl_coeff, coef_sz));

    uchar4* d_src;
    uchar4* d_dst;
    checkCudaErrors(cudaMalloc(&d_src, data_sz));
    checkCudaErrors(cudaMalloc(&d_dst, data_sz));
    checkCudaErrors(cudaMemcpy(d_src, src_img, data_sz, cudaMemcpyHostToDevice));

    dim3    blocksize = { 32, 1, 1 };
    dim3    gridsize;

    // you can do recursively filtering for extremely blur
    for (int k = 0; k < 1; k++)
    {
        // mem allocation is aligned with 256x256, leaving certain margin, also 4-bytes RGBA
        const int lim = (((width + 255) >> 8) * ((height + 255) >> 8) * 4) << (8 + 8);
        size_t d_value;
        cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitMallocHeapSize);
        printf("default heap size %d\n", (int)d_value);
        if (d_value < lim)
            cudaDeviceSetLimit(cudaLimit::cudaLimitMallocHeapSize, lim);
        cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitMallocHeapSize);
        printf("update heap size %d\n", (int)d_value);
        cudaDeviceGetLimit(&d_value, cudaLimit::cudaLimitStackSize);
        printf("default static size %d\n", (int)d_value);

        gridsize = { (height + blocksize.x - 1) / blocksize.x, 1 ,1 };
        d_filter_gaussion << <gridsize, blocksize >> > (true, width, height, N_KERNEL, KNL_SIZE, nullptr/*d_coef*/, d_src, d_dst);

        gridsize = { (width + blocksize.x - 1) / blocksize.x, 1 ,1 };
        d_filter_gaussion << <gridsize, blocksize >> > (false, width, height, N_KERNEL, KNL_SIZE, nullptr/*d_coef*/, d_dst, d_src);
    }
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(dst_img, d_src, data_sz, cudaMemcpyDeviceToHost));

    //checkCudaErrors(cudaFree(d_coef));
    checkCudaErrors(cudaFree(d_src));
    checkCudaErrors(cudaFree(d_dst));
}


#include "tools_cuda_dinfo.h"
#include "helper_image.h"
#include "file_util.h"
void test_gaussian(void)
{
    cudaSetDevice(0);
    cudaDeviceProp d_props;
    cudaGetDeviceProperties(&d_props, 0);
    printf("\ndevice name: %s\n", d_props.name);

    const float blur_scale = 15.f;

//#define _SYNTHETIC
#ifdef  _SYNTHETIC
    const int width = 20, height = 20, asize = width * height;
    uchar4* src_img = new uchar4[asize];
    for (int i = 0, m = 0; i < height; i++, m += width) {
        for (int j = 0, n = m; j < width; j++, n++) {
            src_img[n].x = j + 1;
            src_img[n].y = i + 1;
            src_img[n].z = i + j + 1;
            src_img[n].w = 255;
        }
    }
#else
    uint width, height;
    char* src_img;
    //const char* imgPath = "E:/imageNvideo/NTSC/Blonde.ppm";
    //const char* imgPath = "E:/imageNvideo/720p/Forecast.ppm";
    //const char* imgPath = "E:/imageNvideo/1080/city1.ppm";
    const char* imgPath = "E:/imageNvideo/uhd4k/city1.ppm";
    sdkLoadPPM4(imgPath, (unsigned char**)&src_img, &width, &height);
#endif

    uchar4* dst_img = new uchar4[width * height];

    initial_defocus(blur_scale);

    filter_gaussian((int)width, (int)height, (uchar4*)src_img, dst_img);

    if (!file_util::isfile("result"))
        system("mkdir result");
    FILE* fp;
    errno_t err = fopen_s(&fp, "./result/result.ppm", "wb");

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = 0, m = 0; i < (int)height; i++, m += width) {
        for (int j = 0, n = m; j < (int)width; j++, n++) {
            fputc(dst_img[n].x, fp);
            fputc(dst_img[n].y, fp);
            fputc(dst_img[n].z, fp);
        }
    }

    fclose(fp);
    delete[] src_img;
    delete[] dst_img;
}