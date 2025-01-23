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


__global__ void
d_gen_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0 / float(imageW);
    float py = 1.0 / float(imageH);

    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color =
            (tex2D<float4>(mipInput, (x + 0.f) * px, (y + 0.f) * py)) +
            (tex2D<float4>(mipInput, (x + 1.f) * px, (y + 0.f) * py)) +
            (tex2D<float4>(mipInput, (x + 1.f) * px, (y + 1.f) * py)) +
            (tex2D<float4>(mipInput, (x + 0.f) * px, (y + 1.f) * py));

        color /= 4.0;
        color *= 255.0;
        color = fminf(color, make_float4(255.0));

        surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
    }
}

static void
gen_mipmap(cudaMipmappedArray_t& mipmapArray, cudaExtent size)
{
    size_t width = size.width;
    size_t height = size.height;

    uint level = 0;
    while (width != 1 || height != 1)
    {
        width /= 2;
        width = MAX((size_t)1, width);
        height /= 2;
        height = MAX((size_t)1, height);

        cudaArray_t levelFrom;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        cudaArray_t levelTo;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

        cudaExtent  levelToSize;
        checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
        assert(levelToSize.width == width);
        assert(levelToSize.height == height);
        assert(levelToSize.depth == 0);

        // generate texture object for reading
        cudaTextureObject_t         texInput;
        cudaResourceDesc            texResrc;
        memset(&texResrc, 0, sizeof(cudaResourceDesc));
        texResrc.resType = cudaResourceTypeArray;
        texResrc.res.array.array = levelFrom;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&texInput, &texResrc, &texDescr, NULL));

        // generate surface object for writing
        cudaSurfaceObject_t surfOutput;
        cudaResourceDesc    surfRes;
        memset(&surfRes, 0, sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;

        checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

        // run mipmap kernel
        dim3 blockSize(16, 16, 1);
        dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

        d_gen_mipmap << <gridSize, blockSize >> > (surfOutput, texInput, (uint)width, (uint)height);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaDestroySurfaceObject(surfOutput));
        checkCudaErrors(cudaDestroyTextureObject(texInput));

        level++;
    }
}

__global__ void
d_get_mipmap(cudaTextureObject_t texEngine, const int width, const int height, const float* lod, uchar4* dout)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = yi * width + xi;

    float u = (float)xi / (float)width;
    float v = (float)yi / (float)height;
    bool state;

    if (xi < width && yi < height)
    {
        float4 data = tex2DLod<float4>(texEngine, u, v, lod[idx], &state);
        dout[idx] = to_uchar4(255 * data);
    }
}

__global__ void
d_get_mipmap(cudaTextureObject_t texEngine, const int width, const int height, const float scale, uchar4* dout)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = yi * width + xi;

    float u = (xi + .5f) / (float)width;
    float v = (yi + .5f) / (float)height;

    float lod = abs(u - .5f) * abs(v - .5f) * scale + 1.f;
    lod = log2(lod);

    if (xi < width && yi < height)
    {
        float4 data = tex2DLod<float4>(texEngine, u, v, lod);
        dout[idx] = to_uchar4(255 * data);
    }
}

static void
get_mipmap(cudaMipmappedArray_t mm_array, const int3 img_size, const float scale, uchar4* dout)
{
    const int width = img_size.x; 
    const int height = img_size.y;
    const int n_level = img_size.z;
    const int asize = width * height;

    //-------------------------------------------------
    // generate texture object for reading
    //
    cudaResourceDesc            texResrc;
    memset(&texResrc, 0, sizeof(cudaResourceDesc));
    texResrc.resType = cudaResourceTypeMipmappedArray;// cudaResourceTypeArray;
    texResrc.res.mipmap.mipmap = mm_array;

    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 1;
    texDescr.filterMode = button_State[0] ? cudaFilterModeLinear : cudaFilterModePoint; 
    texDescr.mipmapFilterMode = button_State[1] ? cudaFilterModeLinear : cudaFilterModePoint; 
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.maxMipmapLevelClamp = float(n_level - 1);
    texDescr.readMode = button_State[2]? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    texDescr.disableTrilinearOptimization = button_State[3];

    cudaTextureObject_t         texEngine;
    checkCudaErrors(cudaCreateTextureObject(&texEngine, &texResrc, &texDescr, NULL));

    // generate V-shaped curve for lod (level of detail)
    float* h_lod = new float[asize];
    float* d_lod;
    checkCudaErrors(cudaMalloc(&d_lod, asize * sizeof(float)));
    /*
    for (int i = 0, m = 0; i < height; i++, m += width) {
        float v_lod = (float)std::abs(i - height / 2) / height;
        for (int j = 0, n = m; j < width; j++, n++) {
            float lod = (float)std::abs(j - width / 2) / width;
            lod *= v_lod * scale;
            lod += 1.f;
            lod = std::log2(lod);
            h_lod[n] = lod;
        }
    }
    // copy into device
    cudaMemcpy(d_lod, h_lod, asize * sizeof(float), cudaMemcpyHostToDevice);
    */
    uchar4* d_out;
    checkCudaErrors(cudaMalloc(&d_out, asize * sizeof(uchar4)));

    dim3 blocksize(16, 16, 1);
    dim3 gridsize((width + blocksize.x - 1) / blocksize.x, (height + blocksize.y - 1) / blocksize.y);

    //d_get_mipmap << <gridsize, blocksize >> > (texEngine, width, height, d_lod, d_out);
    d_get_mipmap << <gridsize, blocksize >> > (texEngine, width, height, scale, d_out);

    checkCudaErrors(cudaMemcpy(dout, d_out, asize * sizeof(uchar4), cudaMemcpyKind::cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    checkCudaErrors(cudaDestroyTextureObject(texEngine));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaFree(d_lod));
    delete[] h_lod;
}


void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img)
{
    int n_level = 0, level = max(height, width);
    while (level)
        level >>= 1, n_level++;

    cudaExtent img_size = { (size_t)width, (size_t)height, 0 };
    cudaChannelFormatDesc ch_desc = cudaCreateChannelDesc<uchar4>();
    cudaMipmappedArray_t mm_array;
    checkCudaErrors(cudaMallocMipmappedArray(&mm_array, &ch_desc, img_size, n_level));

    cudaArray_t level0;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mm_array, 0));

    cudaMemcpy3DParms   cpy_param = { 0 };
    cpy_param.srcPtr = make_cudaPitchedPtr((void*)src_img, width * sizeof(uchar4), width, height);
    cpy_param.dstArray = level0;
    cpy_param.extent = img_size;
    cpy_param.extent.depth = 1;
    cpy_param.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&cpy_param));

    gen_mipmap(mm_array, img_size);

    get_mipmap(mm_array, make_int3(width, height, n_level), scale, dst_img);

    checkCudaErrors(cudaFreeMipmappedArray(mm_array));
}


#include "tools_cuda_dinfo.h"
#include "helper_image.h"
#include "file_util.h"
void test_mipmap(void)
{
    cudaSetDevice(0);
    cudaDeviceProp d_props;
    cudaGetDeviceProperties(&d_props, 0);
    printf("\ndevice name: %s\n", d_props.name);

    const float blur_scale = 200.f;

    //#define _SYNTHETIC
#ifdef  _SYNTHETIC
    const int width = 200, height = 200, asize = width * height;
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
    const char* imgPath = "C:/Users/Hawk/Desktop/CSI4900/glow_mipmap/movie_effect/temp_input.ppm";
    //const char* imgPath = "E:/imageNvideo/720p/Forecast.ppm";
    //const char* imgPath = "E:/imageNvideo/1080/letter.ppm";
    //const char* imgPath = "E:/imageNvideo/uhd4k/city1.ppm";
    sdkLoadPPM4(imgPath, (unsigned char**)&src_img, &width, &height);
#endif

    uchar4* dst_img = new uchar4[width * height];
    for (int k = 0; k < 5; k++)
        button_State[k] = true;

    filter_mipmap(width, height, blur_scale, (uchar4*)src_img, dst_img);

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