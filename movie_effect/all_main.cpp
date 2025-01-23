/*******************************************************************************************************************
 * FILE NAME   :    all_main.cpp
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

void test_mipmap(void);
void test_mipmap_short(void);
void test_gaussian(void);
void test_texture(void);
void test_movie_kernel();



int main(int argc, char* argv[])
{
    test_mipmap();
    //test_mipmap_short();
    //test_gaussian();
    //test_texture();
    //test_movie_kernel();

#define __RUN_OLD_MOVIE
#ifdef __RUN_OLD_MOVIE
    auto usage = []() {
        printf("\n");
        printf("   usage: movie_effect.exe  image(.jpg/png/tiff)\n");
        printf("   usage: movie_effect.exe  video(.mp4/mov/mpeg)\n");
        printf("   usage: movie_effect.exe  webcam(0, 1, ..., 9)\n\n");
        printf("   click on the word button at left top corner to switch effect mode\n\n");
    };

    bool is_cuda = false;
    int tmp_int;
    checkCudaErrors(cudaGetDeviceCount(&tmp_int));
    if (tmp_int) {
        tmp_int = gpuGetMaxGflopsDeviceId();
        checkCudaErrors(cudaSetDevice(tmp_int));
        is_cuda = true;
    }

    if (argc < 2) {
        usage();
        exit(0);
    }

    std::string file_nm = std::string(argv[1]);
    if (file_nm.find(".jpg") != std::string::npos ||
        file_nm.find(".png") != std::string::npos ||
        file_nm.find(".bmp") != std::string::npos ||
        file_nm.find(".ppm") != std::string::npos ||
        file_nm.find(".png") != std::string::npos ||
        file_nm.find(".tiff") != std::string::npos
        )
        old_movies_image(argv[1], is_cuda);
    else if (
        file_nm.find(".mp4") != std::string::npos ||
        file_nm.find(".mov") != std::string::npos ||
        file_nm.find(".avi") != std::string::npos ||
        file_nm.find(".mpeg") != std::string::npos
        )
        old_movies_video(argv[1], is_cuda);
    else {
        try {
            int num = std::stoi(file_nm);
            if (num >= 0 && num < 10)
                old_movies(num, is_cuda);
        }
        catch (std::exception& e) {
            printf("%s\n", e.what());
            usage();
        }
    }
#endif
    return 0;
}

