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
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "all_common.h"
#include "tools_video.h"

#define ModePosX    4
#define ModePosY    50
constexpr int N_KERNEL = 256, KNL_CENTER = N_KERNEL >> 1, KNL_SIZE = 32 + 1;


void old_movies(const int id, const bool is_cuda);
void old_movies_image(const char* file_name, const bool is_cuda);
void old_movies_video(const char* video_nm, const bool is_cuda);

void color_polarizer(cv::Mat& src_img, cv::Mat& dst_img);
void initial_defocus(const float param_Fuzzy);
void dynamic_defocus(cv::Mat& image);

void set_sliders(void);
void set_buttons(cv::Mat&);
void mouse_cb(int event, int x, int y, int flag, void* user_data);

void standalone(void);
