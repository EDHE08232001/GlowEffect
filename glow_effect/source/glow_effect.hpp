#ifndef GLOW_EFFECT_HPP
#define GLOW_EFFECT_HPP

// Required CUDA type definitions
#include <cuda_runtime.h>


extern int button_id;
extern int param_KeyScale;
extern int param_KeyLevel;
extern int default_scale;
extern cv::Vec3b param_KeyColor;

//Declare filter_mipmap function
//void filter_mipmap(int width, int height, float scale, const uchar4* src_img, uchar4* dst_img);

void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);



void glow_effect_image(const char* image_nm, const cv::Mat& grayscale_mask);

void glow_effect_video(const char* video_nm);

void glow_blow(const cv::Mat& mask, cv::Mat& dst_rgba, int param_KeyLevel, int Delta);
void apply_mipmap(const cv::Mat& src, cv::Mat& dst, float scale, int param_KeyLevel);
void mix_images(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& mask, cv::Mat& dst, float alpha);


#endif // GLOW_EFFECT_HPP
