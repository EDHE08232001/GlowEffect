/*******************************************************************************************************************
 * FILE NAME   :    dilate_erode.hpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    dilate or erode algorithm
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 DEC 11      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once
#include "all_common.h"

template<typename _Ty, int _Sh>
class dilate_erode_op
{
public:
    dilate_erode_op(const bool d_n_e) : dNe(d_n_e) { }

    //void hor_op(const int img_hsize, const int img_vsize, const int se_int, const int se_frc, const _Ty* din, _Ty* const dout)
    //{
    //    _Ty* buf = new _Ty[img_hsize];
    //    for (int i = 0, m = 0; i < img_vsize; i++, m += img_hsize)
    //    {
    //        std::memcpy(buf, din + m, sizeof(_Ty) * img_hsize);
    //        for (int j = 0, n = m; j < img_hsize; j++, n++)
    //            dout[n] = (*this)(j, se_int, se_frc, img_hsize, buf);
    //    }

    //    delete[] buf;
    //}

    //void ver_op(const int img_hsize, const int img_vsize, const int se_int, const int se_frc, const _Ty* din, _Ty* const dout)
    //{
    //    _Ty* buf = new _Ty[img_vsize];
    //    for (int i = 0; i < img_hsize; i++)
    //    {
    //        for (int j = 0, n = i; j < img_vsize; j++, n += img_hsize)
    //            buf[j] = din[n];
    //        for (int j = 0, n = i; j < img_vsize; j++, n += img_hsize)
    //            dout[n] = (*this)(j, se_int, se_frc, img_vsize, buf);
    //    }

    //    delete[] buf;
    //}

    
    void hor_op(const int img_hsize, const int img_vsize, const int se_int, const int se_frc, const _Ty* din, _Ty* const dout)
    {
        // 假设分割模型的输出路径是动态传入的或者预定义
        std::string segmentation_image_path = "C:/path_to/onnx-to-trt/seg_out/trt_seg_output_scaled_0.png";

        // 读取分割模型的输出
        cv::Mat segmentation_output = cv::imread(segmentation_image_path, cv::IMREAD_GRAYSCALE);
        if (segmentation_output.empty()) {
            std::cerr << "Error reading segmentation output image file." << std::endl;
            return;
        }

        // 调整大小以匹配 img_hsize 和 img_vsize
        cv::resize(segmentation_output, segmentation_output, cv::Size(img_hsize, img_vsize));

        // 确保 segmentation_output 的大小与 img_hsize 和 img_vsize 匹配
        if (segmentation_output.rows != img_vsize || segmentation_output.cols != img_hsize) {
            std::cerr << "Error: resized segmentation output dimensions do not match expected size." << std::endl;
            return;
        }

        // 打印调试信息，证明分割图像读取成功
        std::cout << "分割图像读取成功，大小为: " << segmentation_output.cols << " x " << segmentation_output.rows << std::endl;


        // 将分割模型的输出数据复制到 dout
        for (int i = 0; i < img_vsize; i++) {
            for (int j = 0; j < img_hsize; j++) {
                dout[i * img_hsize + j] = static_cast<_Ty>(segmentation_output.at<uchar>(i, j));
            }
        }
    }



    void ver_op(const int img_hsize, const int img_vsize, const int se_int, const int se_frc, const _Ty* din, _Ty* const dout)
    {
        // 假设分割模型的输出路径是动态传入的或者预定义
        std::string segmentation_image_path = "C:/path_to/onnx-to-trt/seg_out/trt_seg_output_scaled_0.png";

        // 读取分割模型的输出
        cv::Mat segmentation_output = cv::imread(segmentation_image_path, cv::IMREAD_GRAYSCALE);
        if (segmentation_output.empty()) {
            std::cerr << "Error reading segmentation output image file." << std::endl;
            return;
        }

        // 调整大小以匹配 img_hsize 和 img_vsize
        cv::resize(segmentation_output, segmentation_output, cv::Size(img_hsize, img_vsize));

        // 确保 segmentation_output 的大小与 img_hsize 和 img_vsize 匹配
        if (segmentation_output.rows != img_vsize || segmentation_output.cols != img_hsize) {
            std::cerr << "Error: resized segmentation output dimensions do not match expected size." << std::endl;
            return;
        }

        // 打印调试信息，证明分割图像读取成功
        std::cout << "分割图像读取成功，大小为: " << segmentation_output.cols << " x " << segmentation_output.rows << std::endl;


        // 将分割模型的输出数据复制到 dout
        for (int i = 0; i < img_vsize; i++) {
            for (int j = 0; j < img_hsize; j++) {
                dout[i * img_hsize + j] = static_cast<_Ty>(segmentation_output.at<uchar>(i, j));
            }
        }
    }

private:
    _Ty operator()(const int pix_x, const int se_int_size, const int se_frc_size, const int size, const _Ty* din)
    {
        int max_min = dNe ? init_max : init_min;
        int frst_val, last_val;

        for (int i = 0, m = pix_x - se_int_size; i <= 2 * se_int_size; i++, m++)
        {
            int x = std::max(0, std::min(m, size - 1));

            if (i == 0)
                frst_val = (int)din[x];
            else if (i == 2 * se_int_size)
                last_val = (int)din[x];
            else
                max_min = dNe ? std::max(max_min, (int)din[x]) : std::min(max_min, (int)din[x]);
        }

        int inner_maxmin = max_min;
        if (dNe)
            max_min = std::max(max_min, frst_val),
            max_min = std::max(max_min, last_val);
        else
            max_min = std::min(max_min, frst_val),
            max_min = std::min(max_min, last_val);
        int outer_maxmin = max_min;

        int tmp = (outer_maxmin - inner_maxmin) * se_frc_size + (inner_maxmin << _Sh);
        tmp >>= _Sh;
        tmp = std::max(0, tmp);

        return (_Ty)tmp;
    }

    _Ty avg(const int pix_x, const int se_int_size, const int se_frc_size, const int size, const _Ty* din)
    {
        int average = 0;
        for (int k = 0, m = pix_x - se_int_size; k <= 2 * se_int_size; k++, m++) {
            int x = std::(0, std::min(m, size - 1));
            average += din[x];
        }

        average += se_int_size;
        average /= (2 * se_int_size + 1);
        int tmp = dNe ? std::max(average, din[pix_x]) : std::min(average, din[pix_x]);
        return (_Ty)average;
    }

private:
    bool dNe = true;
    const int  init_max = -100000, init_min = 100000;
};
