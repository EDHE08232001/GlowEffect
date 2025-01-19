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

template<typename _Ty>
struct gaussian_blur_op
{
    void operator()(const int img_hsize, const int img_vsize, const int k_size, const float sigma, const _Ty* din, _Ty* const dout)
    {
        float* kernel = new float[k_size + 1];
        const float sigma2 = sigma * sigma;
        for (int k = 0; k <= k_size; k++)
            kernel[k] = std::exp(-k * k / 2. / sigma2) / std::sqrt(2 * M_PI * sigma2);

        _Ty* hline_buf = new _Ty[img_hsize];
        float* vline_buf = new float[img_vsize];
        float* alpha_buf = new float[img_hsize * img_vsize];

        // horizontal blur
        for (int i = 0, m = 0; i < img_vsize; i++, m += img_hsize)
        {
            std::memcpy(hline_buf, din + m, sizeof(_Ty) * img_hsize);

            for (int j = 0, n = i; j < img_hsize; j++, n += img_vsize) {
                float data = kernel[0] * hline_buf[j];
                for (int k = 1; k <= k_size; k++) {
                    _Ty left = hline_buf[j - k < 0 ? 0 : j - k];
                    _Ty rght = hline_buf[j + k < img_hsize ? j + k : img_hsize - 1];
                    data += kernel[k] * (left + rght);
                }
                alpha_buf[n] = data;
            }
        }

        // vertical blur
        float max_data = 0;
        for (int i = 0, m = 0; i < img_hsize; i++, m += img_vsize)
        {
            std::memcpy(vline_buf, alpha_buf + m, sizeof(float) * img_vsize);

            for (int j = 0, n = m; j < img_vsize; j++, n++) {
                float data = kernel[0] * vline_buf[j];
                for (int k = 1; k <= k_size; k++) {
                    float left = vline_buf[j - k < 0 ? 0 : j - k];
                    float rght = vline_buf[j + k < img_vsize ? j + k : img_vsize - 1];
                    data += kernel[k] * (left + rght);
                }
                alpha_buf[n] = data;
                if (data > max_data)
                    max_data = data;
            }
        }

        // scaling
        for (int i = 0, m = 0; i < img_hsize; i++, m += img_vsize)
        {
            std::memcpy(vline_buf, alpha_buf + m, sizeof(float) * img_vsize);

            for (int j = 0, n = i; j < img_vsize; j++, n += img_hsize) {
                _Ty data = (_Ty)(vline_buf[j] * 255 / max_data);
                dout[n] = data;
            }
        }

        delete[] hline_buf;
        delete[] vline_buf;
        delete[] alpha_buf;
        delete[] kernel;
    }
};