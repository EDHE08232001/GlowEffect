/*******************************************************************************************************************
 * FILE NAME   :    dynamic_defocus.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    spatial-variant filter for blurring
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 09      Yu Liu          creation
 *
 ********************************************************************************************************************/
#include "old_movies.hpp"

float knl_coeff[N_KERNEL][KNL_SIZE];

void initial_defocus(const float param_Fuzzy)
{
    const float min_amplitude = 1.e-20f;

    if (param_Fuzzy < 1.f / 255.f) {
        for (int i = 0; i < N_KERNEL; i++) {
            knl_coeff[i][0] = 1.f;
            for (int j = 1; j < KNL_SIZE; j++)
                knl_coeff[i][j] = 0.f;
        }
    }
    else {
        for (int iLoop = 0, m = -KNL_CENTER; iLoop < N_KERNEL; iLoop++, m++) {
            float amplitude = 1.f - std::cos(F_PI * m / KNL_CENTER);
            amplitude *= .5f * param_Fuzzy;
            amplitude *= amplitude;
            if (amplitude < min_amplitude)
                amplitude = min_amplitude;

            float data[KNL_SIZE], sum;
            for (int jLoop = 0; jLoop < KNL_SIZE; jLoop++) {
                float x = (float)jLoop * jLoop;
                x /= amplitude;

                data[jLoop] = std::exp(-x);
                if (jLoop)
                    sum += data[jLoop] * 2;
                else
                    sum = data[jLoop];
            }
            for (int jLoop = 0; jLoop < KNL_SIZE; jLoop++)
                data[jLoop] /= sum;

            std::memcpy(knl_coeff[iLoop], data, sizeof(float) * KNL_SIZE);
        }

    }
}

void dynamic_defocus(cv::Mat& src_img)
{
    const int rows = src_img.rows;
    const int cols = src_img.cols;
    const int chnl = src_img.channels();
    const int hstride = cols * chnl;
    const int vstride = rows;
    uchar* buffer = new uchar[rows * cols];

    // horizontal filter
    for (int iLoop = 0, m = 0; iLoop < rows; iLoop++, m += hstride) {
        for (int jLoop = 0, n = m; jLoop < cols; jLoop++, n += chnl)
        {
            // horizontal dynamic curve
            int k = jLoop * N_KERNEL / cols;
            // initial data
            float sum = src_img.data[n] * knl_coeff[k][0];
            // symetric data: left & right
            for (int kLoop = 1, p = n - chnl, q = n + chnl; kLoop < KNL_SIZE; kLoop++, p -= chnl, q += chnl)
            {
                int data_left = jLoop < kLoop ? src_img.data[m] : src_img.data[p];
                int data_rght = jLoop + kLoop >= cols ? src_img.data[m + hstride - chnl] : src_img.data[q];
                sum += knl_coeff[k][kLoop] * (data_left + data_rght);
            }
            buffer[jLoop * rows + iLoop] = (uchar)std::max<float>(0, std::min<float>(255, sum));
        }
    }

    // vertical filter
    for (int jLoop = 0, m = 0; jLoop < cols; jLoop++, m += vstride) {
        for (int iLoop = 0, n= m; iLoop < rows; iLoop++, n++)
        {
            int k = iLoop * N_KERNEL / rows;
            float sum = buffer[n] * knl_coeff[k][0];
            for (int kLoop = 1, p = n-1, q = n+1; kLoop < KNL_SIZE; kLoop++, p--, q++)
            {
                int data_left = iLoop < kLoop ? buffer[m] : buffer[p];
                int data_rght = iLoop + kLoop >= rows ? buffer[m + vstride - 1] : buffer[q];
                sum += knl_coeff[k][kLoop] * (data_left + data_rght);
            }
            src_img.data[iLoop * hstride + jLoop * chnl] = (uchar)std::max<float>(0, std::min<float>(255, sum));
        }
    }

    delete[] buffer;
}