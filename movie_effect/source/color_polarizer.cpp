/*******************************************************************************************************************
 * FILE NAME   :    color_polarizer.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    various old movie effects
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 07      Yu Liu          translated DVE3D DSP code
 *
 ********************************************************************************************************************/
#include "old_movies.hpp"

extern int      param_Mode;
extern float    param_Cangle;
extern float    param_Csat;
extern float    param_Uofs;
extern float    param_Vofs;
extern float    param_Yslope;
extern float    param_Yofs;

void color_polarizer(cv::Mat& src_img, cv::Mat& dst_img)
{
    const int rows = src_img.rows;
    const int cols = src_img.cols;
    const int chnl = src_img.channels();
    const int stride = cols * chnl;

    uchar* src_data = src_img.data;
    uchar* dst_data = dst_img.data;

    float V_scale = param_Csat * std::sin(param_Cangle + F_PI4);
    float U_scale = param_Csat * std::cos(param_Cangle + F_PI4);

    for (int iLoop = 0, m = 0; iLoop < rows; iLoop++, m += stride) {
        for (int jLoop = 0, n = m; jLoop < cols; jLoop++, n += chnl) {

            float Y = src_data[n + 0] * 1.f;
            float U = src_data[n + 1] / 128.f - 1.f;
            float V = src_data[n + 2] / 128.f - 1.f;

            Y *= param_Yslope;
            Y += param_Yofs;

            if (param_Mode==1) {
                // c + c*(c-1)*K
                U += U * (U - 1) * param_Uofs;
                V += V * (V - 1) * param_Vofs;
                U *= U_scale;
                V *= V_scale;
            }
            else if (param_Mode==0) {
                // c * scale + offset
                U *= U_scale;
                V *= V_scale;
                U += param_Uofs;
                V += param_Vofs;
            }

            int y = std::min<int>(255, std::max<int>(0, int(Y + .5f)));
            int u = std::min<int>(127, std::max<int>(-128, int(U * 128)));
            int v = std::min<int>(127, std::max<int>(-128, int(V * 128)));

            dst_data[n + 0] = y;
            dst_data[n + 1] = u + 128;
            dst_data[n + 2] = v + 128;
        }
    }

}

void standalone(void)
{
    constexpr int n_frame = 20;
    const char* f_nm = "Z:/test_images/NTSC/Blonde.ppm";
    int hsize = 720, vsize = 486;
    if(!vio::ppm_header(f_nm, hsize, vsize))
        std::exit(0);

    const int asize = hsize * vsize;
    short* src_y = new short[asize], * dst_y = new short[asize];
    short* src_u = new short[asize], * dst_u = new short[asize];
    short* src_v = new short[asize], * dst_v = new short[asize];
    vio::load_rgb2yuv(f_nm, hsize, vsize, src_y, src_u, src_v);

    //float C_delta = -0.5f;//
    float U_offset = 110 / 128.f - 1.f;
    float V_offset = 145 / 128.f - 1.f;

    float angle = -F_PI2;
    float scale = 0.4f;
    for (int kLoop = 0; kLoop < n_frame; kLoop++)
    {
        std::printf("processing frame %d ...\n", kLoop);

        float V_scale = scale * std::sin(angle + F_PI4);
        float U_scale = scale * std::cos(angle + F_PI4);
        angle += F_PI / n_frame;

        //U_offset = -C_delta;
        //V_offset = C_delta;
        //C_delta += 1.f / n_frame;

        for (int iLoop = 0, m = 0; iLoop < vsize; iLoop++, m += hsize) {
            for (int jLoop = 0, n = m; jLoop < hsize; jLoop++, n++) {

                short Y = src_y[n];
                float U = src_u[n] / 128.f;
                float V = src_v[n] / 128.f;

                if (0) {
                    // c + c*(c-1)*K
                    U += U * (U - 1) * U_offset;
                    V += V * (V - 1) * V_offset;
                    U *= U_scale;
                    V *= V_scale;
                }
                else if (1) {
                    // c * scale + offset
                    U *= U_scale;
                    V *= V_scale;
                    U += U_offset;
                    V += V_offset;
                }

                int u = std::min<int>(127, std::max<int>(-128, int(U * 128)));
                int v = std::min<int>(127, std::max<int>(-128, int(V * 128)));

                dst_y[n] = Y;
                dst_u[n] = u;
                dst_v[n] = v;
            }
        }

        char file_nm[200];
        std::sprintf(file_nm, "result/out_%04d.ppm", kLoop);
        vio::save_yuv2rgb(file_nm, hsize, vsize, dst_y, dst_u, dst_v);
    }

    delete[] src_u;
    delete[] src_v;
    delete[] dst_y;
    delete[] dst_u;
    delete[] dst_v;
}