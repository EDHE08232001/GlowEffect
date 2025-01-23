/*******************************************************************************************************************
 * FILE NAME   :    tools_video.h
 *
 * PROJECTION  :    Prefetching Cache
 *
 * DESCRIPTION :    a file defines a few of windowed-sinc FIR filters
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2017 FEB 15      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#ifndef __TOOLS_VIDEO_H__
#define __TOOLS_VIDEO_H__

#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "tools_report.h"
using namespace std;

#pragma warning(push)
#pragma warning(disable: 4996)

namespace   vio {


    enum RGB2YUV { ITU601, ITU709, WEBWIKI, GXB };
    typedef     RGB2YUV YUV2RGB;

    // 444-to-422 down sampling filter coefficients, symetric 9 taps
    const float coef_to422[5] = { -2.7689423e-017f, -0.0022717039f, -3.1446186e-009f, 0.23516852f, 0.53420633f };


    template<int D = 8>
    void rgb2yuv(const RGB2YUV mode, const short color[3], short& luma, short(&chrm)[2])
    {
        float       Y, Cr, Cb;
        float       R = (float)color[0], G = (float)color[1], B = (float)color[2];
        float       y_unit = (float)(1 << D), y_mask = y_unit - 1, c_unit = y_unit / 2, c_mask = c_unit - 1;

        switch (mode) {
        case ITU601:
            R /= y_mask, G /= y_mask, B /= y_mask;
            Y = (0.299f * R) + (0.587f * G) + (0.114f * B);
            Cb = -(0.169f * R) - (0.331f * G) + (0.500f * B);
            Cr = (0.500f * R) - (0.419f * G) - (0.081f * B);
            Y *= y_mask, Cb *= c_unit, Cr *= c_unit;
            break;
        case ITU709:
            R /= y_mask, G /= y_mask, B /= y_mask;
            Y = +(0.2126f * R) + (0.7152f * G) + (0.0722f * B);
            //Cb = -(0.2126f * R) - (0.7152f * G) + (0.9278f * B); Cb /= 1.8556f;
            Cb = -0.114572f * R - 0.385428f * G + 0.5f * B;
            //Cr = +(0.7874f * R) - (0.7152f * G) - (0.0722f * B); Cr /= 1.5748f;
            Cr = +0.5f * R - 0.454153f * G - 0.045847f * B;
            Y *= y_mask, Cb *= c_unit, Cr *= c_unit;
            break;
        case WEBWIKI:
            Y = (2104 * color[0] + 4130 * color[1] + 802 * color[2] + 4096 + 131072) / 13235.0f;
            Cb = (-1214 * color[0] - 2384 * color[1] + 3598 * color[2] + 4096 + 1348576) / 13240.0f;
            Cr = (3598 * color[0] - 3013 * color[1] - 585 * color[2] + 4096 + 1348576) / 13240.0f;
            break;
        case GXB:
            Y = float((16829 * color[0] + 33039 * color[1] + 6416 * color[2]) >> 16);// + 64;
            Cr = float((28784 * color[0] - 24103 * color[1] - 4680 * color[2]) >> 16);// + 512;
            Cb = float((-9714 * color[0] - 19070 * color[1] + 28784 * color[2]) >> 16);// + 512;
            break;
        default:
            Y = (0.257f * R) + (0.504f * G) + (0.098f * B);// + 16;
            Cr = (0.439f * R) - (0.368f * G) - (0.071f * B);// + 128;
            Cb = -(0.148f * R) - (0.291f * G) + (0.439f * B);// + 128;
        }
        luma = (Y < 0) ? 0 : (Y > y_mask) ? (short)y_mask : (short)Y;
        chrm[0] = (Cb < -c_unit) ? -(short)c_unit : (Cb > c_mask) ? (short)c_mask : (short)Cb;
        chrm[1] = (Cr < -c_unit) ? -(short)c_unit : (Cr > c_mask) ? (short)c_mask : (short)Cr;
    }

    template<short D = 8>
    void yuv2rgb(const YUV2RGB mode, const short luma, const short chrm[2], short(&color)[3])
    {
        float       R, G, B;
        float       Y = (float)luma, Cb = (float)chrm[0], Cr = (float)chrm[1];
        float       y_unit = (float)(1 << D), y_mask = y_unit - 1, c_unit = y_unit / 2;

        switch (mode) {
        case ITU601:
            Y /= y_mask, Cb /= c_unit, Cr /= c_unit;
            R = Y + 0.000f * Cb + 1.403f * Cr;
            G = Y - 0.344f * Cb - 0.714f * Cr;
            B = Y + 1.773f * Cb + 0.000f * Cr;
            R *= y_mask, G *= y_mask, B *= y_mask;
            break;
        case ITU709:
            Y /= y_mask, Cb /= c_unit, Cr /= c_unit;
            R = Y + 0.000000f * Cb + 1.574800f * Cr;
            G = Y - 0.187324f * Cb - 0.468124f * Cr;
            B = Y + 1.855600f * Cb + 0.000000f * Cr;
            R *= y_mask, G *= y_mask, B *= y_mask;
            break;
        case WEBWIKI:
            R = (9535 * luma + 13074 * chrm[1]) / 13255.0f;
            G = (9535 * luma - 06660 * chrm[1] - 3203 * chrm[0]) / 13255.0f;
            B = (9535 * luma + 16531 * chrm[0]) / 13255.0f;
            break;
        case GXB:
            R = float((76309 * luma + 104597 * chrm[1]) >> 16);
            G = float((76309 * luma - 25674 * chrm[0] - 53279 * chrm[1]) >> 16);
            B = float((76309 * luma + 132201 * chrm[0]) >> 16);
            break;
        default:
            R = 1.164f * Y + 0.000f * Cb + 1.596f * Cr;
            G = 1.164f * Y - 0.391f * Cb - 0.813f * Cr;
            B = 1.164f * Y + 2.018f * Cb + 0.000f * Cr;
        }

        color[0] = (short)(R<0 ? 0 : R>y_mask ? y_mask : R);
        color[1] = (short)(G<0 ? 0 : G>y_mask ? y_mask : G);
        color[2] = (short)(B<0 ? 0 : B>y_mask ? y_mask : B);
    }


    inline FILE* ppm_header(const char* file_name, const char* mode, const int hsize, const int fsize)
    {
        char    buffer[200];
        int     width, height;

        FILE* fp = fopen(file_name, mode);
        if (fp == NULL)
            rpt::ERROR_report("files doesn't exist");

        if (std::strstr(mode, "rb")) {

            fgets(buffer, 200, fp);
            if (!std::strstr(buffer, "P6"))
                rpt::ERROR_report("not ppm file");

            do {
                fgets(buffer, 200, fp);
            } while (buffer[0] == '#');

            sscanf(buffer, "%d %d", &width, &height);
            if (width != hsize || height != fsize)
                rpt::ERROR_report("image size doesn't match");

            fgets(buffer, 200, fp);
        }
        else if (std::strstr(mode, "wb")) {
            fprintf(fp, "P6\n");
            fprintf(fp, "%d %d\n", hsize, fsize);
            fprintf(fp, "255\n");
        }
        else
            rpt::ERROR_report("file mode doesn't exist");

        return fp;
    }


    inline bool ppm_header(const char* file_name, int& hsize, int& fsize)
    {
        FILE* fp = fopen(file_name, "rb");
        if (fp == nullptr) {
            std::printf("file %s doesn't exist\n", file_name);
            return false;
        }
        char    buffer[200];
        fgets(buffer, 200, fp);
        if (!std::strstr(buffer, "P6"))
            rpt::ERROR_report("not ppm file");

        do {
            fgets(buffer, 200, fp);
        } while (buffer[0] == '#');

        sscanf(buffer, "%d %d", &hsize, &fsize);
        std::fclose(fp);
        return true;
    }

    inline FILE* ppm_header(const char* file_name, const bool rnw, int& hsize, int& fsize)
    {
        char    buffer[200];

        FILE* fp = fopen(file_name, rnw ? "rb" : "wb");
        if (fp == NULL)
            rpt::ERROR_report("files doesn't exist");

        if (rnw)
        {
            fgets(buffer, 200, fp);
            if (!std::strstr(buffer, "P6"))
                rpt::ERROR_report("not ppm file");

            do {
                fgets(buffer, 200, fp);
            } while (buffer[0] == '#');

            sscanf(buffer, "%d %d", &hsize, &fsize);
            fgets(buffer, 200, fp);
        }
        else {
            fprintf(fp, "P6\n");
            fprintf(fp, "%d %d\n", hsize, fsize);
            fprintf(fp, "255\n");
        }

        return fp;
    }

    inline bool pgm_header(const char* file_name, int& hsize, int& fsize)
    {
        FILE* fp = fopen(file_name, "rb");
        if (fp == nullptr) {
            fclose(fp);
            return false;
        }
        char    buffer[200];
        fgets(buffer, 200, fp);
        if (!std::strstr(buffer, "P5"))
            rpt::ERROR_report("not pgm file");

        do {
            fgets(buffer, 200, fp);
        } while (buffer[0] == '#');

        sscanf(buffer, "%d %d", &hsize, &fsize);
        std::fclose(fp);
        return true;
    }

    inline FILE* pgm_header(const char* file_name, const char* mode, const int hsize, const int fsize)
    {
        char    buffer[200];
        int     width, height;

        FILE* fp = fopen(file_name, mode);
        if (fp == NULL)
            rpt::ERROR_report("files doesn't exist");

        if (std::strstr(mode, "rb")) {

            fgets(buffer, 200, fp);
            if (!std::strstr(buffer, "P5"))
                rpt::ERROR_report("not ppm file");

            do {
                fgets(buffer, 200, fp);
            } while (buffer[0] == '#');

            sscanf(buffer, "%d %d", &width, &height);
            if (width != hsize || height != fsize)
                rpt::ERROR_report("image size doesn't match");

            fgets(buffer, 200, fp);
        }
        else if (std::strstr(mode, "wb")) {
            fprintf(fp, "P5\n");
            fprintf(fp, "%d %d\n", hsize, fsize);
            fprintf(fp, "255\n");
        }
        else
            rpt::ERROR_report("file mode doesn't exist");

        return fp;
    }

    inline FILE* tga_header(const char *file_name, const char *mode, const int hsize, const int fsize)
    {
        FILE *fp = fopen(file_name, mode);
        if (!fp)
            rpt::ERROR_report("file doesn't exist");

        std::putc(0, fp);
        std::putc(0, fp);
        std::putc(2, fp);                         // uncompressed RGB
        std::putc(0, fp); std::putc(0, fp);
        std::putc(0, fp); std::putc(0, fp);
        std::putc(0, fp);
        std::putc(0, fp); std::putc(0, fp);        // X origin
        std::putc(0, fp); std::putc(0, fp);        // y origin
        std::putc((hsize & 0x00FF) >> 0, fp);
        std::putc((hsize & 0xFF00) >> 8, fp);
        std::putc((fsize & 0x00FF) >> 0, fp);
        std::putc((fsize & 0xFF00) >> 8, fp);
        std::putc(32, fp);                        // 32 bit bitmap
        std::putc(0x28, fp);                      // 

        return fp;
    }

    
    inline void load_rgb2rgb(FILE* fp, const int hsize, const int fsize,
                             int8_t* const img_r, int8_t* const img_g, int8_t* const  img_b)
    {
        int     i, j, m, n;
        for (i = m = 0; i < fsize; i++, m += hsize) {
            for (j = 0, n = m; j < hsize; j++, n++) {
                img_r[n] = (int8_t)std::fgetc(fp);
                img_g[n] = (int8_t)std::fgetc(fp);
                img_b[n] = (int8_t)std::fgetc(fp);
            }
        }
    }

    template<int H, int W, int D>
    void load_rgb2rgb(char* path, char* file, const int hsize, const int fsize,
                      short(&img_r)[H][W], short(&img_g)[H][W], short(&img_b)[H][W])
    {
        char    filename[200];  sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int shift = D - 8;
        int     i, j;
        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++) {
                img_r[i][j] = (short)(std::fgetc(fp) << shift);
                img_g[i][j] = (short)(std::fgetc(fp) << shift);
                img_b[i][j] = (short)(std::fgetc(fp) << shift);
            }
        }
        std::fclose(fp);
    }

    template<int D = 8>
    void load_rgb2rgb(FILE* fp, const int hsize, const int fsize, const int vincr,
                      short* const img_r, short* const img_g, short* const  img_b)
    {
        const int   shift = D - 8;

        int     i, j, m, n;
        if (vincr < 2) {
            for (i = m = 0; i < fsize; i += 2, m += hsize)
            {
                if (vincr == 1)
                    for (j = 0; j < hsize; j++)
                        std::fgetc(fp), std::fgetc(fp), std::fgetc(fp);

                for (j = 0, n = m; j < hsize; j++, n++) {
                    img_r[n] = (short)(std::fgetc(fp) << shift);
                    img_g[n] = (short)(std::fgetc(fp) << shift);
                    img_b[n] = (short)(std::fgetc(fp) << shift);
                }

                if (vincr == 0)
                    for (j = 0; j < hsize; j++)
                        std::fgetc(fp), std::fgetc(fp), std::fgetc(fp);
            }
        }
        else {
            for (i = m = 0; i < fsize; i++, m += hsize) {
                for (j = 0, n = m; j < hsize; j++, n++) {
                    img_r[n] = (short)(std::fgetc(fp) << shift);
                    img_g[n] = (short)(std::fgetc(fp) << shift);
                    img_b[n] = (short)(std::fgetc(fp) << shift);
                }
            }
        }
    }

    template<int D = 8>
    void load_rgb2rgb(const char* filename, const int hsize, const int fsize, const int vincr,
                      short* const img_r, short* const img_g, short* const  img_b)
    {
        FILE    *fp = ppm_header(filename, "rb", hsize, fsize);
        load_rgb2rgb<D>(fp, hsize, fsize, vincr, img_r, img_g, img_b);
        std::fclose(fp);
    }

    template<int D = 8>
    void load_rgb2rgb(char* path, char* file, const int hsize, const int fsize, const int vincr,
                      short* const img_r, short* const img_g, short* const  img_b)
    {
        char    filename[200];  sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);
        load_rgb2rgb<D>(fp, hsize, fsize, vincr, img_r, img_g, img_b);
        std::fclose(fp);
    }


    template<int D = 8>
    void load_rgb2yuv(const char* filename, const int hsize, const int fsize,
                      short* img_y0, short* img_u0, short* img_v0,
                      short* img_y1, short* img_u1, short* img_v1)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        short   color[3], luma, chrm[2];
        for (int i = 0, j, k, m = 0, n; i < fsize; i++) {
            for (j = 0, n = m; j < hsize; j++, n++)
            {
                for (k = 0; k < 3; k++)
                    color[k] = (short)(std::fgetc(fp) << (D - 8));
                vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                if (i & 1)  img_y1[n] = luma, img_u1[n] = chrm[0], img_v1[n] = chrm[1];
                else        img_y0[n] = luma, img_u0[n] = chrm[0], img_v0[n] = chrm[1];
            }
            if (i & 1)
                m += hsize;
        }
        std::fclose(fp);
    }

    template<int D = 8>
    void load_rgb2yuv(const char* filename, const int hsize, const int fsize, const int field,
                      short* img_y, short* img_u, short* img_v)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);
        short   color[3], luma, chrm[2];

        for (int i = 0, j, k, m = 0, n; i < fsize; i += 2, m += hsize)
        {
            if (field) {
                for (j = 0; j < hsize; j++) {
                    for (k = 0; k < 3; k++) std::fgetc(fp);
                }
            }

            for (j = 0, n = m; j < hsize; j++, n++)
            {
                for (k = 0; k < 3; k++)
                    color[k] = (short)std::fgetc(fp) << (D - 8);
                vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                img_y[n] = luma, img_u[n] = chrm[0], img_v[n] = chrm[1];
            }

            if (!field) {
                for (j = 0; j < hsize; j++) {
                    for (k = 0; k < 3; k++) std::fgetc(fp);
                }
            }
        }

        std::fclose(fp);
    }


    template<int D = 8>
    void load_rgb2yuv(const char* filename, const int hsize, const int fsize,
                      short* img_y, short* img_u, short* img_v)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);
        short   color[3], luma, chrm[2];

        for (int i = 0, j, k, m = 0, n; i < fsize; i++, m += hsize)
        {
            for (j = 0, n = m; j < hsize; j++, n++)
            {
                for (k = 0; k < 3; k++)
                    color[k] = (short)std::fgetc(fp) << (D - 8);
                vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                img_y[n] = luma, img_u[n] = chrm[0], img_v[n] = chrm[1];
            }
        }

        std::fclose(fp);
    }

    template<int D = 8, bool OFS=false>
    void load_rgb2yc(const char* filename, const int hsize, const int fsize, short* img_y, short* img_c)
    {
        constexpr int C_OFS = 1 << (D - 1);
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);
        short   color[3], luma, chrm[2];

        for (int i = 0, j, k, m = 0, n; i < fsize; i++, m += hsize)
        {
            for (j = 0, n = m; j < hsize; j++, n++)
            {
                for (k = 0; k < 3; k++)
                    color[k] = (short)std::fgetc(fp) << (D - 8);
                vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                if (OFS)
                    chrm[0] += C_OFS, chrm[1] += C_OFS;
                img_y[n] = luma, img_c[n] = chrm[j & 1];
            }
        }

        std::fclose(fp);
    }

    template<int D = 8, bool OFS = false>
    void load_rgb2yc(const char* path, const char* name, const int hsize, const int fsize, short* img_y, short* img_c)
    {
        char filename[1000];
        sprintf(filename, "%s/%s", path, name);
        load_rgb2yc<D, OFS>(filename, hsize, fsize, img_y, img_c);
    }


    template<int H, int W, int D>
    void load_rgb2yc(char* path, char* file, const int hsize, const int fsize,
                     short(&img_y)[H][W], short(&img_c)[H][W])
    {
        char    filename[200];  sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   unit = 1 << (D - 1), mask = unit - 1;
        int         i, j, k, m;
        short       color[3], luma, chrm[2];
        short       buf[3][9], c_dly;
        float       data;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < (hsize + 4); j++)
            {
                // loading
                if (j < hsize) {
                    for (k = 0; k < 3; k++)
                        color[k] = std::fgetc(fp) << (D - 8);
                    vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                }

                // filtering
                if (!j) {
                    buf[0][0] = buf[0][1] = buf[0][2] = buf[0][3] = luma;
                    buf[1][0] = buf[1][1] = buf[1][2] = buf[1][3] = chrm[0];
                    buf[2][0] = buf[2][1] = buf[2][2] = buf[2][3] = chrm[1];
                }

                for (k = 0; k < 3; k++) {
                    for (m = 8; m > 0; m--)
                        buf[k][m] = buf[k][m - 1];
                }
                buf[0][0] = luma;
                buf[1][0] = chrm[0];
                buf[2][0] = chrm[1];

                for (k = 1; k < 3; k++) {
                    data = buf[k][4] * coef_to422[4];
                    for (m = 0; m < 4; m++)
                        data += (buf[k][m] + buf[k][8 - m]) * coef_to422[m];
                    data += data < 0 ? -0.5f : 0.5f;
                    color[k] = (short)data;
                    if (color[k] > mask)   color[k] = mask;
                    if (color[k] < -unit)   color[k] = -unit;
                }
                c_dly = color[2];   // cr delay 

                m = j - 4;
                if (m >= 0) {
                    img_y[i][m] = buf[0][4];
                    img_c[i][m] = (m & 1) ? c_dly : color[1];
                }

            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void load_rgb2yc(const char* path, char* const file, const int hsize, const int fsize,
                     short(&img_y0)[H / 2][W], short(&img_c0)[H / 2][W],
                     short(&img_y1)[H / 2][W], short(&img_c1)[H / 2][W])
    {
        char    filename[200];  sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   unit = 1 << (D - 1), mask = unit - 1;
        int         i, j, k, m, n;
        short       color[3], luma, chrm[2];
        short       buf[3][9], c_dly;
        float       data;

        for (i = 0; i < fsize; i++) {
            n = i >> 1;
            for (j = 0; j < (hsize + 4); j++)
            {
                // loading
                if (j < hsize) {
                    for (k = 0; k < 3; k++)
                        color[k] = std::fgetc(fp) << (D - 8);
                    vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                }

                // filtering
                if (!j) {
                    buf[0][0] = buf[0][1] = buf[0][2] = buf[0][3] = luma;
                    buf[1][0] = buf[1][1] = buf[1][2] = buf[1][3] = chrm[0];
                    buf[2][0] = buf[2][1] = buf[2][2] = buf[2][3] = chrm[1];
                }

                for (k = 0; k < 3; k++) {
                    for (m = 8; m > 0; m--)
                        buf[k][m] = buf[k][m - 1];
                }
                buf[0][0] = luma;
                buf[1][0] = chrm[0];
                buf[2][0] = chrm[1];

                for (k = 1; k < 3; k++) {
                    data = buf[k][4] * coef_to422[4];
                    for (m = 0; m < 4; m++)
                        data += (buf[k][m] + buf[k][8 - m]) * coef_to422[m];
                    data += data < 0 ? -0.5f : 0.5f;
                    color[k] = (short)data;
                    if (color[k] > mask)   color[k] = mask;
                    if (color[k] < -unit)   color[k] = -unit;
                }
                c_dly = color[2];   // cr delay 

                m = j - 4;
                if (m >= 0) {
                    if (i & 1) {
                        img_y1[n][m] = buf[0][4];
                        img_c1[n][m] = (m & 1) ? c_dly : color[1];
                    }
                    else {
                        img_y0[n][m] = buf[0][4];
                        img_c0[n][m] = (m & 1) ? c_dly : color[1];
                    }
                }
            }
        }

        std::fclose(fp);
    }

    template<int H, int W, int D>
    void load_rgb_422(char* path, char* file, const int hsize, const int fsize,
                      short(&img_y)[H][W], short(&img_u)[H][W / 2], short(&img_v)[H][W / 2])
    {
        char    filename[200];  sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   unit = 1 << (D - 1), mask = unit - 1;
        int         i, j, k, m;
        short       color[3], luma, chrm[2];
        short       buf[3][9], c_dly;
        float       data;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < (hsize + 4); j++)
            {
                // loading
                if (j < hsize) {
                    for (k = 0; k < 3; k++)
                        color[k] = std::fgetc(fp) << (D - 8);
                    vio::rgb2yuv<D>(vio::ITU709, color, luma, chrm);
                }

                // filtering
                if (!j) {
                    buf[0][0] = buf[0][1] = buf[0][2] = buf[0][3] = luma;
                    buf[1][0] = buf[1][1] = buf[1][2] = buf[1][3] = chrm[0];
                    buf[2][0] = buf[2][1] = buf[2][2] = buf[2][3] = chrm[1];
                }

                for (k = 0; k < 3; k++) {
                    for (m = 8; m > 0; m--)
                        buf[k][m] = buf[k][m - 1];
                }
                buf[0][0] = luma;
                buf[1][0] = chrm[0];
                buf[2][0] = chrm[1];

                for (k = 1; k < 3; k++) {
                    data = buf[k][4] * coef_to422[4];
                    for (m = 0; m < 4; m++)
                        data += (buf[k][m] + buf[k][8 - m]) * coef_to422[m];
                    data += data < 0 ? -0.5f : 0.5f;
                    color[k] = (short)data;
                    if (color[k] > mask)   color[k] = mask;
                    if (color[k] < -unit)   color[k] = -unit;
                }
                c_dly = color[2];   // cr delay 

                m = j - 4;
                if (m >= 0) {
                    img_y[i][m] = buf[0][4];
                    if ((m & 1) == 0) {
                        m >>= 1;
                        img_u[i][m] = color[1];
                        img_v[i][m] = c_dly;
                    }
                }

            }
        }
        std::fclose(fp);
    }

    template<int H, int W, int D>
    void save_rgb2rgb(const char* path, const char* file, const int hsize, const int fsize,
                      short(&img_r)[H][W], short(&img_g)[H][W], short(&img_b)[H][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8;
        int         i, j;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                std::fputc(img_r[i][j] >> shift, fp);
                std::fputc(img_g[i][j] >> shift, fp);
                std::fputc(img_b[i][j] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_rgba2rgba(const char* path, const char* file, const int hsize, const int fsize,
                        const short(&img_r)[H][W], const short(&img_g)[H][W], const short(&img_b)[H][W], const short(&img_a)[H][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.tga", path, file);
        FILE* fp = tga_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8, unit = 1 << D, mask = unit - 1;
        int         i, j, alpha;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                std::fputc(img_b[i][j] >> shift, fp);
                std::fputc(img_g[i][j] >> shift, fp);
                std::fputc(img_r[i][j] >> shift, fp);

                alpha = img_a[i][j];
                if (alpha > mask)
                    alpha = mask;
                std::fputc(alpha >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int D = 8>
    void save_rgb2rgb(const char *path, const char *file, const int hsize, const int fsize,
                      short* const img_r0, short* const img_g0, short* const img_b0,
                      short* const img_r1, short* const img_g1, short* const img_b1)
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        save_rgb2rgb<D>(filename, hsize, fsize, img_r0, img_g0, img_b0, img_r1, img_g1, img_b1);
    }
    
    template<int D = 8>
    void save_rgb2rgb(const char* filename, const int hsize, const int fsize,
                      short* const img_r0, short* const img_g0, short* const img_b0,
                      short* const img_r1, short* const img_g1, short* const img_b1)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        const int   shift = D - 8;
        int         i, j, m, n;

        for (i = m = 0; i < fsize; i += 2, m += hsize) {
            for (j = 0, n = m; j < hsize; j++, n++) {
                std::fputc(img_r0[n] >> shift, fp);
                std::fputc(img_g0[n] >> shift, fp);
                std::fputc(img_b0[n] >> shift, fp);
            }
            for (j = 0, n = m; j < hsize; j++, n++) {
                std::fputc(img_r1[n] >> shift, fp);
                std::fputc(img_g1[n] >> shift, fp);
                std::fputc(img_b1[n] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_yc2rgb(const char* path, const char* file, const int hsize, const int fsize,
                     const short(&img_y)[H][W], const short(&img_c)[H][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8;
        short       luma, chrm[2][2], color[3];
        int         i, j, k, pair;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                luma = img_y[i][j];
                pair = j & 1;

                if (pair == 0)
                {
                    chrm[0][0] = chrm[1][0] = img_c[i][j + 0];
                    chrm[0][1] = chrm[1][1] = img_c[i][j + 1];

                    k = j + 2;
                    if (k >= W)
                        k = j;

                    chrm[1][0] += img_c[i][k + 0];
                    chrm[1][1] += img_c[i][k + 1];

                    chrm[1][0] >>= 1;
                    chrm[1][1] >>= 1;
                }

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm[pair], color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_yc2rgb(const char* path, char* const file, const int hsize, const int fsize,
                     const short(&img_y0)[H / 2][W], const short(&img_c0)[H / 2][W], const short(&img_y1)[H / 2][W], const short(&img_c1)[H / 2][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8;
        short       luma, chrm[2][2], color[3];
        int         i, j, k, n, pair;

        for (i = 0; i < fsize; i++) {
            n = i >> 1;
            for (j = 0; j < hsize; j++)
            {
                luma = i & 1 ? img_y1[n][j] : img_y0[n][j];
                pair = j & 1;

                if (pair == 0) {
                    chrm[0][0] = chrm[1][0] = i & 1 ? img_c1[n][j + 0] : img_c0[n][j + 0];
                    chrm[0][1] = chrm[1][1] = i & 1 ? img_c1[n][j + 1] : img_c0[n][j + 1];

                    k = j + 2;
                    if (k >= W)
                        k = j;

                    chrm[1][0] += i & 1 ? img_c1[n][k + 0] : img_c0[n][k + 0];
                    chrm[1][1] += i & 1 ? img_c1[n][k + 1] : img_c0[n][k + 1];

                    chrm[1][0] >>= 1;
                    chrm[1][1] >>= 1;
                }

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm[pair], color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_yuv2rgb(const char* path, const char* file, const int hsize, const int fsize,
                      const short(&img_y)[H][W], const short(&img_u)[H][W], const short(&img_v)[H][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8;
        short       luma, chrm[2], color[3];
        int         i, j;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                luma = img_y[i][j];
                chrm[0] = img_u[i][j];
                chrm[1] = img_v[i][j];

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm, color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D = 8>
    void save_yuv2rgb(
        const char* path, char* const file, const int hsize, const int fsize,
        const short(&img_y0)[H / 2][W], const short(&img_u0)[H / 2][W], const short(&img_v0)[H / 2][W],
        const short(&img_y1)[H / 2][W], const short(&img_u1)[H / 2][W], const short(&img_v1)[H / 2][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        save_yuv2rgb<H, W, D>(filename, hsize, fsize, img_y0, img_u0, img_v0, img_y1, img_u1, img_v1);
    }
    
    template<int H, int W, int D>
    void save_yuv2rgb(
        char* const filename, const int hsize, const int fsize,
        const short(&img_y0)[H / 2][W], const short(&img_u0)[H / 2][W], const short(&img_v0)[H / 2][W],
        const short(&img_y1)[H / 2][W], const short(&img_u1)[H / 2][W], const short(&img_v1)[H / 2][W])
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8;
        short       luma, chrm[2], color[3];
        int         i, j, n;

        for (i = 0; i < fsize; i++) {
            n = i >> 1;
            for (j = 0; j < hsize; j++)
            {
                luma = i & 1 ? img_y1[n][j] : img_y0[n][j];
                chrm[0] = i & 1 ? img_u1[n][j] : img_u0[n][j];
                chrm[1] = i & 1 ? img_v1[n][j] : img_v0[n][j];

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm, color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int D = 8>
    void save_alpha(const char* filename, const int hsize, const int fsize, short* img_a)
    {
        FILE* fp = pgm_header(filename, "wb", hsize, fsize);

        const int   shift = D - 8;
        int         i, j, m, n;

        for (i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (j = 0, n = m; j < hsize; j++, n++)
                std::fputc(img_a[n] >> shift, fp);
        }
        std::fclose(fp);
    }

    template<int D = 8>
    void save_alpha(const char* filename, const int hsize, const int vsize, short* img0_a, short* img1_a)
    {
        FILE* fp = pgm_header(filename, "wb", hsize, vsize << 1);

        const int   shift = D - 8;
        int         i, j, k, m, n;

        for (i = 0, m = 0; i < vsize; i++, m += hsize) {
            for (j = 0, n = m; j < hsize; j++, n++)
                std::fputc(img0_a[n] >> shift, fp);
            for (j = 0, n = m; j < hsize; j++, n++)
                std::fputc(img1_a[n] >> shift, fp);
        }
        std::fclose(fp);
    }


    template<int D = 8>
    void save_yc2rgb(const char* filename, const int hsize, const int fsize, short* img_y, short* img_c)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        const int   shift = D - 8;
        short       luma, chrm[2], color[3];
        int         i, j, m, n;

        for (i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (j = 0, n = m; j < hsize; j++, n++)
            {
                luma = img_y[n];
                if ((j & 1) == 0)
                    chrm[0] = img_c[n + 0], chrm[1] = img_c[n + 1];
                if (j < hsize - 2) {
                    if (j & 1) {
                        chrm[0] += img_c[n - 1 + 2]; chrm[0] >>= 1;
                        chrm[1] += img_c[n + 0 + 2]; chrm[1] >>= 1;
                    }
                }
                vio::yuv2rgb<D>(vio::ITU709, luma, chrm, color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }

    template<int D = 8>
    void save_yc2rgb(const char* filename, const int hsize, const int vsize, short* img0_y, short* img0_c, short* img1_y, short* img1_c)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, vsize << 1);

        const int   shift = D - 8;
        short       luma, chrm[2], color[3];
        int         i, j, k, m, n;

        for (i = 0, m = 0; i < vsize; i++, m += hsize) {
            for (k = 0; k < 2; k++) {
                for (j = 0, n = m; j < hsize; j++, n++)
                {
                    luma = k ? img1_y[n] : img0_y[n];
                    if ((j & 1) == 0)
                        chrm[0] = k ? img1_c[n + 0] : img0_c[n + 0],
                        chrm[1] = k ? img1_c[n + 1] : img0_c[n + 1];
                    if (j < hsize - 2) {
                        if (j & 1) {
                            chrm[0] += k ? img1_c[n - 1 + 2] : img0_c[n - 1 + 2]; chrm[0] >>= 1;
                            chrm[1] += k ? img1_c[n + 0 + 2] : img0_c[n + 0 + 2]; chrm[1] >>= 1;
                        }
                    }
                    vio::yuv2rgb<D>(vio::ITU709, luma, chrm, color);

                    std::fputc(color[0] >> shift, fp);
                    std::fputc(color[1] >> shift, fp);
                    std::fputc(color[2] >> shift, fp);
                }
            }
        }
        std::fclose(fp);
    }

    template<int D=8>
    void save_yuv2rgb(const char* filename, const int hsize, const int fsize, short* img_y, short* img_u, short* img_v)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        const int   shift = D - 8;
        short       luma, chrm[2], color[3];
        int         i, j, m, n;

        for (i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (j = 0, n = m; j < hsize; j++, n++)
            {
                luma = img_y[n];
                chrm[0] = img_u[n];
                chrm[1] = img_v[n];

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm, color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int D=8>
    void save_yuv2rgb(const char* filename, const int hsize, const int fsize,
                      short* img_y0, short* img_u0, short* img_v0, short* img_y1, short* img_u1, short* img_v1)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        const int   shift = D - 8;
        short       luma, chrm[2], color[3];
        int         i, j, k, m, n;

        for (i = 0, m = 0; i < fsize; i += 2, m += hsize) {
            for (k = 0; k < 2; k++) {
                for (j = 0, n = m; j < hsize; j++, n++)
                {
                    luma = k ? img_y1[n] : img_y0[n];
                    chrm[0] = k ? img_u1[n] : img_u0[n];
                    chrm[1] = k ? img_v1[n] : img_v0[n];

                    vio::yuv2rgb<D>(vio::ITU709, luma, chrm, color);

                    std::fputc(color[0] >> shift, fp);
                    std::fputc(color[1] >> shift, fp);
                    std::fputc(color[2] >> shift, fp);
                }
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_422_rgb(const char* path, const char* file, const int hsize, const int fsize,
                      const short(&img_y)[H][W], const short(&img_u)[H][W / 2], const short(&img_v)[H][W / 2])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8, hhalf = W / 2;
        short       luma, chrm[2][2], color[3];
        int         i, j, k, pair;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                luma = img_y[i][j];
                pair = j & 1;

                if (pair == 0)
                {
                    k = j >> 1;
                    chrm[0][0] = chrm[1][0] = img_u[i][k];
                    chrm[0][1] = chrm[1][1] = img_v[i][k];

                    k++;
                    chrm[1][0] += k < hhalf ? img_u[i][k] : img_u[i][hhalf - 1];
                    chrm[1][1] += k < hhalf ? img_v[i][k] : img_v[i][hhalf - 1];

                    chrm[1][0] >>= 1;
                    chrm[1][1] >>= 1;
                }

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm[pair], color);

                std::fputc(color[0] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[2] >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_4224_rgba(const char* path, const char* file, const int hsize, const int fsize,
                        const short(&img_y)[H][W], const short(&img_u)[H][W / 2], const short(&img_v)[H][W / 2], const short(&img_a)[H][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = tga_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8, hhalf = W / 2, unit = 1 << D, mask = unit - 1;
        short       luma, chrm[2][2], color[3], alpha;
        int         i, j, k, pair;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                luma = img_y[i][j];
                pair = j & 1;

                if (pair == 0) {
                    k = j >> 1;
                    chrm[0][0] = chrm[1][0] = img_u[i][k];
                    chrm[0][1] = chrm[1][1] = img_v[i][k];

                    k++;
                    chrm[1][0] += k < hhalf ? img_u[i][k] : img_u[i][hhalf - 1];
                    chrm[1][1] += k < hhalf ? img_v[i][k] : img_v[i][hhalf - 1];

                    chrm[1][0] >>= 1;
                    chrm[1][1] >>= 1;
                }

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm[pair], color);

                std::fputc(color[2] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[0] >> shift, fp);

                alpha = img_a[i][j];
                if (alpha > mask)
                    alpha = mask;
                std::fputc(alpha >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D>
    void save_4224_rgba(const char* path, const char* file, const int hsize, const int fsize,
                        const short(&img_y)[H][W], const short(&img_c)[H][W], const short(&img_a)[H][W])
    {
        char    filename[200]; sprintf(filename, "%s/%s.tga", path, file);
        FILE* fp = tga_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8, unit = 1 << D, mask = unit - 1;
        short       luma, chrm[2][2], color[3], alpha;
        int         i, j, k, pair;

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                luma = img_y[i][j];
                pair = j & 1;

                if (pair == 0) {
                    chrm[0][0] = chrm[1][0] = img_c[i][j + 0];
                    chrm[0][1] = chrm[1][1] = img_c[i][j + 1];

                    k = j + 2;
                    if (k >= W)
                        k = j;

                    chrm[1][0] += img_c[i][k + 0];
                    chrm[1][1] += img_c[i][k + 1];

                    chrm[1][0] >>= 1;
                    chrm[1][1] >>= 1;
                }

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm[pair], color);

                std::fputc(color[2] >> shift, fp);
                std::fputc(color[1] >> shift, fp);
                std::fputc(color[0] >> shift, fp);

                alpha = img_a[i][j];
                if (alpha > mask)
                    alpha = mask;
                std::fputc(alpha >> shift, fp);
            }
        }
        std::fclose(fp);
    }


    template<int H, int W, int D, int A>
    void save_4224_rgb(const char* path, const char* file, const int hsize, const int fsize, const int color_num,
                       const short(&img_y)[H][W], const short(&img_c)[H][W], const short(&img_a)[H][W])
    {
        const int ALPHA_UNIT = 1 << A, ALPHA_FBIT = A;
        char    filename[200]; sprintf(filename, "%s/%s.ppm", path, file);
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        if (hsize > W || fsize > H)
            rpt::ERROR_report("image out of size");

        const int   shift = D - 8, unit = 1 << D, mask = unit - 1;
        short       luma, chrm[2][2], color[3], alpha;
        int         i, j, k, pair, i_data;

        short       bg_color[3];
        switch (color_num) {
        case 0: bg_color[0] = 1023, bg_color[1] = 1023, bg_color[2] = 1023; break;
        case 1: bg_color[0] = 1023, bg_color[1] = 0, bg_color[2] = 0; break;
        case 2: bg_color[0] = 0, bg_color[1] = 1023, bg_color[2] = 0; break;
        case 3: bg_color[0] = 0, bg_color[1] = 0, bg_color[2] = 1023; break;
        case 4: bg_color[0] = 1023, bg_color[1] = 1023, bg_color[2] = 0; break;
        case 5: bg_color[0] = 1023, bg_color[1] = 0, bg_color[2] = 1023; break;
        case 6: bg_color[0] = 0, bg_color[1] = 1023, bg_color[2] = 1023; break;
        case 7: bg_color[0] = 512, bg_color[1] = 512, bg_color[2] = 512; break;
        default:bg_color[0] = 0, bg_color[1] = 0, bg_color[2] = 0; break;
        }

        for (i = 0; i < fsize; i++) {
            for (j = 0; j < hsize; j++)
            {
                alpha = img_a[i][j];
                luma = img_y[i][j];
                pair = j & 1;

                if (pair == 0) {
                    chrm[0][0] = chrm[1][0] = img_c[i][j + 0];
                    chrm[0][1] = chrm[1][1] = img_c[i][j + 1];

                    k = j + 2;
                    if (k >= W)
                        k = j;

                    chrm[1][0] += img_c[i][k + 0];
                    chrm[1][1] += img_c[i][k + 1];

                    chrm[1][0] >>= 1;
                    chrm[1][1] >>= 1;
                }

                vio::yuv2rgb<D>(vio::ITU709, luma, chrm[pair], color);

                for (k = 0; k < 3; k++) {
                    i_data = color[k] * alpha + bg_color[k] * (ALPHA_UNIT - alpha);
                    i_data >>= ALPHA_FBIT + shift;
                    std::fputc(i_data, fp);
                }
            }
        }
        std::fclose(fp);
    }



}

#pragma warning(pop)
#endif
