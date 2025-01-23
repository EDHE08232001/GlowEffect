/*******************************************************************************************************************
 * FILE NAME   :    tools_cuda_image.h
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    a file defines cuda-oriented image load/save
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * ^^^^^^^^^^^      ^^^^^^          ^^^^^^^^
 * 2022 MAR 05      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once
#include "all_common.h"
#include <vector_types.h>

namespace cuda_vio
{
    enum RGB2YUV { ITU601, ITU709, WEBWIKI, GXB };
    typedef     RGB2YUV YUV2RGB;
    typedef     unsigned char uchar;

    // 444-to-422 down sampling filter coefficients, symetric 9 taps
    const float coef_to422[5] = { -2.7689423e-017f, -0.0022717039f, -3.1446186e-009f, 0.23516852f, 0.53420633f };

    inline void error_report(const char* info) { printf(info); exit(0); }


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
            R /= y_unit, G /= y_unit, B /= y_unit;
            Y = (0.2215f * R) + (0.7154f * G) + (0.0721f * B);
            Cb = -(0.1145f * R) - (0.3855f * G) + (0.5000f * B);
            Cr = (0.5016f * R) - (0.4556f * G) - (0.0459f * B);
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
            R = Y + 0.0000f * Cb + 1.5701f * Cr;
            G = Y - 0.1870f * Cb - 0.4667f * Cr;
            B = Y + 1.8556f * Cb + 0.0000f * Cr;
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

        FILE* fp;;
        if (fopen_s(&fp, file_name, mode))
            error_report("files doesn't exist");

        if (std::strstr(mode, "rb")) {

            fgets(buffer, 200, fp);
            if (!std::strstr(buffer, "P6"))
                error_report("not ppm file");

            do {
                fgets(buffer, 200, fp);
            } while (buffer[0] == '#');

            sscanf_s(buffer, "%d %d", &width, &height);
            if (width != hsize || height != fsize)
                error_report("image size doesn't match");

            fgets(buffer, 200, fp);
        }
        else if (std::strstr(mode, "wb")) {
            fprintf(fp, "P6\n");
            fprintf(fp, "%d %d\n", hsize, fsize);
            fprintf(fp, "255\n");
        }
        else
            error_report("file mode doesn't exist");

        return fp;
    }

    inline bool ppm_header(const char* file_name, int& hsize, int& fsize)
    {
        FILE* fp;
        if (fopen_s(&fp, file_name, "rb"))
            return false;

        char    buffer[200];
        fgets(buffer, 200, fp);
        if (!std::strstr(buffer, "P6"))
            error_report("not ppm file");

        do {
            fgets(buffer, 200, fp);
        } while (buffer[0] == '#');

        sscanf_s(buffer, "%d %d", &hsize, &fsize);
        std::fclose(fp);
        return true;
    }

    inline FILE* ppm_header(const char* file_name, const bool rnw, int& hsize, int& fsize)
    {
        char    buffer[200];

        FILE* fp;
        if (fopen_s(&fp, file_name, rnw ? "rb" : "wb"))
            error_report("files doesn't exist");

        if (rnw)
        {
            fgets(buffer, 200, fp);
            if (!std::strstr(buffer, "P6"))
                error_report("not ppm file");

            do {
                fgets(buffer, 200, fp);
            } while (buffer[0] == '#');

            sscanf_s(buffer, "%d %d", &hsize, &fsize);
            fgets(buffer, 200, fp);
        }
        else {
            fprintf_s(fp, "P6\n");
            fprintf_s(fp, "%d %d\n", hsize, fsize);
            fprintf_s(fp, "255\n");
        }

        return fp;
    }

    inline bool pgm_header(const char* file_name, int& hsize, int& fsize)
    {
        FILE* fp;
        
        if (fopen_s(&fp, file_name, "rb"))
            return false;

        char    buffer[200];
        fgets(buffer, 200, fp);
        if (!std::strstr(buffer, "P5"))
            error_report("not pgm file");

        do {
            fgets(buffer, 200, fp);
        } while (buffer[0] == '#');

        sscanf_s(buffer, "%d %d", &hsize, &fsize);
        std::fclose(fp);
        return true;
    }

    inline FILE* pgm_header(const char* file_name, const char* mode, const int hsize, const int fsize)
    {
        char    buffer[200];
        int     width, height;

        FILE* fp;
        if (fopen_s(&fp, file_name, mode))
            error_report("files doesn't exist");

        if (std::strstr(mode, "rb")) {

            fgets(buffer, 200, fp);
            if (!std::strstr(buffer, "P5"))
                error_report("not ppm file");

            do {
                fgets(buffer, 200, fp);
            } while (buffer[0] == '#');

            sscanf_s(buffer, "%d %d", &width, &height);
            if (width != hsize || height != fsize)
                error_report("image size doesn't match");

            fgets(buffer, 200, fp);
        }
        else if (std::strstr(mode, "wb")) {
            fprintf(fp, "P5\n");
            fprintf(fp, "%d %d\n", hsize, fsize);
            fprintf(fp, "255\n");
        }
        else
            error_report("file mode doesn't exist");

        return fp;
    }


    inline void load_pgm(const char* file_name, const int hsize, const int fsize, float* dout)
    {
        FILE* fp;
        if (fopen_s(&fp, file_name, "rb"))
            return;

        char    buffer[200];
        fgets(buffer, 200, fp);
        if (!std::strstr(buffer, "P5"))
            error_report("not pgm file");

        do {
            fgets(buffer, 200, fp);
        } while (buffer[0] == '#');

        fgets(buffer, 200, fp);

        const int size = hsize * fsize;
        uint8_t* data = new uint8_t[size];
        std::fread(data, 1, size, fp);
        std::transform(data, data + size, dout, [](uint8_t din) {return (float)din; });

        delete[] data;
        fclose(fp);
    }

    inline void load_ppm2rgb(const char* filename, const int hsize, const int fsize, uint8_t* const dout)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        fread(dout, 3, hsize * fsize, fp);
        fclose(fp);
    }

    inline void load_ppm2rgba(const char* filename, const int hsize, const int fsize, uchar4* const dout)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                dout[n].x = fgetc(fp);
                dout[n].y = fgetc(fp);
                dout[n].z = fgetc(fp);
                dout[n].w = 255;
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2rgba_2field(const char* filename, const int hsize, const int fsize, short4* const dout)
    {
        const int offset = fsize * hsize;
        FILE* fp = ppm_header(filename, "rb", hsize, fsize * 2);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int field = 0; field < 2; field++) {
                for (int j = 0, n = m + field * offset; j < hsize; j++, n++) {
                    dout[n].x = (short)fgetc(fp);
                    dout[n].y = (short)fgetc(fp);
                    dout[n].z = (short)fgetc(fp);
                    dout[n].w = 255;
                }
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2rgba(const char* filename, const int hsize, const int fsize, float4* const dout)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                dout[n].x = (float)fgetc(fp);
                dout[n].y = (float)fgetc(fp);
                dout[n].z = (float)fgetc(fp);
                dout[n].w = 1.f;
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2rgba_2field(const char* filename, const int hsize, const int fsize, float4* const dout)
    {
        const int offset = hsize * fsize;
        FILE* fp = ppm_header(filename, "rb", hsize, fsize * 2);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int field = 0; field < 2; field++) {
                for (int j = 0, n = m + field * offset; j < hsize; j++, n++) {
                    dout[n].x = (float)fgetc(fp);
                    dout[n].y = (float)fgetc(fp);
                    dout[n].z = (float)fgetc(fp);
                    dout[n].w = 1.f;
                }
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2rgba_cube(const char* filename, const int hsize, const int fsize, float4* const dout)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        int n = 0;
        for (int i = 0; i < fsize; i++) {
            for (int j = 0; j < hsize; j++) {
                if (j < fsize) {
                    dout[n].x = (float)fgetc(fp);
                    dout[n].y = (float)fgetc(fp);
                    dout[n].z = (float)fgetc(fp);
                    dout[n].w = 1.f;
                    n++;
                }
                else {
                    fgetc(fp);
                    fgetc(fp);
                    fgetc(fp);
                }
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2rgba(const char* fnm, const int hsize, const int fsize, const int block[4], const int trnsp, uchar4* const dout)
    {
        FILE* fp = ppm_header(fnm, "rb", hsize, fsize);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                dout[n].x = fgetc(fp);
                dout[n].y = fgetc(fp);
                dout[n].z = fgetc(fp);
                dout[n].w = (j < block[0] || j > block[1] || i <block[2] || i >block[3]) ? trnsp : 255;
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2rgba(const char* filename, const int hsize, const int fsize, const int block[4], const float trnsp, float4* const dout)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                dout[n].x = (float)fgetc(fp);
                dout[n].y = (float)fgetc(fp);
                dout[n].z = (float)fgetc(fp);
                dout[n].w = (j < block[0] || j > block[1] || i <block[2] || i >block[3]) ? trnsp : 1.f;
            }
        }
        std::fclose(fp);
    }

    inline void load_ppm2yca(const char* filename, const int hsize, const int fsize, int* const dout)
    {
        FILE* fp = ppm_header(filename, "rb", hsize, fsize);

        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++)
            {
                short color[3], luma, chrm[2];
                for (int k = 0; k < 3; k++)
                    color[k] = (short)fgetc(fp);
                rgb2yuv<8>(ITU709, color, luma, chrm);
                int data = luma & 0x3ff | ((chrm[j & 1] & 0x3ff) << 10) | (255 << 20);
                dout[n] = data;
            }
        }
        std::fclose(fp);
    }




    inline void save_pgm(const char* file, const int hsize, const int fsize, bool* const din)
    {
        FILE* fp = pgm_header(file, "wb", hsize, fsize);
        for (int k = 0; k < hsize * fsize; k++)
            fputc(din[k] ? 255 : 0, fp);
        fclose(fp);
    }

    inline void save_pgm(const char* file, const int hsize, const int fsize, char* const din)
    {
        FILE* fp = pgm_header(file, "wb", hsize, fsize);
        fwrite(din, 1, hsize * fsize, fp);
        fclose(fp);
    }

    inline void save_pgm(const char* file, const int hsize, const int fsize, uint8_t* const din)
    {
        FILE* fp = pgm_header(file, "wb", hsize, fsize);
        fwrite(din, 1, hsize * fsize, fp);
        fclose(fp);
    }

    template<int D = 8>
    void save_pgm(const char* file, const int hsize, const int fsize, float* const din)
    {
        const int unit = 1 << D;
        const int mask = unit - 1;

        FILE* fp = pgm_header(file, "wb", hsize, fsize);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                float data = din[n];
                fputc((int)(data * mask), fp);
            }
        }
        fclose(fp);
    }

    inline void save_rgb2ppm(const char* filename, const int hsize, const int fsize, uint8_t* const din)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);

        fwrite(din, 3, hsize * fsize, fp);
        fclose(fp);
    }

    inline void save_rgb2ppm(const char* filename, const int hsize, const int fsize, const int bg_color[3],
                             uint8_t* const video, uint8_t* const alpha)
    {
        FILE* fp = ppm_header(filename, "wb", hsize, fsize);
        for (int i = 0, m = 0, u = 0; i < fsize; i++, m += hsize * 3, u += hsize) {
            for (int j = 0, n = m, v = u; j < hsize; j++, n += 3, v++)
            {
                int fg_color[3] = { video[n], video[n + 1], video[n + 2] };
                for (int k = 0; k < 3; k++) {
                    fg_color[k] = bg_color[k] + (fg_color[k] - bg_color[k]) * alpha[v] / 255;
                    fputc(fg_color[k], fp);
                }
            }
        }
        fclose(fp);
    }

    inline void save_rgba2ppm(const char* file, const int hsize, const int fsize, uchar4* const din)
    {
        std::string vname(file); vname += ".ppm";
        std::string aname(file); aname += ".pgm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, fsize);
        FILE* fp_alpha = pgm_header(aname.c_str(), "wb", hsize, fsize);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                fputc(din[n].x, fp_video);
                fputc(din[n].y, fp_video);
                fputc(din[n].z, fp_video);
                fputc(din[n].w, fp_alpha);
            }
        }
        fclose(fp_video);
        fclose(fp_alpha);
    }

    inline void save_yuva2ppm(const char* file, const int hsize, const int fsize, short4* const din)
    {
        std::string vname(file); vname += ".ppm";
        std::string aname(file); aname += ".pgm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, fsize);
        FILE* fp_alpha = pgm_header(aname.c_str(), "wb", hsize, fsize);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                short luma = din[n].x;
                short chrm[2] = { din[n].y, din[n].z };
                short color[3];
                yuv2rgb<8>(YUV2RGB::ITU709, luma, chrm, color);
                fputc(color[0], fp_video);
                fputc(color[1], fp_video);
                fputc(color[2], fp_video);
                fputc(din[n].w, fp_alpha);
            }
        }
        fclose(fp_video);
        fclose(fp_alpha);
    }

    inline void save_rgba2ppm_2field(const char* file, const int hsize, const int fsize, short4* const din)
    {
        const int offset = hsize * fsize;

        std::string vname(file); vname += ".ppm";
        std::string aname(file); aname += ".pgm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, fsize*2);
        FILE* fp_alpha = pgm_header(aname.c_str(), "wb", hsize, fsize*2);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int field = 0; field < 2; field++) {
                for (int j = 0, n = m + field * offset; j < hsize; j++, n++) {
                    fputc(std::min<short>(255, std::max<short>(0, din[n].x)), fp_video);
                    fputc(std::min<short>(255, std::max<short>(0, din[n].y)), fp_video);
                    fputc(std::min<short>(255, std::max<short>(0, din[n].z)), fp_video);
                    fputc(std::min<short>(255, std::max<short>(0, din[n].w)), fp_alpha);
                }
            }
        }
        fclose(fp_video);
        fclose(fp_alpha);
    }

    inline void save_rgba2ppm(const char* file, const int hsize, const int fsize, const int bg_color[3], uchar4* const din)
    {
        std::string vname(file); vname += ".ppm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, fsize);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                int fg_color[3] = { din[n].x, din[n].y, din[n].z };
                for (int k = 0; k < 3; k++) {
                    fg_color[k] = bg_color[k] + (fg_color[k] - bg_color[k]) * din[n].w / 255;
                    if (fg_color[k] > 255)    fg_color[k] = 255;
                    else if (fg_color[k] < 0) fg_color[k] = 0;
                    fputc(fg_color[k], fp_video);
                }
            }
        }
        fclose(fp_video);
    }

    inline void save_rgba2ppm_2field(const char* file, const int hsize, const int vsize, const int bg_color[3], short4* const din)
    {
        const int offset = hsize * vsize;

        std::string vname(file); vname += ".ppm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, vsize * 2);
        for (int i = 0, m = 0; i < vsize; i++, m += hsize) {
            for (int field = 0; field < 2; field++) {
                for (int j = 0, n = m + field * offset; j < hsize; j++, n++) {
                    int fg_color[3] = { din[n].x, din[n].y, din[n].z };
                    for (int k = 0; k < 3; k++) {
                        fg_color[k] = bg_color[k] + (fg_color[k] - bg_color[k]) * din[n].w / 255;
                        if (fg_color[k] > 255)  fg_color[k] = 255;
                        else if (fg_color[k] < 0) fg_color[k] = 0;
                        fputc((int)fg_color[k], fp_video);
                    }
                }
            }
        }
        fclose(fp_video);
    }

    inline void save_rgba2ppm(const char* file, const int hsize, const int fsize, float4* const din)
    {
        std::string vname(file); vname += ".ppm";
        std::string aname(file); aname += ".pgm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, fsize);
        FILE* fp_alpha = pgm_header(aname.c_str(), "wb", hsize, fsize);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m, data; j < hsize; j++, n++) {
                data = (int)std::min(255.f, std::max<float>(0, din[n].x)); fputc(data, fp_video);
                data = (int)std::min(255.f, std::max<float>(0, din[n].y)); fputc(data, fp_video);
                data = (int)std::min(255.f, std::max<float>(0, din[n].z)); fputc(data, fp_video);
                data = std::min(255, std::max(0, (int)(din[n].w * 255))); fputc(data, fp_alpha);
            }
        }
        fclose(fp_video);
        fclose(fp_alpha);
    }

    inline void save_rgba2ppm(const char* file, const int hsize, const int fsize, const int bg_color[3], float4* const din)
    {
        std::string vname(file); vname += ".ppm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, fsize);
        for (int i = 0, m = 0; i < fsize; i++, m += hsize) {
            for (int j = 0, n = m; j < hsize; j++, n++) {
                float fg_color[3] = { din[n].x, din[n].y, din[n].z };
                for (int k = 0; k < 3; k++) {
                    fg_color[k] = bg_color[k] + (fg_color[k] - bg_color[k]) * din[n].w;
                    if (fg_color[k] > 255.f)  fg_color[k] = 255.f;
                    else if (fg_color[k] < 0) fg_color[k] = 0;
                    fputc((int)fg_color[k], fp_video);
                }
            }
        }
        fclose(fp_video);
    }

    inline void save_rgba2ppm_2field(const char* file, const int hsize, const int vsize, const int bg_color[3], float4* const din)
    {
        const int offset = hsize * vsize;

        std::string vname(file); vname += ".ppm";
        FILE* fp_video = ppm_header(vname.c_str(), "wb", hsize, vsize * 2);
        for (int i = 0, m = 0; i < vsize; i++, m += hsize) {
            for (int field = 0; field < 2; field++) {
                for (int j = 0, n = m + field * offset; j < hsize; j++, n++) {
                    float fg_color[3] = { din[n].x, din[n].y, din[n].z };
                    for (int k = 0; k < 3; k++) {
                        fg_color[k] = bg_color[k] + (fg_color[k] - bg_color[k]) * din[n].w;
                        if (fg_color[k] > 255.f)  fg_color[k] = 255.f;
                        else if (fg_color[k] < 0) fg_color[k] = 0;
                        fputc((int)fg_color[k], fp_video);
                    }
                }
            }
        }
        fclose(fp_video);
    }

    inline void synth_rgb_chess(const int hsize, const int fsize, uint8_t* const dout, const int pttn_size = 1)
    {
        const int stride = 3 * hsize;
        const uint8_t r[8] = { 255, 0,   0,   255, 255, 0,  255, 128 };
        const uint8_t g[8] = { 0,   255, 0,   255, 0,   255,255, 128 };
        const uint8_t b[8] = { 0,   0,   255, 0,   255, 255,255, 128 };
        for (int i = 0, m = 0, v_id = 0; i < fsize; i++, m += stride) {
            if (i % pttn_size == 0 && i) {
                v_id++;
                v_id %= 8;
            }
            for (int j = 0, n = m, h_id = v_id; j < hsize; j++, n += 3) {
                if (j % pttn_size == 0 && j) {
                    h_id++;
                    h_id %= 8;
                }
                dout[n + 0] = r[h_id];
                dout[n + 1] = g[h_id];
                dout[n + 2] = b[h_id];
            }
        }
    }

    inline void synth_rgba_chess(const int hsize, const int fsize, uchar4* const dout, const int pttn_size = 16)
    {
        const int stride = hsize;
        const uint8_t r[8] = { 255, 0,   0,   255, 255, 0,  255, 128 };
        const uint8_t g[8] = { 0,   255, 0,   255, 0,   255,255, 128 };
        const uint8_t b[8] = { 0,   0,   255, 0,   255, 255,255, 128 };
        for (int i = 0, m = 0, v_id = 0; i < fsize; i++, m += stride) {
            if (i % pttn_size == 0 && i) {
                v_id++;
                v_id %= 8;
            }
            for (int j = 0, n = m, h_id = v_id; j < hsize; j++, n++) {
                if (j % pttn_size == 0 && j) {
                    h_id++;
                    h_id %= 8;
                }
                dout[n].x = r[h_id];
                dout[n].y = g[h_id];
                dout[n].z = b[h_id];
                dout[n].w = 255;
            }
        }
    }

}