/*******************************************************************************************************************
 * FILE NAME   :    tools_cuda_synth.h
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    a file defines cuda-oriented synthesized image load/save
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * ^^^^^^^^^^^      ^^^^^^          ^^^^^^^^
 * 2022 MAR 05      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#pragma once
#include "mixtree_common.h"

namespace cuda_syn
{
    constexpr uchar4 color[N_CHNL]{
        {255,0,0,255}, {0,255,0,255}, {0,0,255,255}, {255,255,0,255},
        {255,0,255,255}, {0,255,255,255}, {220,220,220,255}, {100,100,100,255}
    };

    inline void multi_color(const int w, const int h, const int id, uchar4* dout)
    {
        for (int i = 0, m = 0; i < h; i++, m += w) {
            for (int j = 0, n = m; j < w; j++, n++)
                dout[n] = color[id % N_CHNL];
        }
    }

    inline void multi_color(const int w, const int h, const int id, float4* dout)
    {
        for (int i = 0, m = 0; i < h; i++, m += w) {
            for (int j = 0, n = m; j < w; j++, n++) {
                dout[n].x = color[id % N_CHNL].x;
                dout[n].y = color[id % N_CHNL].y;
                dout[n].z = color[id % N_CHNL].z;
                dout[n].w = color[id % N_CHNL].w / 255.f;
            }
        }
    }

    inline void multi_color(const int w, const int h, const int id, const int block[4], const int trnsp, uchar4* dout)
    {
        for (int i = 0, m = 0; i < h; i++, m += w) {
            for (int j = 0, n = m; j < w; j++, n++) {
                if (j<block[0] || j>block[1] || i<block[2] || i>block[3])
                    dout[n].x = color[id].x,
                    dout[n].y = color[id].y,
                    dout[n].z = color[id].z,
                    dout[n].w = trnsp;
                else
                    dout[n] = color[id];
            }
        }
    }

    inline void multi_color(const int w, const int h, const int id, const int block[4], const int trnsp, float4* dout)
    {
        for (int i = 0, m = 0; i < h; i++, m += w) {
            for (int j = 0, n = m; j < w; j++, n++) {
                if (j<block[0] || j>block[1] || i<block[2] || i>block[3])
                    dout[n].x = color[id].x,
                    dout[n].y = color[id].y,
                    dout[n].z = color[id].z,
                    dout[n].w = trnsp * color[id].w / 255.f;
                else
                    dout[n].x = color[id % N_CHNL].x,
                    dout[n].y = color[id % N_CHNL].y,
                    dout[n].z = color[id % N_CHNL].z,
                    dout[n].w = color[id % N_CHNL].w / 255.f;
            }
        }
    }
}
