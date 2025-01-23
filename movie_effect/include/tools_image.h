/*******************************************************************************************************************
 * FILE NAME   :    designA.cpp
 *
 * PROJECTION  :    DVE3G
 *
 * DESCRIPTION :    Design A is a design only with a single DVE channel.
 *                  
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2013 JAN 17      Yu Liu          Creation
 * 2013 JAN 21      Yu Liu          Striped dspInst away from dveChannel and defined it in the design level
 *
 ********************************************************************************************************************/
#ifndef __TOOLS_IMAGE_H__
#define __TOOLS_IMAGE_H__

#include "all_common.h"

namespace synimg{

    template<int H, int W>
    void initial_luma( const short init_data, const int hsize, const int fsize, short (&data_inout)[H][W] )
    {
        int     line_cnt, pix_cnt;
        for( line_cnt=0; line_cnt<fsize; line_cnt++ ){
        for( pix_cnt=0; pix_cnt<hsize; pix_cnt++ )
            data_inout[line_cnt][pix_cnt] = init_data;
        }
    }

    template<int H, int W, int D>
    void image_hor_bar( const int hsize, const int fsize, const int bar_num, const bool sin2bin, const float soft, const bool color,
                        short (&luma)[H][W], short (&chrm)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        int     data, line_cnt, line_num;
        float   dbl_data;

        for( line_cnt=0; line_cnt<fsize; line_cnt++ )
        {
            line_num = line_cnt;
            
            for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++)
            {
                dbl_data = sin( PI *bar_num *pix_cnt /hsize);
                if(sin2bin)
                    dbl_data = fabs(dbl_data)<=soft? dbl_data/soft : dbl_data<0? -1 : 1;

                dbl_data += 1;
                dbl_data /= 2.5;        // reduces dynamic range
                dbl_data *= mask;       // in order to give
                dbl_data += 1<<(D-4);   // offsets for black and white

                data = int(dbl_data);

                luma[line_num][pix_cnt] = data;
                chrm[line_num][pix_cnt] = color? (data-half) : 0;
            }
        }
    }

    template<int H, int W, int D>
    void image_hor_bar( const int hsize, const int fsize, const int bar_num, const bool sin2bin, const float soft, const bool color,
                        short (&R)[H][W], short (&G)[H][W], short (&B)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1;
        int     data, line_cnt, line_num;
        float   dbl_data;
        
        for( line_cnt=0; line_cnt<fsize; line_cnt++ )
        {
            line_num = line_cnt;
            
            for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++)
            {
                dbl_data = sin( PI *bar_num *pix_cnt /hsize);
                if(sin2bin)
                    dbl_data = fabs(dbl_data)<=soft? dbl_data/soft : dbl_data<0? -1 : 1;

                dbl_data += 1;
                dbl_data /= 2.5;        // reduces dynamic range
                dbl_data *= mask;       // in order to give
                dbl_data += 1<<(D-4);   // offsets for black and white

                data = int(dbl_data);

                R[line_num][pix_cnt] = data;
                G[line_num][pix_cnt] = data;
                B[line_num][pix_cnt] = data;
            }
        }
    }

    template<int H, int W, int D>
    void image_ver_bar( const int hsize, const int fsize, const int bar_num, const bool sin2bin, const float soft, bool color, 
                        short (&luma)[H][W], short (&chrm)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        int     data, line_cnt, line_num;
        float   dbl_data;
        
        for( line_cnt=0; line_cnt<fsize; line_cnt++)
        {
            dbl_data = sin(PI *bar_num *line_cnt /fsize);
            
            if(sin2bin)
                dbl_data = fabs(dbl_data)<=soft? dbl_data/soft : (dbl_data<0? -1 : 1);

            dbl_data += 1;
            dbl_data /= 2.5;        // reduces dynamic range
            dbl_data *= mask;       // in order to give
            dbl_data += 1<<(D-4);   // offsets for black and white

            data = int(dbl_data);

            line_num = line_cnt;
            for(int pix_cnt=0; pix_cnt<W; pix_cnt++)
            {
                luma[line_num][pix_cnt] = data;
                chrm[line_num][pix_cnt] = color? (data-half) : 0;
            }
        }
    }

    template<int H, int W, int D>
    void image_ver_bar( const int hsize, const int fsize, const int bar_num, const bool sin2bin, const float soft, 
                        short (&R)[H][W], short (&G)[H][W], short (&B)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        int     data, line_cnt, line_num;
        float   dbl_data;
        
        for( line_cnt=0; line_cnt<fsize; line_cnt++)
        {
            dbl_data = sin(PI *bar_num *line_cnt /fsize);
            
            if(sin2bin)
                dbl_data = fabs(dbl_data)<=soft? dbl_data/soft : (dbl_data<0? -1 : 1);

            dbl_data += 1;
            dbl_data /= 2.5;        // reduces dynamic range
            dbl_data *= mask;       // in order to give
            dbl_data += 1<<(D-4);   // offsets for black and white

            data = int(dbl_data);

            line_num = line_cnt;
            for(int pix_cnt=0; pix_cnt<W; pix_cnt++)
            {
                R[line_num][pix_cnt] = data;
                G[line_num][pix_cnt] = data;
                B[line_num][pix_cnt] = data;
            }
        }
    }

    template<int H, int W, int D>
    void image_checker( const int hsize, const int fsize, const int bar_hnum, const int bar_vnum, const bool sin2bin, const float soft, bool color,
                        short (&Y)[H][W], short (&C)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        float   vdata, hdata, dbl_data;
        int     data, line_cnt, line_num;

        for( line_cnt=0; line_cnt<fsize; line_cnt++ )
        {
            vdata = sin(PI *bar_vnum *line_cnt /fsize);
            
            if(sin2bin)
                vdata = fabs(vdata)<=soft? vdata/soft : vdata<0? -1 : 1;

            line_num = line_cnt;
            for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++)
            {
                hdata = sin(PI *bar_hnum *pix_cnt /hsize);
                
                if(sin2bin)
                    hdata = fabs(hdata)<=soft? hdata/soft : hdata<0? -1 : 1;

                dbl_data = hdata * vdata;

                dbl_data += 1;
                dbl_data /= 2.5;        // reduces dynamic range
                dbl_data *= mask;       // in order to give
                dbl_data += 1<<(D-4);   // offsets for black and white

                data = int(dbl_data);

                Y[line_num][pix_cnt] = data;
                C[line_num][pix_cnt] = color? (data - half) : 0;
            }
        }
    }

    template<int H, int W, int D>
    void image_checker( const int hsize, const int fsize, const int bsize, const bool sin2bin, const float soft, bool color,
                        short (&Y)[H][W], short (&C)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        float   vdata, hdata, dbl_data;
        int     data, line_cnt, line_num;

        for( line_cnt=0; line_cnt<fsize; line_cnt++ )
        {
            vdata = sin(PI * line_cnt /bsize);
            
            if(sin2bin)
                vdata = fabs(vdata)<=soft? vdata/soft : vdata<0? -1 : 1;

            line_num = line_cnt;
            for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++)
            {
                hdata = sin(PI * pix_cnt /bsize);
                
                if(sin2bin)
                    hdata = fabs(hdata)<=soft? hdata/soft : hdata<0? -1 : 1;

                dbl_data = hdata * vdata;

                dbl_data += 1;
                dbl_data /= 2.5;        // reduces dynamic range
                dbl_data *= mask;       // in order to give
                dbl_data += 1<<(D-4);   // offsets for black and white

                data = int(dbl_data);

                Y[line_num][pix_cnt] = data;
                C[line_num][pix_cnt] = color? (data - half) : 0;
            }
        }
    }


    template<int H, int W, int D>
    void image_checker( const int hsize, const int fsize, const int bar_hnum, const int bar_vnum, const bool sin2bin, const float soft,
                        short (&R)[H][W], short (&G)[H][W], short (&B)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        float   vdata, hdata, dbl_data;
        int     data, line_cnt, line_num;

        for( line_cnt=0; line_cnt<fsize; line_cnt++ )
        {
            vdata = sin(PI *bar_vnum *line_cnt /fsize);
            
            if(sin2bin)
                vdata = fabs(vdata)<=soft? vdata/soft : vdata<0? -1 : 1;

            line_num = line_cnt;
            for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++)
            {
                hdata = sin(PI *bar_hnum *pix_cnt /hsize);
                
                if(sin2bin)
                    hdata = fabs(hdata)<=soft? hdata/soft : hdata<0? -1 : 1;

                dbl_data = hdata * vdata;

                dbl_data += 1;
                dbl_data /= 2.5;        // reduces dynamic range
                dbl_data *= mask;       // in order to give
                dbl_data += 1<<(D-4);   // offsets for black and white

                data = int(dbl_data);

                R[line_num][pix_cnt] = data;
                G[line_num][pix_cnt] = data;
                B[line_num][pix_cnt] = data;
            }
        }
    }

    template<int H, int W, int D>
    void image_checker( const int hsize, const int fsize, const int bsize, const bool sin2bin, const float soft,
                        short (&R)[H][W], short (&G)[H][W], short (&B)[H][W] )
    {
        const int   unit=1<<D, mask=unit-1, half=unit>>1;
        float   vdata, hdata, dbl_data;
        int     data, line_cnt, line_num;

        for( line_cnt=0; line_cnt<fsize; line_cnt++ )
        {
            vdata = sin(PI * line_cnt /bsize);
            
            if(sin2bin)
                vdata = fabs(vdata)<=soft? vdata/soft : vdata<0? -1 : 1;

            line_num = line_cnt;
            for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++)
            {
                hdata = sin(PI * pix_cnt /bsize);
                
                if(sin2bin)
                    hdata = fabs(hdata)<=soft? hdata/soft : hdata<0? -1 : 1;

                dbl_data = hdata * vdata;

                dbl_data += 1;
                dbl_data /= 2.5;        // reduces dynamic range
                dbl_data *= mask;       // in order to give
                dbl_data += 1<<(D-4);   // offsets for black and white

                data = int(dbl_data);

                R[line_num][pix_cnt] = data;
                G[line_num][pix_cnt] = data;
                B[line_num][pix_cnt] = data;
            }
        }
    }


/*
void single_hline(const int &line_num, const int &line_width,
                              short output_luma[H][W], short output_chrm[H][W] ){

    int     data, row_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        data = (line_cnt>(line_num-line_width) && line_cnt<(line_num+line_width) )? 0x3ac : 64;

        row_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){
            output_luma[row_num][pix_cnt] = data;
            output_chrm[row_num][pix_cnt] = noColor? 0 : (data - half);
        }
    }
}

void single_vline(const int &pixel_num, const int &pixel_width,
                              short output_luma[H][W], short  output_chrm[H][W] ){

    int     data, line_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            data = (pix_cnt>(pixel_num-pixel_width) && pix_cnt<(pixel_num+pixel_width) )? 0x3ac : 64;

            output_luma[line_num][pix_cnt] = data;
            output_chrm[line_num][pix_cnt] = noColor? 0 : (data - half);
        }
    }
}

void frame_bounds(const int &bound_width,
                              short output_luma[H][W], short  output_chrm[H][W] ){

    short   vdata, hdata, line_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        vdata = (line_cnt>=(fsize-bound_width) || line_cnt<bound_width)? 0x3ff : 64;
        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            hdata = (pix_cnt>=(hsize-bound_width) || pix_cnt<bound_width || vdata)? 0x3ac : 64;

            output_luma[line_num][pix_cnt] = hdata;
            output_chrm[line_num][pix_cnt] = noColor? 0 : (hdata-512);
        }
    }
}

void one_solid_color( short output_luma[H][W], short  output_chrm[H][W] ){

    int line_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            output_luma[line_num][pix_cnt] = solidLuma[0];
            output_chrm[line_num][pix_cnt] = noColor? 0 : solidChrm[0][pix_cnt&1];
        }
    }
}

void two_solid_color( short output_luma[H][W], short  output_chrm[H][W] ){

    int     line_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            output_luma[line_num][pix_cnt] = (line_cnt & 1)? solidLuma[1] : solidLuma[0];
            output_chrm[line_num][pix_cnt] = (line_cnt & 1)? solidChrm[0][pix_cnt&1] : solidChrm[0][pix_cnt&1];
        }
    }
}

void diagonal_pattern(const int &edge_size,
                                  short output_luma[H][W], short output_chrm[H][W]){

    int     line_cnt, pix_cnt, line_num;
    float  x, y;
    short   luma[2] = {0x2f0, 0x40};
    short   chrm[2][2] = { {0x200, 0x10}, {0x20, 0x100} };

    for(y=-fsize/2, line_cnt=0; line_cnt<fsize; line_cnt++, y++){

        line_num = line_cnt;
        for(x=-hsize/2, pix_cnt=0; pix_cnt<hsize; pix_cnt++, x++){

            if(pix_cnt>line_cnt){
                output_luma[line_num][pix_cnt] = luma[0];
                output_chrm[line_num][pix_cnt] = noColor? 0 : chrm[0][pix_cnt&1];
            }else{
                output_luma[line_num][pix_cnt] = luma[1];
                output_chrm[line_num][pix_cnt] = noColor? 0 : chrm[1][pix_cnt&1];
            }
        }
    }
}

void color_wheel_1(short output_luma[H][W], short  output_chrm[H][W]){

    int         line_cnt, pix_cnt, line_num;
    short       luma, chrm[2];
    float       data, x, y, u, v, w;
    float       radius = fsize/2;
    float       init_x = hsize/2;

    for(line_cnt=0, y=radius; line_cnt<fsize; y--, line_cnt++){

        v = y*ASPECT_VSCALE/radius;
        line_num = line_cnt;

        for(pix_cnt=0, x=-init_x; pix_cnt<hsize; x++, pix_cnt++){

            u = x*ASPECT_HSCALE/radius;
            w = sqrt(u*u+v*v);

            if(w>1){       //fabs(u)>1 || fabs(v)>1){    //
                luma =chrm[0] = chrm[1] = 0;
            }else{
                chrm[0] = (short)(u*512.0);
                chrm[1] = (short)(v*512.0);
                data = (1-w)*1023;
                if(data<0)          luma =0;
                else if(data>1023)  luma =1023;
                else                luma =short(data);
            }
            output_luma[line_num][pix_cnt] = luma;
            output_chrm[line_num][pix_cnt] = noColor? 0 : chrm[pix_cnt&1];
        }
    }
}



void sin_image_hor(short output_luma[H][W], short output_chrm[H][W]){

    float   data;
    int     line_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            //data = sin(2*PI*horBarNum*pix_cnt/hsize);
            data = sin(2.0f * PI * 120000.0f/(line_cnt*12.5f + 1000.0f)*(pix_cnt-hsize/2.0f)/hsize);
            if(sin2bin)
                data = fabs(data)<=softCut? data/softCut : data<0? -1.0f : 1.0f;
            data += 1.0f;
            data /= 2.5f;       // reduces dynamic range
            data *= 1023.0f;    // in order to give
            data += 102.0f;     // offsets for black and white 
            output_luma[line_num][pix_cnt] = short(data);
            output_chrm[line_num][pix_cnt] = noColor? 0 : short(data-512.0f);
        }
    }
}


void sin_image_ver(short output_luma[H][W], short output_chrm[H][W] ){

    float      data;
    for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

        for(int line_cnt=0; line_cnt<fsize; line_cnt++){
            
            data = sin(2.0f * PI * 68000.0f / (pix_cnt * 8.5f + 1000.0f) * ((float)line_cnt-(float)fsize/2.0f)/fsize);
            if(sin2bin)
                data = fabs(data)<=softCut? data/softCut : data<0? -1.0f : 1.0f;
            data += 1.0f;
            data /= 2.5f;    // reduces dynamic range
            data *= 1023.0f;   // in order to give
            data += 102.0f;    // offsets for black and white 

            output_luma[line_cnt][pix_cnt] = short(data);
            output_chrm[line_cnt][pix_cnt] = noColor? 0 : short(data-512.0f);
        }
    }
}

void sin_image_all(short output_luma[H][W], short output_chrm[H][W]){

    float       vdata, hdata;
    int         data, line_num;
    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        // vertical data is sinual wave
        vdata = sin(2.0f * PI * verBarNum * line_cnt / fsize);
        if(sin2bin)
            vdata = fabs(vdata)<=softCut? vdata/softCut : vdata<0? -1.0f : 1.0f;
        vdata += 1.0f;
        vdata /= 2.5f;      // reduces dynamic range
        vdata *= 1023.0f;   // in order to give
        vdata += 102.0f;    // offsets for black and white 

        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            // horizontal data is sinual wave
            hdata = sin(2.0f * PI * horBarNum * pix_cnt / hsize);
            if(sin2bin)
                hdata = fabs(hdata)<=softCut? hdata/softCut : hdata<0? -1.0f : 1.0f;
            hdata += 1.0f;
            hdata /= 2.5f;      // reduces dynamic range
            hdata *= 1023.0f;   // in order to give
            hdata += 102.0f;    // offsets for black and white 

            // vertical data modulates horizontal data
            data = (int)(hdata * vdata);
            data /= 1023;

            if(data>1023)   data = 1023;
            if(data<0)      data = 0;

            output_luma[line_num][pix_cnt] = data;
            output_chrm[line_num][pix_cnt] = noColor? 0 : short(data - half);
        }
    }
}

void ramp_image_hor(short output_luma[H][W], short output_chrm[H][W]){

    int     pix_cnt, luma, chrm, luma_inc, chrm_inc, line_num;

    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        line_num = line_cnt;
        luma_inc = hsize <1024? 2 : 1;
        chrm_inc = 1;

        for(pix_cnt=0, luma=1, chrm=4; pix_cnt<hsize; pix_cnt++){

            output_luma[line_num][pix_cnt] = luma;
            output_chrm[line_num][pix_cnt] = noColor? 0 : chrm-512;

            if(chrm>1022)       chrm_inc =-1;
            else if(chrm<4)     chrm_inc = 1;
            chrm += chrm_inc;

            luma_inc *= (luma>=0x3ac)? -1 : 1;
            luma += luma_inc;
            if(luma<1){
                luma_inc *= -1;
                luma = 1;
            }

        }
    }
}

void box_frame_ramp( short output_luma[H][W], short output_chrm[H][W] ){

    int     line_cnt, line_num, pix_cnt;
    float   x, y, diff_x, diff_y, data_x, data_y, data;

    for(line_cnt=0, y=fsize/2; line_cnt<fsize; line_cnt++, y--){

        diff_y = fabs(y) - rampSize[0];
        if(diff_y<0)
            data_y = 0;
        else{
            diff_y -= rampSize[1];
            data_y = diff_y>0.0f? 0.0f : (diff_y / rampSize[1]);
            data_y += 1.0f;
        }

        line_num = line_cnt;
        for(pix_cnt=0, x=-hsize/2; pix_cnt<hsize; pix_cnt++, x++){

            diff_x = fabs(x) - rampSize[0];
            if(diff_x<0)
                data_x = 0.0f;
            else{
                diff_x -= rampSize[1];
                data_x = diff_x>0.0f? 0.0f : (diff_x / rampSize[1]);
                data_x += 1.0f;
            }
            data = data_x + data_y - data_x * data_y;
            output_luma[line_num][pix_cnt] = int(1023.0f * data);
            output_chrm[line_num][pix_cnt] = 0;
        }
    }
}



const float     BURST_NUM =40.0f;
const float     THICKNESS =3.0f;
const float     BLK_LEVEL =0.0f;
const float     SHARPNESS = 1.0f;

void radiate_image(short output_luma[H][W], short  output_chrm[H][W]){

    int         line_cnt, pix_cnt, line_num, draw_num;
    float       angle, y, x;
    float       cos_a, sin_a, norm, proj_x, proj_y;
    float       diff_x, diff_y, data_flt;
    short       data_int, data;

    fprintf(stdout, "start generating radiate image\n");
    for(line_cnt=0; line_cnt<fsize; line_cnt++){
        for(pix_cnt=0; pix_cnt<hsize; pix_cnt++){
            output_luma[line_cnt][pix_cnt] = 0;
            output_chrm[line_cnt][pix_cnt] = 0;
        }
    }

    for(angle=PI/2.0f - 0.0001f, draw_num=0; draw_num<(int)BURST_NUM; angle-=PI/BURST_NUM, draw_num++){

        fprintf(stdout,"drawing line %d\n", draw_num);

        cos_a = cos(angle);
        sin_a = sin(angle);

        for( y=fsize/2.0f, line_cnt=0; line_cnt<fsize; y--, line_cnt++){

           line_num = line_cnt;
           for( x=0, pix_cnt=0; x<hsize; x++, pix_cnt++){

                // projection
                norm = x * cos_a + y * sin_a;

                proj_x = norm * cos_a;
                proj_y = norm * sin_a;

                diff_x = proj_x - x;
                diff_y = proj_y - y;

                norm = sqrt(diff_x * diff_x + diff_y * diff_y);

                data_flt = norm>THICKNESS? BLK_LEVEL : ( (cos(PI * norm/THICKNESS) +1)*SHARPNESS*(1024-BLK_LEVEL) +BLK_LEVEL);

                if(data_flt>1023)   data_int =1023;
                else if(data_flt<0) data_int =0;
                else                data_int = (ushort)data_flt;

                data = output_luma[line_num][pix_cnt];
                data = data>data_int? data : data_int;
                output_luma[line_num][pix_cnt] = data;

                // chroma just copy the luma
                output_chrm[line_num][pix_cnt] = noColor? 0 : (data - half);
            }
        }
    }
    fprintf(stdout,"end of drawing\n");
}


const float     ROTATE_LINENUM = 18.0f;

void rotated_lines(const float init_angle[2], short output_luma[H][W], short output_chrm[H][W]){

    int         field_cnt, line_cnt, pix_cnt, line_num, draw_num;
    short       data_int, data;
    float       angle, y, x;
    float       cos_a, sin_a, norm, proj_x, proj_y;
    float       diff_x, diff_y, data_flt;


    for(line_cnt=0; line_cnt<fsize; line_cnt++){
        for(pix_cnt=0; pix_cnt<hsize; pix_cnt++){
            output_luma[line_cnt][pix_cnt] = 0;
            output_chrm[line_cnt][pix_cnt] = 0;
        }
    }

    for(field_cnt=0; field_cnt<fieldNum; field_cnt++){

        for(angle=0, draw_num=0; draw_num<(int)ROTATE_LINENUM; angle+=PI/ROTATE_LINENUM, draw_num++){

            fprintf(stdout,"drawing line %d at field %d\n", draw_num, field_cnt);

            cos_a = cos(angle+init_angle[field_cnt]);
            sin_a = sin(angle+init_angle[field_cnt]);

            for( y=fsize/2.0f-field_cnt, line_cnt=field_cnt; line_cnt<fsize; y-=fieldNum, line_cnt+=fieldNum ){

                line_num = line_cnt;
                for( x=-hsize/2.0f, pix_cnt=0; pix_cnt<hsize; x++, pix_cnt++){

                    // projection
                    norm = x * cos_a + y * sin_a;

                    proj_x = norm * cos_a;
                    proj_y = norm * sin_a;

                    diff_x = proj_x - x;
                    diff_y = proj_y - y;

                    norm = sqrt(diff_x * diff_x + diff_y * diff_y);

                    data_flt = norm>THICKNESS? 0 : (cos(PI * norm/THICKNESS) +1)/2.0f *1024.0f;

                    if(data_flt>1023)   data_int =1023;
                    else if(data_flt<0) data_int =0;
                    else                data_int = (ushort)data_flt;

                    data = output_luma[line_num][pix_cnt];
                    data = data>data_int? data : data_int;

                    output_luma[line_num][pix_cnt] = data;
                    output_chrm[line_num][pix_cnt] = noColor? 0 : (data - half);
                }
            }
        }
    }
    fprintf(stdout,"end of drawing\n");
}

void color_bars(short output_luma[H][W], short output_chrm[H][W]){


    short       luma, chrm[2], color_index;
    short       color[8][3]= { {250,250, 250}, {250,250, 16}, { 16, 250, 250}, { 16, 250, 16},
                               {250, 16, 250}, {250, 16, 16}, { 16,  16, 250}, { 16, 16,  16}};
    int         line_cnt, pix_cnt, line_num, i, j;
    vidInOut    vidOp;

    for(i=0; i<8; i++){
        for(j=0; j<3; j++){
            color[i][j] *=4;
        }
    }

    for(line_cnt=0; line_cnt<fsize; line_cnt++){

        color_index = 0;
        line_num = line_cnt;
        for(pix_cnt=0; pix_cnt<hsize; pix_cnt++){

            color_index = pix_cnt*8/hsize;
            vidOp.rgb2yuv( color[color_index], &luma, chrm );
            
            // horizontal data is sinual wave
            output_luma[line_num][pix_cnt] = luma;
            output_chrm[line_num][pix_cnt] = noColor? 0 : chrm[pix_cnt & 1];
        }
    }
}

void plate_zone(short output_luma[H][W], short output_chrm[H][W]){

    const float     center_size = 50.0f; // 50 pixels
    const float     zone_range = float(hsize)/1.5f;

    int     line_cnt, pix_cnt, line_num;
    short   luma, chrm[2];
    float   x, y, data, period, dist, x_sq, y_sq, radius;
    FILE    *fp;

    fp = fopen("data", "wb");

    for(y=fsize/2, line_cnt=0; line_cnt<fsize; line_cnt++, y--){

        y_sq = y*y;

        line_num = line_cnt;
        for(x=-hsize/2, pix_cnt=0; pix_cnt<hsize; pix_cnt++, x++){

            x_sq = x*x;

            dist = sqrt(x_sq + y_sq);

            radius = dist/zone_range;
            if(radius>1){
                period = 1.0f;
            }else{
                period = center_size*(cos(PI*radius)+1)/2;
                period += 4.0f;
            }

//            period = (center_size - (center_size-1)/zone_range *dist);
//            if(period<4.0f)
//                period = 4.0f;

            data = cos(PI*dist/period);
            chrm[0] = int(data * 511);
            chrm[1] = int(-data * 511);
            
            data = (data + 1.0f)/2.0f;
            luma = (ushort)(data * 1023.0f);

            output_luma[line_num][pix_cnt] = luma;
            output_chrm[line_num][pix_cnt] = noColor? 0 : chrm[pix_cnt & 1];
            
            if(!line_cnt)
                fprintf(fp, "%f\n",data);
        }
    }
    fclose(fp);
}

void sweep_image_hor(short output_luma[H][W], short output_chrm[H][W]){

    float   incr_cycle = (maxHcycle-minHcycle)/float(hsize);
    float   period, data;
    int     line_cnt, pix_cnt, line_num;

    for(line_cnt=0; line_cnt<fsize; line_cnt++){
        
        line_num = line_cnt;
        for(pix_cnt=0, period = maxHcycle; pix_cnt<hsize; pix_cnt++, period-=incr_cycle){

            data = sin(PI * pix_cnt/period);
            if(sin2bin)
                data = fabs(data)<=softCut? data/softCut : data<0? -1 : 1;
            data += 1.0f;
            data /= 2.5f;    // reduces dynamic range
            data *= 1023.0f;   // in order to give
            data += 102.0f;    // offsets for black and white 
            
            output_luma[line_num][pix_cnt] = short(data);
            output_chrm[line_num][pix_cnt] = noColor? 0 : short(data - half);
        }
    }
}

void sweep_image_ver(short output_luma[H][W], short output_chrm[H][W]){

    float   incr_cycle = (maxVcycle-minVcycle)/float(fsize);
    float   period, data;
    int     line_cnt, pix_cnt, line_num;

    for(line_cnt=0, period=maxVcycle; line_cnt<fsize; line_cnt++, period-=incr_cycle){

        data = sin(PI * line_cnt/period);
        if(sin2bin)
            data = fabs(data)<=softCut? data/softCut : data<0? -1 : 1;
        data += 1.0f;
        data /= 2.5f;    // reduces dynamic range
        data *= 1023.0f;   // in order to give
        data += 102.0f;    // offsets for black and white 

        line_num = line_cnt;
        for(pix_cnt=0; pix_cnt<hsize; pix_cnt++){           
            output_luma[line_num][pix_cnt] = short(data);
            output_chrm[line_num][pix_cnt] = noColor? 0 : short(data - half);
        }
    }
}

void sweep_color_444(short lm_output[H][W], short cb_output[H][W], short cr_output[H][W]){

    // parameters def
    const float     r_init_period = 1.50f, r_max_period = 5.0f;
    const float     g_init_period = 11.5f, g_max_period = 10.0f;
    const float     b_init_period = 1.50f, b_max_period = 10.0f;

    //
    short       color[3], luma, chrm[2];
    float       data_r, data_g, data_b;
    float       r_period, g_period, b_period;
    int         line_num;
    vidInOut    vidOp;

    for(int line_cnt=0; line_cnt<fsize; line_cnt++){

        line_num = line_cnt;
        for(int pix_cnt=0; pix_cnt<hsize; pix_cnt++){
            
            r_period = r_init_period + r_max_period/hsize * pix_cnt;
            g_period = g_init_period - g_max_period/hsize * pix_cnt;
            b_period = b_init_period + b_max_period/hsize * abs(pix_cnt - hsize/2);


            data_r = sin(PI*pix_cnt/r_period);
            color[0] = (ushort)((data_r+1) *511.0f);

            data_g = sin(PI*pix_cnt/g_period);
            color[1] = (ushort)((data_g+1) *511.0f);

            data_b = sin(PI*pix_cnt/b_period);
            color[2] = (ushort)((data_b+1) *511.0f);

            vidOp.rgb2yuv( color, &luma, chrm );

            lm_output[line_num][pix_cnt] = luma;
            cb_output[line_num][pix_cnt] = noColor? 0 : chrm[0];
            cr_output[line_num][pix_cnt] = noColor? 0 : chrm[1];
        }
    }
}

void pairs_pattern1( short y_o[H][W], short c_o[H][W],
                                 short r_o[H][W], short g_o[H][W], short b_o[H][W] ) const
{
    int     pix_cnt, line_cnt;
    short   r, g, b;

    if( y_o && c_o ){

        return;
    }

    if( r_o && g_o && b_o ){
        for( line_cnt=0; line_cnt<fsize; line_cnt++ ){
            for( pix_cnt=0; pix_cnt<hsize; pix_cnt++ ){

                switch( pix_cnt&3 ){
                case 0: r = 512, g = 0, b = 0; break;
                case 1: r = 1023, g = 0, b = 0; break;
                case 2: r = 0, g = 0, b = 512; break;
                case 3: r = 0, g = 0, b = 1023; break;
                }

                if( (pix_cnt>>1) <= line_cnt ){
                    r_o[line_cnt][pix_cnt] = r;
                    g_o[line_cnt][pix_cnt] = g;
                    b_o[line_cnt][pix_cnt] = b;
                }else{
                    r_o[line_cnt][pix_cnt] = 1023;
                    g_o[line_cnt][pix_cnt] = 1023;
                    b_o[line_cnt][pix_cnt] = 1023;
                }
            }
        }
        return;
    }

}

void pairs_pattern2( short y_o[H][W], short c_o[H][W],
                                 short r_o[H][W], short g_o[H][W], short b_o[H][W] ) const
{
    int     pix_cnt, line_cnt;
    short   r, g, b;

    if( y_o && c_o ){

        return;
    }

    if( r_o && g_o && b_o ){
        for( line_cnt=0; line_cnt<fsize; line_cnt++ ){
            for( pix_cnt=0; pix_cnt<hsize; pix_cnt++ ){

                if( line_cnt&1 ){
                    switch( pix_cnt&3 ){
                    case 0: r = 0, g = 512, b = 0; break;
                    case 1: r = 0, g = 1023, b = 0; break;
                    case 2: r = 512, g = 0, b = 512; break;
                    case 3: r = 1023, g = 0, b = 1023; break;
                    }

                    if( (pix_cnt>>1) <= line_cnt ){
                        r_o[line_cnt][pix_cnt] = r;
                        g_o[line_cnt][pix_cnt] = g;
                        b_o[line_cnt][pix_cnt] = b;
                    }else{
                        r_o[line_cnt][pix_cnt] = 1023;
                        g_o[line_cnt][pix_cnt] = 1023;
                        b_o[line_cnt][pix_cnt] = 1023;
                    }
                }else{
                    switch( pix_cnt&3 ){
                    case 0: r = 512, g = 0, b = 0; break;
                    case 1: r = 1023, g = 0, b = 0; break;
                    case 2: r = 0, g = 0, b = 512; break;
                    case 3: r = 0, g = 0, b = 1023; break;
                    }

                    if( (pix_cnt>>1) <= line_cnt ){
                        r_o[line_cnt][pix_cnt] = r;
                        g_o[line_cnt][pix_cnt] = g;
                        b_o[line_cnt][pix_cnt] = b;
                    }else{
                        r_o[line_cnt][pix_cnt] = 1023;
                        g_o[line_cnt][pix_cnt] = 1023;
                        b_o[line_cnt][pix_cnt] = 1023;
                    }
                }
            }
        }
        return;
    }

}

*/



}

#endif
