#include <cstdint>
#include <algorithm>
#include "def_movies.h"
#include "all_common.h"
#include "tools_video.h"


static s_movie_vars m_MoviePadPtr[NUM_ULTRIX_WARP_CHANNELS] = { 0 };

// Change sizes for UHD!?
static short m_Movie_Rand[NUM_ULTRIX_WARP_CHANNELS][2 * MOVIE_WARPPIX_ITER_MAX] = { 0 }; // magnify blend value
static short m_Movie_Offset[NUM_ULTRIX_WARP_CHANNELS][2 * MOVIE_WARPPIX_ITER_MAX] = { 0 }; //offset ""  ""    ""     ""
static uint16_t m_Movie_Width[NUM_ULTRIX_WARP_CHANNELS][2 * MOVIE_WARPPIX_ITER_MAX] = { 0 }; //width for each scratch pos

// fake array
static int16_t  d_Movie_Rand_symbol[NUM_ULTRIX_WARP_CHANNELS][2 * MOVIE_WARPPIX_ITER_MAX]; // device symbol
static int16_t  d_Movie_Offset_symbol[NUM_ULTRIX_WARP_CHANNELS][2 * MOVIE_WARPPIX_ITER_MAX]; // device symbol
static uint16_t d_Movie_Width_symbol[NUM_ULTRIX_WARP_CHANNELS][2 * MOVIE_WARPPIX_ITER_MAX]; // device symbol


void init_symbol(const int init, const int length, const float angle = 60.f * F_PI / 180.f)
{
    for (int x = 0; x < 2 * MOVIE_WARPPIX_ITER_MAX; x++) {
        d_Movie_Offset_symbol[0][x] = -1 & 0x7fff;
        d_Movie_Width_symbol[0][x] = 0;
    }
    if (0) {
        const float slope = tanf(angle);
        for (int x = init, k = 0; k < length; x++, k++) {         // X: I.1f
            const int y = int(slope * x * 8);   // Y: I.4f
            d_Movie_Offset_symbol[0][k] = y;    //
            d_Movie_Width_symbol[0][k] = 5 << 4;  // W: I.4f
        }
    }
    if (1) {
        const int shift = 21;
        const int one = 1 << shift;
        int inv_len = one / length / length;

        if (1) {
            for (int x = 0; x < length * 2; x++) {         // X: I.1f
                int pos = x - length;
                int arc = pos * pos * inv_len;                    //
                d_Movie_Width_symbol[0][x] = 5 * (one - arc) >> shift - 4;  // W: I.4f
                d_Movie_Offset_symbol[0][x] = 10 * arc >> shift - 4;    //
            }
        }
        else if (0) {
            for (int x = 0; x < length * 2; x++) {         // X: I.1f
                int pos = x - length;
                int arc = pos * pos * inv_len;                    //
                d_Movie_Width_symbol[0][x] = 15 * abs(one/2 - arc) >> shift - 4;  // W: I.4f
                d_Movie_Offset_symbol[0][x] = 10 * arc >> shift - 4;    //
            }
        }
        else if (0) {
            for (int k = 0; k < 2 * MOVIE_WARPPIX_ITER_MAX; k) {
                for (int x = 0; x < length * 2; x++, k++) {         // X: I.1f
                    int pos = x - length;
                    int arc = pos * pos * inv_len;                    //
                    d_Movie_Width_symbol[0][k] = 5 * abs(one - arc) >> shift - 4;  // W: I.4f
                    d_Movie_Offset_symbol[0][k] = 10 * arc >> shift - 4;    //
                }
                k += 40;
            }
        }
    }
}

void movie_kernel(
    texObject_t dest,
    texObject_t data_tex,
    s_frame_info input_format,
    s_movie_vars m_MoviePadPtr,
    int warpChan)
{
    const int rows_factor = (input_format.vid_mode == 1/*e_VidModeScan_Interlaced*/) ? 2 : 1;
    // pointer to the Movie structure in the scratch pad
    s_movie_vars* scratch = &m_MoviePadPtr;

    const uint32_t cols = input_format.cols;
    const uint32_t rows = input_format.rows * rows_factor;

    float WH_ratio = (float)cols / (float)rows;
    float HW_ratio = (float)rows / (float)cols;

    // pointers to random tables
    const short* const randval1 = &d_Movie_Rand_symbol[warpChan][0];
    const short* const randval2 = &d_Movie_Rand_symbol[warpChan][MOVIE_MAX_RANDOM];

    const short radius = (short)scratch->movie_radius;

    const uint32_t  pixel_noise = scratch->noise >= 0.f;
    const uint16_t  noise_intensity = pixel_noise ? scratch->noise : -scratch->noise;

    // 1 / max scratch width
    const int inv_scratch_width = (scratch->scratch_max_width == 0) ? 0 : (1 << 16) / scratch->scratch_max_width;

    // horizontal position of the vertical lines scratches
    const int   vert_scratch_pos1 = scratch->vert_pos1;
    const int   vert_scratch_pos2 = scratch->vert_pos2;
    uint32_t    vert_scratch_mask;

    // sometimes vertical scratches are momentarily hidden
    const uint32_t  show_scratch1 = scratch->show_scratch1;
    const uint32_t  show_scratch2 = scratch->show_scratch2;

    // Screen jumps
    // Ideally I would like to use RAND, but the screen jump wouldn't work
    // This might have to do with how warpInterface.c uses the strobe effect.
    //  Mike A, July 16,2008
    const short jump_y =
        (abs((int32_t)scratch->film_jump) * (randval1[0] & 0xFFF) >> MOVIE_JUMP_SHIFT) -
        (abs((int32_t)scratch->film_jump) * 0x1000 >> MOVIE_JUMP_SHIFT + 1) * scratch->ratio480iToCurrentY;
    const short jump_x =
        (abs((int32_t)scratch->film_jump) * (randval2[0] & 0x7FF) >> MOVIE_JUMP_SHIFT) -
        (abs((int32_t)scratch->film_jump) * 0x0800 >> MOVIE_JUMP_SHIFT + 1) * scratch->ratio480iToCurrentX;

    const int32_t softQ = (int32_t)
        (((MOVIE_SOFTNESS_MAX - (int32_t)scratch->softness) / (float)MOVIE_SOFTNESS_MAX) * (1 << X_INT_BITS));

    const int16_t movie_model = scratch->movieModel;

    for (int y_i = 0; y_i < input_format.rows; y_i++) {
        for (int x_i = 0; x_i < input_format.cols; x_i++) {

            int y_out = (input_format.vid_mode == 1/*e_VidModeScan_Interlaced*/) ?
                (input_format.isOddField) ? (2 * y_i + 1) : (2 * y_i) : (y_i);


            // range check for non-UHD vidmodes: 
            //  This step is to avoid cuda-memcheck errors in any vidmodes with non-round  
            //  block-number calculations.
            if (x_i >= input_format.cols || y_out >= (input_format.rows * rows_factor))
                return;

            int idx = y_i * input_format.cols + x_i;

            float xout, yout, aout = WARP_ALPHA_IDENTITY, lout = WARP_LIGHT_IDENTITY;


            int32_t   x = x_i, x_dst = x_i;    // loop counter through a line
            uint32_t  u_light;   // light due to noise
            uint32_t  circle_light;

            uint16_t  y_dst_Q0 = y_out;

            int16_t   scratch_light;

            int16_t   light;      // final light value
            int16_t   alpha;      // final alpha value

            // Colorize variables
            float cr_scale;
            float cb_scale;
            float rb_angle;
            float sat;
            float cr_coeff;
            float cb_coeff;

            // the contour of the screen is darkened, this measures the distance
            // to the center
            const int corner_y_2 = ((y_dst_Q0 - (rows >> 1)) * (y_dst_Q0 - (rows >> 1)) * WH_ratio) * WH_ratio;
            // (x-cx)^2+(y-cy)^2
            int     corner_x2_y2 = (((x_dst - (cols >> 1)) * (x_dst - (cols >> 1)))) + corner_y_2;
            // increment to corner_x2_y2 (2x+1-2c)

            uint32_t  x_noise_small = 0;   // index to random table for noise (either x>>1 or x)
            int16_t   random_line;
            int16_t   random_next_line;
            // Distance to the scratch starting point (in y and x)
            short   scratch_y = (y_dst_Q0 << 4) - (scratch->scratch_pos >> MAX_BITS_COORD);
            const int  sinA_scratch_y = -scratch->scratch_sinA * scratch_y;
            const int  cosA_scratch_y = scratch->scratch_cosA * scratch_y;
            int     scratch_x = -(uint16_t)scratch->scratch_pos;
            int     scratch_diff;

            // Pre-calculations for rotation formula

            // identity + random jump value
            int   delta_x_jump = jump_x * scratch->ratio480iToCurrentY;
            const int   shifted_line_jump = (y_dst_Q0) + (jump_y * scratch->ratio480iToCurrentY);

            //** VHS **
            // top edge is curved and noise is added (average of 4 lines)
            int     edge_curve;   // to emulate something like bad VHS tracking
            int     track_x;    //** VHS **/

            // position relative to the scratch after rotating back
            int   rot_x;
            int   rot_y;

            uint16_t rand1, rand2, rand3, rand4;

            int hasLight;

            scratch->rand_Q32 = GET_RAND;
            random_line = scratch->rand_Q32 >> 16;

            scratch->rand_Q32 = GET_RAND;
            random_next_line = scratch->rand_Q32 >> 16;

            scratch->rand_Q32 = GET_RAND;
            rand1 = scratch->rand_Q32 >> 16;

            scratch->rand_Q32 = GET_RAND;
            rand2 = scratch->rand_Q32 >> 16;

            scratch->rand_Q32 = GET_RAND;
            rand3 = scratch->rand_Q32 >> 16;

            scratch->rand_Q32 = GET_RAND;
            rand4 = scratch->rand_Q32 >> 16;

            const int rand_sum =
                ((rand1 >> 5) - 0x100) +
                ((rand2 >> 5) - 0x100) +
                ((rand3 >> 5) - 0x100) +
                ((rand4 >> 5) - 0x100);
            edge_curve = y_dst_Q0 < scratch->line_error_max ?
                ((y_dst_Q0 >> 1) - (scratch->line_error_max >> 1)) *
                ((y_dst_Q0 >> 1) - (scratch->line_error_max >> 1)) *
                (rand_sum) >> 10 + 2 + 2
                : 0;

            scratch->rand_Q32 = GET_RAND;
            rand1 = scratch->rand_Q32 >> 16;

            scratch->rand_Q32 = GET_RAND;
            rand2 = scratch->rand_Q32 >> 16;

            //VHS
            track_x = -edge_curve + ((int32_t)scratch->rand_scratch * // MOVIE_RandScratch(warp)
                (rand1 >> 20) + (int32_t)scratch->rand_scratch * // MOVIE_RandScratch(warp)
                (rand2 >> 20) >> 10 + 2 + 1) + jump_x;

            // Proceed with the different effects except if film type is none.
            if (movie_model == 0)//e_Movie_none)
            {
                // Do nothing.
                light = WARP_LIGHT_IDENTITY;
                alpha = WARP_ALPHA_IDENTITY;
            }
            else
            {
                if (SD_X_SIZE != cols)
                {
                    //radius = (short)(radius * SD_X_SIZE / cols) / 2;
                }

                if (movie_model == 1)//e_Movie_VHS)
                {
                    // No hair scratches, no dark borders (1920's)
                    // Jump coeff is used to modified tracking
                     // Update:
                       // moving to the next pixel
                    track_x += x;// * (1<<SUBPIX_BITS);

                    // Either "2x2 patch" or "pixel" noise
                    //     x_noise_small = x/2  or   x_noise_small = x
                    //     += 0,1,0,1...  or       += 1
                    x_noise_small = ((x & 1) & pixel_noise);

                    // update x
                    x = track_x;

                    scratch->rand_Q32 = GET_RAND;
                    rand1 = (!x_noise_small * scratch->rand_Q32 >> 16) + ((x_noise_small)*rand1);

                    scratch->rand_Q32 = GET_RAND;
                    rand2 = (!x_noise_small * scratch->rand_Q32 >> 16) + ((x_noise_small)*rand2);

                    // random light pattern (2x2 "patch" or pixel-by-pixel)
                    // (x>>1) or (x)
                    u_light = (((uint32_t)rand1 * random_line >> 10) & WARP_LIGHT_MAX)
                        + (((uint32_t)rand2 * random_line >> 10) & WARP_LIGHT_MAX)
                        + (((uint32_t)rand1 * random_next_line >> 10) & WARP_LIGHT_MAX)
                        + (((uint32_t)rand2 * random_next_line >> 10) & WARP_LIGHT_MAX)
                        >> 5;

                    // modulated by the chosen amplitude
                    u_light = u_light * noise_intensity >> 9;

                    light = u_light + WARP_LIGHT_IDENTITY;

                    // then add the global "flashing" (light level variation
                    // from field to field)
                    light += scratch->light_flash;

                    alpha = WARP_ALPHA_IDENTITY;

                    // Identity (+ random jumps)
                    x_dst = track_x;
                }
                else
                {
                    // moving to the next pixel
                    delta_x_jump += (x);
                    // position in x relative to the scratch starting point (Q4)
                    scratch_x += x << 4;

                    // Either "2x2 patch" or "pixel" noise
                    //     if pixel noise is false, we want to have large noise.
                    x_noise_small = !((x & 1) & pixel_noise);
                    scratch->rand_Q32 = GET_RAND;
                    rand1 = (x_noise_small * (scratch->rand_Q32 >> 16)) + ((!x_noise_small) * rand1);

                    scratch->rand_Q32 = GET_RAND;
                    rand2 = (x_noise_small * (scratch->rand_Q32 >> 16)) + ((!x_noise_small) * rand2);

                    // random light pattern (2x2 "patch" or pixel-by-pixel)
                    // u_light controls amount of noise and size of noise on screen

                    u_light =
                        (((uint32_t)rand1 * random_line >> 10) & WARP_LIGHT_MAX) +
                        (((uint32_t)rand2 * random_line >> 10) & WARP_LIGHT_MAX) +
                        (((uint32_t)rand1 * random_next_line >> 10) & WARP_LIGHT_MAX) +
                        (((uint32_t)rand2 * random_next_line >> 10) & WARP_LIGHT_MAX)
                        >> 5;
                    // modulated by the chosen amplitude, the smaller the shift the larger the noise
                    u_light = u_light * noise_intensity >> 9;

                    // darkening of the edge

                    // >>9 because any position on the screen is less than 1<<9 pixels away
                    // from the center, which means, if squared and shifted right by 9,
                    // in the range of [0, WARP_LIGHT_IDENTITY]
                    // corner_x2_y2 = (x-Cx)�+(y-Cy)�
                    // Not sure why we add 9. In SD we added 8.  I increased it to 9 so
                    // that the radius was smaller, therefore increasing the viewable area
                    // in MD (Mike A)/
                    // CA - Change back to 8 to get same radius look in Acuity, I think due
                    // to there being one more bit in light values now.
                    circle_light = corner_x2_y2 * radius >> ((X_INT_BITS - 1) + 8);
                    // So we remove that value from our previous light value
                    light = u_light + WARP_LIGHT_IDENTITY - circle_light;

                    // limit light to 0 and remove softness from inside circle as required.
                    hasLight = (light > 0);
                    light = hasLight * (light + ((softQ * circle_light) >> X_INT_BITS));

                    // rotate back
                    rot_x = (scratch->scratch_cosA * scratch_x) + sinA_scratch_y;
                    rot_y = (scratch->scratch_sinA * scratch_x) + cosA_scratch_y;

                    // proper fraction bits for future use
                    rot_x >>= MOVIE_TRIG_FRAB + 4 - 1;    // keep 1 FRAB
                    rot_y >>= MOVIE_TRIG_FRAB + 4 - 4;    // keep 4 FRABs

                    // table bounds
                    rot_x += scratch->scratch_length;
                    if (rot_x > (2 * cols - 1))
                        rot_x = (2 * cols - 1);
                    if (rot_x < 0)
                        rot_x = 0;

                    // if y position from the scratch (w/ offset) is smaller than
                    // scratch width, then apply the scratch light instead of
                    // any other light
                    short size = d_Movie_Width_symbol[warpChan][rot_x];
                    short diff = rot_y - d_Movie_Offset_symbol[warpChan][rot_x];
                    scratch_diff = size - abs(diff);

                    // 0 to 1 represented as 0 to 1<<10
                    // (current distance from the scratch edge / max scratch width)
                    scratch_diff = scratch_diff * inv_scratch_width >> 16 + 4 - SHIFT_TO_LIGHTING;

                    // If pixel is located on a scratch, apply the light to "draw" it.
                    // On the scratch edges (scratch_diff=0), light += 0
                    // Linearly increasing as we approch the vertical center of the scratch,
                    // potentially to light=scratch_max_light if the scratch width at this
                    // position (defined in PreField) is maximum

                    scratch_light =
                        (scratch_diff * (scratch->scratch_max_light - light)) >> SHIFT_TO_LIGHTING;
                    scratch_light = scratch_light * (scratch_diff >= 0);   // Y. Liu Note

                    light += scratch_light;

                    vert_scratch_mask = ((x == vert_scratch_pos1) & show_scratch1)
                        | ((x == vert_scratch_pos2) & show_scratch2);
                    if (vert_scratch_mask)
                        light = scratch->vert_scratch_light;

                    // then add the global "flashing" (light level variation
                    // from field to field)
                    light += scratch->light_flash;

                    alpha = WARP_ALPHA_IDENTITY;

                    // Identity (+ random jumps)
                    x_dst = delta_x_jump;
                }
            }

            // Chroma modification only with color movie
            if (scratch->colorType >= 1) //e_Movie_color)
            {
                // "hue" 2Pi = 4k, Pi/4 = k/2
                rb_angle = (float)(((float)scratch->hue / 4096.f) + 1.1f);  // offset of Pi/4

                // saturation (amplitude)
                sat = (float)scratch->sat / 1024.f;

                // if the color type is 2-color, then no offset to "hue" to allow
                // negative angle, and all satuation values are negative
                if (scratch->colorType == 2) //e_Movie_color_2)
                {
                    rb_angle = ((float)scratch->hue / 4096.f) + 0.85f;
                    sat = -(float)scratch->sat / 1024.f;
                }
                // 2Pi = 4k, Pi = 2k
                cr_scale = sat * sin((float)rb_angle * CUDART_PI_F * 2.f);
                cb_scale = sat * cos((float)rb_angle * CUDART_PI_F * 2.f);

                if (scratch->colorType == 3) //e_Movie_color_fade)
                {
                    cr_coeff = scratch->red;// WARP_FLOAT(scratch->red, e_float3_dec);
                    cb_coeff = scratch->blue;// WARP_FLOAT(scratch->blue, e_float3_dec);
                }
                else
                {
                    // the curves that increase red or blue "sensitivity" are generated
                    // by neg. coeffs
                    cr_coeff = -scratch->red;// -WARP_FLOAT(scratch->red, e_float3_dec);
                    cb_coeff = -scratch->blue;// -WARP_FLOAT(scratch->blue, e_float3_dec);
                }
            }

            y_dst_Q0 = (input_format.vid_mode == 1/*e_VidModeScan_Interlaced*/) ?
                (input_format.isOddField) ? (y_dst_Q0 - 1) / rows_factor
                : y_dst_Q0 / rows_factor : y_dst_Q0 / rows_factor;

            // No protection against out-of-range addresses, since they can't occur.
            // print calculated data at quarter of the image

            xout = x_dst;
            yout = y_dst_Q0;
            aout = (float)alpha / (float)WARP_ALPHA_IDENTITY;
            lout = (float)light / (float)WARP_LIGHT_IDENTITY;

            int4 d_int;
            float4 d_flt;
            float u = xout;
            float v = yout;
            //d_int = tex2D<int4>(data_tex, u, v);
            if (xout < 0)       xout = 0;
            if (xout >= cols)   xout = cols - 1;
            if (yout < 0)       yout = 0;
            if (yout >= rows)   yout = rows - 1;
            const int addr = yout * cols + xout;
            d_int.x = data_tex.y[addr];
            d_int.y = data_tex.v[addr];
            d_int.z = data_tex.u[addr];

            // Color calc:
            float lu = (float)d_int.x / Y_MAX;// (((float)d_int.x - Y_MIN) / (Y_MAX - Y_MIN));
            float cr = (float)d_int.y / CHRM_MID;// (((float)d_int.y - CHRM_MID) / (CHRM_MID));// [-1.f,1.f]
            float cb = (float)d_int.z / CHRM_MID;//(((float)d_int.z - CHRM_MID) / (CHRM_MID));// [-1.f,1.f]

            float delta_chroma_cr, delta_chroma_cb;
            float cr_value, cb_value;

            //if (scratch->colorType == e_Movie_color || scratch->colorType == e_Movie_color_2)
            if (scratch->colorType == 1 || scratch->colorType == 2)
            {
                /* This case has "contrast" control and linear rescaling */
                delta_chroma_cr = cr;
                delta_chroma_cb = cb;

                // warp (gs_warp_preprocess_data) color adjustment
                //   c' = c + K*c*(c-1)
                //       K is cb/crCoeff
                //       c is the centered chroma value [-1,1]
                cr_value = delta_chroma_cr;
                cb_value = delta_chroma_cb;

                cr_value += ((delta_chroma_cr * (delta_chroma_cr - 1.f))) * cr_coeff;
                cb_value += ((delta_chroma_cb * (delta_chroma_cb - 1.f))) * cb_coeff;

                // c'' = K * c'     where K is cb/crScale
                cr_value = cr_value * cr_scale;
                cb_value = cb_value * cb_scale;
            }
            else if (scratch->colorType == 3) //e_Movie_color_fade)
            {
                /* This case has "offset" control and linear rescaling */
                delta_chroma_cr = cr;
                delta_chroma_cb = cb;

                // Rescaling
                cr_value = (cr_scale * delta_chroma_cr);
                cb_value = (cb_scale * delta_chroma_cb);
                // Offset + back to proper LUT range
                cr_value += (cr_coeff);
                cb_value += (cb_coeff);
            }
            else if (scratch->colorType == 4) //e_Movie_sepia) // need a tint
            {
                // only apply some hue shift and saturation
                cr_value = 0.1f;
                cb_value = -0.3f;

            }
            else if (scratch->colorType == 5)   // e_Movie_bw) // middle value
            {
                cr_value = 0.0f;
                cb_value = 0.0f;
            }
            else //same value
            {
                // no effect
                cr_value = cr;
                cb_value = cb;
            }

            // Luma Adjustment:
            float delta_luma;
            float luma_value = lu;
            float inv_scale = 1;// lumCoeff used to rescale (w/ centering)
            float lumScale = scratch->lum / 4096.f;
            float lumCoeff = (scratch->luma_contrast + 4096.f) / 8192.f;

            if (lumCoeff > 0)
            {
                inv_scale = 1 / (lumCoeff);
                lumCoeff = 0;
            }

            if (scratch->colorType != 10/*e_Movie_color_max*/)
            {
                // luma value relative to the center
                delta_luma = lu;// [0.f,1.f]

                // warp (gs_warp_preprocess_data) color adjustment
                //   c' = c + K*c*(c-1)
                //       K is lumCoeff
                //       c is the luma value [0,1]
                luma_value = delta_luma;

                luma_value += ((delta_luma * (delta_luma - 1))) * lumCoeff;

                // rescaling (centered around 512)
                // (because of the way inv_scale is found, this rescaling always
                //  reduces the distance to the center)
                luma_value = (luma_value * inv_scale);

                // c'' = K * c'     where K is lumScale
                // (rescaling, not centered)
                luma_value = luma_value * (lumScale);
            }

            if (movie_model != 0/*e_Movie_none*/ || scratch->colorType != 10/*e_Movie_color_max*/)
            {
                d_int.x = (luma_value * (Y_MAX - Y_MIN)) + Y_MIN;
                d_int.y = (cr_value * CHRM_MID) + CHRM_MID;
                d_int.z = (cb_value * CHRM_MID) + CHRM_MID;
            }

            if (d_int.x < Y_MIN) d_int.x = Y_MIN;
            if (d_int.y < CR_MIN) d_int.y = CR_MIN;
            if (d_int.z < CB_MIN) d_int.z = CB_MIN;
            if (d_int.x >= Y_MAX) d_int.x = (Y_MAX);
            if (d_int.y >= CR_MAX) d_int.y = (CR_MAX);
            if (d_int.z >= CB_MAX) d_int.z = (CB_MAX);
            d_int.w = 255;

            // apply color
            // limit the value ranges
            d_flt.x = std::min((float)Y_MAX, std::max((float)Y_MIN, ((float)(d_int.x - Y_MIN) * aout * lout) + Y_MIN));
            d_flt.y = std::min((float)CR_MAX, std::max((float)CR_MIN, ((float)(d_int.y - CHRM_MID) * aout * lout) + CHRM_MID));
            d_flt.z = std::min((float)CB_MAX, std::max((float)CR_MIN, ((float)(d_int.z - CHRM_MID) * aout * lout) + CHRM_MID));
            d_flt.w = std::min((float)ALPH_MAX, std::max(0.f, (float)(d_int.w) * aout));

            // Output is int4 format
            int4 data;
            data.x = (int)d_flt.x;
            data.y = (int)d_flt.y - CHRM_MID;
            data.z = (int)d_flt.z - CHRM_MID;
            data.w = (int)d_flt.w;

            dest.y[idx] = data.x;
            dest.v[idx] = data.y;
            dest.u[idx] = data.z;
        }
    }
}


void test_movie_kernel() {
    const int HSIZE = 720, FSIZE = 486;

    int asize = HSIZE * FSIZE;
    char* din_nm = "C:/imageNvideo2/NTSC/Anchors.ppm";
    short* din_y = new short[asize];
    short* din_u = new short[asize];
    short* din_v = new short[asize];
    vio::load_rgb2yuv(din_nm, HSIZE, FSIZE, din_y, din_u, din_v);

    texObject_t img_tex;
    img_tex.y = din_y;
    img_tex.u = din_u;
    img_tex.v = din_v;

    short* dout_y = new short[asize];
    short* dout_u = new short[asize];
    short* dout_v = new short[asize];
    texObject_t img_out;
    img_out.y = dout_y;
    img_out.u = dout_u;
    img_out.v = dout_v;

    s_movie_vars vars;

    vars.movieModel = 2;    // 0:/none, 1/VHS, others/

    const float radius = 100.f;
    vars.movie_radius = (1<<(X_INT_BITS - 1 + 8)) / radius / radius;
    vars.noise = 10.f;
    vars.film_jump = 10.f;
    vars.presets = 0.f;
    vars.softness = 10.f;
    vars.rand_scratch = 40.f;

    vars.cols = HSIZE;
    vars.rows = FSIZE;

    // Color Tables:
    const float R = .01f, G = 0.005f, B = 0.002f;
    vars.colorType = 1; // 0~10: 
    vars.hue = 10;      // 0~4096
    vars.sat = 10;      // 0~1024
    vars.lum = 4096;    // 0~4096
    vars.red = 0.01;// +0.5f * R - 0.454153f * G - 0.045847f * B;
    vars.blue = 0.001;// -0.114572f * R - 0.385428f * G + 0.5f * B;          // Panel input
    vars.luma_contrast = 4096;  // 0~8192

    uint32_t rand_Q32 = 12345678;

    const int PosX = HSIZE / 2, PosY = 160;// FSIZE / 2;
    const int ScratchW = HSIZE;
    vars.scratch_max_width = 4;     // max width of the scratch
    vars.scratch_pos = (PosY <<20)+(PosX <<4); // position _pack2(y,x). 11.4f
    vars.scratch_length = 180;       // length of the scrath
    const float angle = 90.f;
    vars.scratch_cosA = (short)(std::cosf(angle * PI / 180) * (1 << MOVIE_TRIG_FRAB)); // cosine of the angle of the scratch (w/ x-axis)
    vars.scratch_sinA = (short)(std::sinf(angle * PI / 180) * (1 << MOVIE_TRIG_FRAB)); // sine of the angle of the scratch

    // Vertical Scratches
    vars.vert_pos1 = 160;     // current position of the 1st vertical scratch
    vars.vert_pos2 = 200;     // current position of the 2nd vertical scratch
    short old_vert_pos1; // "center value" of the 1st vertical scratch's position
    short old_vert_pos2; // "center value" of the 2nd vertical scratch's position
    vars.show_scratch1 = 0x1; // TRUE: show the 1st vertical scratch; FALSE: hide it
    vars.show_scratch2 = 0x1; // same for 2nd vertical scratch
    uint32_t line_time_left1;  // time to switch between hide/show status
    uint32_t line_time_left2;  // ...//typedef struct {

     // Randomizer Elements
    short re_random_in;   // number of fields before we "re-randomize"
                          // (re-calculate the random table)
    uint32_t random_next;   // next random seed  both

    // Last value of MOVIE_Presets(warp) (models list)
    short last_model_coeff;

    // Default coeffs for each film model
    // (the compile-time defaults stored in far ram are copied to the scratch
    // pad in the init function)
    vars.ratio480iToCurrentX = 4.f/3.f;
    vars.ratio480iToCurrentY = 3.f/4.f;
    vars.light_flash = 100;
    vars.line_error_max = 10;
    int count;
    vars.scratch_max_light = 600;
    vars.vert_scratch_light = 200;


    s_frame_info  info;
    info.vid_mode = 0;
    info.isOddField = false;
    info.cols = HSIZE;
    info.rows = FSIZE;

    //init_symbol(-HSIZE, HSIZE*2, 60.f*F_PI/180.f);
    init_symbol(0, vars.scratch_length);
    movie_kernel(img_out, img_tex, info, vars, 0);

    char* out_nm = "./test.ppm";
    vio::save_yuv2rgb(out_nm, HSIZE, FSIZE, img_out.y, img_out.u, img_out.v);

    delete[] din_y;
    delete[] din_u;
    delete[] din_v;
    delete[] dout_y;
    delete[] dout_u;
    delete[] dout_v;
}