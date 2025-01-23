/*******************************************************************************************************************
 * FILE NAME   :    tools_winfir.h
 *
 * PROJECTION  :    2.1D DVE
 *
 * DESCRIPTION :    This file defines FIR filter related parameters and calculations
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2017 DEC 06      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#ifndef __TOOLS_WINFIR_H__
#define __TOOLS_WINFIR_H__

#include "all_common.h"
#include <sys/stat.h>

namespace   winfir
{
    const int       COEF_FBIT = 14;             // shift by 16
    const int       COEF_UNIT = 1 << COEF_FBIT;   // 16b for fractions
    const int       COEF_MASK = COEF_UNIT - 1;   //
    const int       FREQ_MAX_POINT = 500;       // resolutions for freq response display
    const bool      ADJUST_ERROR_ON = false;    // false: off, true: on, to show the errors in integer coefficients
    const float     fPI = 3.1415927f;

    typedef enum class eFILT_TYPE : int {
        eWIN_BOX,
        eWIN_SINUAL,
        eWIN_SIN_SQ,
        eWIN_COSINE,
        eWIN_COS_SQ,
        eWIN_TRIANGLE,
        eWIN_NUTTAL,
        eWIN_BLACKMAN,
        eWIN_FLAT_TOP,
        eWIN_GAUSSIAN,
        eWIN_BLK_HARRIS,
        eWIN_BLK_NUTTAL
    } fir_type_e;

    //---------------- Various Windows ------------------
    inline float box_window(float const& sample, const float win_size)
    {
        return fabs(sample) > win_size ? 0.0f : 1.0f;
    }


    inline float cos_window(float const& sample, const float win_size)
    {
        return fabs(sample) > win_size ? 0.0f : (cos(fPI * sample / (win_size + 1)) + 1.0f) / 2.0f;
    }


    inline float tri_window(float const& sample, const float win_size)
    {
        float   win_value = 0.0f;

        if (fabs(sample) <= win_size) {
            win_value = sample > 0.0f ? (win_size - sample) : (win_size + sample);
            win_value /= win_size;
        }

        return win_value;
    }


    inline float sin_window(float const& sample, const float win_size)
    {
        float       win_value = 0.0f;
        float       sin_value = sin(fPI * sample / win_size);

        if (fabs(sample) <= win_size) {
            if (sample > 0)
                win_value = sample < win_size / 2 ? (1.0f - sin_value) : (sin_value - 1.0f);
            else
                win_value = sample > -win_size / 2 ? (1.0f + sin_value) : -(1.0f + sin_value);
        }

        win_value = (win_value + 1.0f) / 2.0f;
        return win_value;
    }


    // reference to window function at wikipedia
    inline float nuttall_win(float const& sample, const float win_size)
    {
        float  sum = 0.355768f;

        sum += 0.487396f * cos(fPI * sample / win_size);
        sum += 0.144232f * cos(2.0f * fPI * sample / win_size);
        sum += 0.012604f * cos(3.0f * fPI * sample / win_size);
        return sum;
    }


    inline float blackman_win(float const& sample, const float win_size)
    {
        float  sum = 0.42659f;
        sum += 0.49656f * cos(fPI * sample / win_size);
        sum += 0.076849f * cos(2 * fPI * sample / win_size);
        return sum;
    }


    inline float flat_top_win(float const& sample, const float win_size)
    {
        float  sum = 1.0f;

        sum += 1.930f * cos(fPI * sample / win_size);
        sum += 1.290f * cos(2 * fPI * sample / win_size);
        sum += 0.388f * cos(3 * fPI * sample / win_size);
        sum += 0.028f * cos(4 * fPI * sample / win_size);
        return sum;
    }


    inline float gaussian_win(float const& sample, const float win_size)
    {
        float  data;
        static float   sigma = 0.5f;    // sigma<=0.5;
        data = sample / win_size / sigma;
        data *= data;
        data /= 2.0f;
        data = exp(-data);
        return data;
    }

    inline float blackman_harris(float const& sample, const float win_size)
    {
        float  sum = 0.35875f;

        sum += 0.48829f * cos(fPI * sample / win_size);
        sum += 0.14128f * cos(2 * fPI * sample / win_size);
        sum += 0.01168f * cos(3 * fPI * sample / win_size);
        return sum;
    }

    inline float blackman_nuttal(float const& sample, const float win_size)
    {
        float  sum = 0.3635819f;

        sum += 0.4891775f * cos(fPI * sample / win_size);
        sum += 0.1365995f * cos(2 * fPI * sample / win_size);
        sum += 0.0106411f * cos(3 * fPI * sample / win_size);
        return sum;
    }


    // sample: (-MAX_TAPS+1) -> (+MAX_TAPS), cetering at 0
    // cutoff: 1 -> 0; 1= full passband, 0.5= half-band,
    inline float sinc_function(float const& sample, float const& cut_off)
    {
        float  x = fPI * sample * cut_off;
        float  y = x ? sin(x) / x : 1.0f;
        return y;
    }

    //========================================================
    //               Taps for PolyPhase FIR
    //
    // The number of supports range [-MAX_YTAPS : +MAX_YTAPS].
    // For examples in the following figure, range: -4 to +4.
    // Howerver, supports will shift to the right. Therefore,
    // -MAX_YTAPS is never sampled. The real sampling range:
    // [-MAX_YTAPS+1 : +MAX_YTAPS]. In the example, -3 to +4.
    //
    // phase with 0:
    //                 +
    //               /   \
    //             /   |   \
    //           /           \
    //         /       |       \
    //       /                   \
    //     /           |           \
    //   /                           \
    // +               |               +
    // |---+---+---+---+---+---+---+---|---
    //     -3  -2  -1  0   1   2   3   4
    //                 |<->|  phase shift range
    //
    // phase by half:
    //                   +
    //                 /   \
    //               /   |   \
    //             /           \
    //           /       |       \
    //         /                   \
    //       /           |           \
    //     /                           \
    //   +               |               +
    // +-|-+---+---+---+---+---+---+---+-|-+
    //     -3  -2  -1  0   1   2   3   4
    //                 |<->|  phase shift range
    //
    // phase to unit one:
    //                     +
    //                   /   \
    //                 /   |   \
    //               /           \
    //             /       |       \
    //           /                   \
    //         /           |           \
    //       /                           \
    //     +               |               +
    // +---|---+---+---+---+---+---+---+---|-
    //     -3  -2  -1  0   1   2   3   4
    //                 |<->|  phase shift range
    //

    //=====================================================
    // unifies coefficients for single phase coefficients
    //=====================================================
    inline bool unify_coef(const int tap_num, int* coef)
    {
        int     sum, diff, sign, tap_index;

        sum = 0;
        for (tap_index = 0; tap_index < tap_num; tap_index++)
            sum += coef[tap_index];

        diff = COEF_UNIT - sum;
        if (diff && ADJUST_ERROR_ON)
            fprintf(stdout, "adjusting coefficients, error = %d\n", diff);

        tap_index = 0;
        while (diff)
        {
            // prepare compensation value
            sign = diff < 0 ? 1 : -1;
            // adds compensation value
            if (diff & 1) {        // odd difference
                coef[tap_num / 2] += sign;
                diff += sign;
            }
            else {               // even difference: split into 1+1
                coef[tap_num / 2 + tap_index] += sign;
                coef[tap_num / 2 - tap_index] += sign;
                diff += sign * 2;
                tap_index++;
            }

            if (tap_index == tap_num / 2) {
                fprintf(stderr, "Error: single phase unificaton\n");
                return false;
            }
        }
        return  true;
    }


    static bool gen_single_phase_coef(const char* path, const char* name,
        const int win_type, const int tap_num, const float cut_off,
        float* flt_coef, int* int_coef)
    {
        if (!(tap_num & 1))
        {
            std::cout << "ERROR: filter design must be with odd number for single phase" << std::endl;
            return false;
        }

        int         tap_index;

        float       data_sum, freq_sum;
        float       tap_loop, sample, sinc_data, win_data;
        float       win_size = float(tap_num / 2);        // truncate to integer, then float
        float* support = new float[tap_num];
        float* coef_flt = new float[tap_num];
        int* coef_int = new int[tap_num];
        float      (*window_func)(float const&, const float);

        FILE* fp = 0;
        char        file_name[200];

        // select different window
        switch (win_type) {
        case 0: window_func = &sin_window;      break;
        case 1: window_func = &cos_window;      break;
        case 2: window_func = &tri_window;      break;
        case 3: window_func = &nuttall_win;     break;
        case 4: window_func = &blackman_win;    break;
        case 5: window_func = &flat_top_win;    break;
        case 6: window_func = &gaussian_win;    break;
        case 7: window_func = &blackman_harris; break;
        case 8: window_func = &blackman_nuttal; break;
        default:window_func = &box_window;      break;
        }


        // luma filter coefficients
        for (tap_index = 0, tap_loop = -win_size; tap_loop <= win_size; tap_loop++, tap_index++)
        {
            sample = tap_loop;

            sinc_data = sinc_function(sample, cut_off);
            win_data = (*window_func)(sample, win_size);

            support[tap_index] = sinc_data * win_data;
        }

        // scale to unit
        data_sum = 0;
        for (tap_index = 0; tap_index < tap_num; tap_index++)
            data_sum += support[tap_index];

        for (tap_index = 0; tap_index < tap_num; tap_index++)
        {
            coef_flt[tap_index] = support[tap_index] / data_sum;
            coef_int[tap_index] = (int)(coef_flt[tap_index] * COEF_UNIT);
        }

        // Adjusting integer coefficients
        unify_coef(tap_num, coef_int);

        for (tap_index = 0; tap_index < tap_num; tap_index++)
        {
            if (flt_coef)
                flt_coef[tap_index] = coef_flt[tap_index];

            if (int_coef)
                int_coef[tap_index] = coef_int[tap_index];
        }

        // Printing out performance
        if (path && name)
        {
            // printout of time response
            struct stat info;
            if (stat(path, &info))
                system("mkdir graph"),
                sprintf(file_name, "./graph/single_phase_time_%s", name);
            else
                sprintf(file_name, "%s/single_phase_time_%s", path, name);
            fp = fopen(file_name, "wb");
            for (tap_index = 0; tap_index < tap_num; tap_index++)
                fprintf(fp, "%f %d\n", coef_flt[tap_index], coef_int[tap_index]);
            fclose(fp);

            // printout of frequency response
            sprintf(file_name, "%s/single_phase_freq_%s", path, name);
            fp = fopen(file_name, "wb");

            for (sample = 0; sample < FREQ_MAX_POINT; sample++)
            {
                freq_sum = 0;
                for (tap_index = 0, tap_loop = -win_size; tap_index < tap_num; tap_index++, tap_loop++)
                    freq_sum += coef_flt[tap_index] * cos(fPI * sample / FREQ_MAX_POINT * tap_loop);

                fprintf(fp, "%f ", freq_sum);

                freq_sum = 0;
                for (tap_index = 0, tap_loop = -win_size; tap_index < tap_num; tap_index++, tap_loop++)
                    freq_sum += (float)coef_int[tap_index] * cos(fPI * sample / FREQ_MAX_POINT * tap_loop);

                fprintf(fp, "%f\n", fabs(freq_sum));
            }
            fclose(fp);
        }

        delete[] coef_flt;
        delete[] coef_int;
        // finished
        return  true;
    }


    static bool gen_poly_phase_coef(const char* path, const char* name,
        const int win_type, const int tap_num, const int phs_num, const float cut_off,
        float* coef_flt, int* coef_int)
    {

        float* support = new float[tap_num], support_sum, freq_sum;
        float       phs_loop, tap_loop, sample;
        float       sinc_data, win_data;
        float       cutoff_val, cutoff_inc, att_at_one;
        int         phs_index, tap_index;

        float       win_size = tap_num / 2.0f;

        // a pointer for selection of different window functions
        float      (*window_func)(float const&, const float);

        // for output files
        FILE* fp = 0;
        char        file_name[200];

        // select different window
        switch (win_type) {
        case 0: window_func = sin_window;       break;
        case 1: window_func = cos_window;       break;
        case 2: window_func = tri_window;       break;
        case 3: window_func = nuttall_win;      break;
        case 4: window_func = blackman_win;     break;
        case 5: window_func = flat_top_win;     break;
        case 6: window_func = gaussian_win;     break;
        case 7: window_func = blackman_harris;  break;
        case 8: window_func = blackman_nuttal;  break;
        default:window_func = box_window;       break;
        }

        //=====================================================
        // Special Case: Interpolation Application
        // 1) cutoff may need to be shrunk to reduce alias
        // 2) in order to give a similar magnitude responses 
        //    in different phase, shrink bandwidths
        //    near phase 0 or phase PI by cutoff_inc
        //=====================================================
        cutoff_val = cut_off;

        if (cutoff_val == 1) {  // special design

            // in the middle phase
            for (phs_index = phs_num / 2, tap_index = 0,
                tap_loop = -win_size; tap_loop < win_size; tap_loop++,
                tap_index++)
            {
                sample = tap_loop + 0.5f;
                sinc_data = sinc_function(sample, cutoff_val);
                win_data = (*window_func)(sample, win_size);
                support[tap_index] = sinc_data * win_data;
            }

            for (support_sum = 0, tap_index = 0; tap_index < tap_num; tap_index++)
                support_sum += support[tap_index];

            for (tap_index = 0; tap_index < tap_num; tap_index++)
                support[tap_index] /= support_sum;

            // checks the attenuation      
            for (sample = FREQ_MAX_POINT * 0.98f, freq_sum = 0, tap_loop = -win_size + 0.5f,
                tap_index = 0; tap_index < tap_num; tap_index++, tap_loop++) {
                freq_sum += support[tap_index] * cos(fPI * sample / FREQ_MAX_POINT * tap_loop);
            }
            // records the attenuation
            att_at_one = fabs(freq_sum);  // attenuation at cutoff_val 1

            phs_index = 0;
            phs_loop = 0;
            freq_sum = 1;
            while (fabs(freq_sum) > att_at_one) {

                cutoff_val -= 0.01f;

                for (tap_index = 0, tap_loop = -win_size; tap_loop < win_size; tap_loop++, tap_index++)
                {
                    sample = tap_loop + phs_loop / (float)phs_num;
                    sinc_data = sinc_function(sample, cutoff_val);
                    win_data = (*window_func)(sample, win_size);
                    support[tap_index] = sinc_data * win_data;
                }

                for (support_sum = 0, tap_index = 0; tap_index < tap_num; tap_index++)
                    support_sum += support[tap_index];

                for (tap_index = 0; tap_index < tap_num; tap_index++)
                    support[tap_index] /= support_sum;

                for (sample = FREQ_MAX_POINT * 0.98, freq_sum = 0, tap_loop = -win_size + phs_loop,
                    tap_index = 0; tap_index < tap_num; tap_index++, tap_loop++)
                {
                    freq_sum += support[tap_index] * cos(fPI * sample / FREQ_MAX_POINT * tap_loop);
                }
            }

            cutoff_val = 0.85f;
            cutoff_inc = (1.0f - cutoff_val) * 2.0f / (phs_num + 1);
        }
        else {
            cutoff_inc = 0;
        }

        //================== WARNING WARNING ====================
        // if considering cut_off==1 must drop the bandwidth,
        // comment out the following assignment, otherwise
        // the following assignaments overwrite the previous
        // special case, in other words, the previous special
        // design is ignored.
        //================== WARNING WARNING =====================
        cutoff_inc = 0;
        cutoff_val = cut_off;

        for (phs_index = 0, phs_loop = 0; phs_index < phs_num; phs_loop++, phs_index++)
        {
            for (tap_index = 0, tap_loop = -win_size; tap_loop < win_size; tap_loop++, tap_index++)
            {
                // sample location = tap position + phase position
                sample = tap_loop + phs_loop / (float)phs_num;

                // Sinc values at samples
                sinc_data = sinc_function(sample, cutoff_val);
                // window values at samples
                win_data = (*window_func)(sample, win_size);

                // values of windowed Sinc at samples
                support[tap_index] = sinc_data * win_data;
            }

            //=====================
            // scale to unit one
            //=====================
            for (support_sum = 0, tap_index = 0; tap_index < tap_num; tap_index++)
            {
                support_sum += support[tap_index];
            }

            for (tap_index = 0; tap_index < tap_num; tap_index++)
            {
                support[tap_index] /= support_sum;
                if (coef_int)
                    coef_int[phs_index * tap_num + tap_index] = (int)(support[tap_index] * COEF_UNIT);

                if (coef_flt)
                    coef_flt[phs_index * tap_num + tap_index] = support[tap_index];
            }

            if (coef_int)
                unify_coef(tap_num, &coef_int[phs_index * tap_num]);

            //======================================================
            // experiments' codes : 
            // no cut_off reduction by cut_off_inc near phase 0 or PI.
            cutoff_val += (phs_index >= phs_num / 2) ? -cutoff_inc : +cutoff_inc;
        }

        //====================================================
        // the following are data printout for debugging
        //====================================================   
        if (path && name)
        {
            struct stat info;
            if (stat(path, &info))
                system("mkdir graph"),
                sprintf(file_name, "./graph/coef_%s", name);
            else
                sprintf(file_name, "%s/coef_%s", path, name);
            fp = fopen(file_name, "wb");
            for (phs_index = 0; phs_index < phs_num; phs_index++) {
                fprintf(fp, "\tphase %d\n", phs_index);
                for (tap_index = 0; tap_index < tap_num; tap_index++) {
                    fprintf(fp, "%x \t %f\n",
                        coef_int ? coef_int[phs_index * tap_num + tap_index] : 0,
                        coef_flt ? coef_flt[phs_index * tap_num + tap_index] : 0);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);

            sprintf(file_name, "%s/freq_%s", path, name);
            fp = fopen(file_name, "wb");
            for (phs_loop = 0, phs_index = 0; phs_index < phs_num; phs_index++, phs_loop += 1.0f / (float)phs_num) {
                for (sample = 0; sample < FREQ_MAX_POINT; sample++) {

                    freq_sum = 0;
                    for (tap_index = 0, tap_loop = -win_size + phs_loop; tap_index < tap_num; tap_index++, tap_loop++)
                        freq_sum += (coef_flt ? coef_flt[phs_index * tap_num + tap_index] : 0) * cos(fPI * sample / FREQ_MAX_POINT * tap_loop);
                    fprintf(fp, "%f\n", fabs(freq_sum));
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }

        delete[] support;

        return true;
    }

}

#endif