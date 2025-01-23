/*******************************************************************************************************************
 * FILE NAME   :    math_tools.hpp
 *
 * PROJECTION  :    genreral purpose
 *
 * DESCRIPTION :    a file implements DVE related math calculations
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * ^^^^^^^^^^^      ^^^^^^          ^^^^^^^^
 * 2022 MAR 18      Yu Liu          migrated from math_tools.h/math_tools.cpp
 *
 ********************************************************************************************************************/
#include "all_common.h"


namespace math_calc
{
#define  PRECISION 1e-10

    inline int  array_3Dto1D(const int i, const int j, const int k) { return i * 256 * 256 + j * 256 + k; }

    //=======================================
    //       Covariance Matrix
    // cov(X) = E[ (X-E[X]) (X-E[X])^-T ]
    //        = E[X X^-1] - u u^-T
    //

    //=========================================
    //    Multivariant Gaussian Prior Model
    //
    //   Prior(X) = Lamda * exp(-X^T cov(X)^-1 X)
    //


    template<typename FTYPE = float>
    void  covar_matrix(
        int* data_in, int data_dim, int data_size,  //inputs: data vector, dimension, data size
        FTYPE* covar_out, FTYPE* avg_out)           //outputs: covariance matrix, average vector
    {
        FTYPE* data_diff;
        int     data_cnt, dim_cnt, var_xcnt, var_ycnt;
        int     data_addr, var_yaddr;

        data_diff = new FTYPE[data_dim];

        // clear data_average
        for (dim_cnt = 0; dim_cnt < data_dim; dim_cnt++)
            avg_out[dim_cnt] = 0;

        // accumulate data average
        for (data_cnt = 0, data_addr = 0; data_cnt < data_size; data_cnt++, data_addr += data_dim) {
            // averaging
            for (dim_cnt = 0; dim_cnt < data_dim; dim_cnt++)
                avg_out[dim_cnt] += data_in[data_addr + dim_cnt];
        }

        // averaging
        for (dim_cnt = 0; dim_cnt < data_dim; dim_cnt++)
            avg_out[dim_cnt] /= data_size;

        // clear covariance matrix
        for (dim_cnt = 0; dim_cnt < data_dim * data_dim; dim_cnt++)
            covar_out[dim_cnt] = 0;

        // calculates sum = (dataX-avgX)(dataX-avgX)^T
        for (data_cnt = 0, data_addr = 0; data_cnt < data_size; data_cnt++, data_addr += data_dim) {

            // generates difference
            for (dim_cnt = 0; dim_cnt < data_dim; dim_cnt++)
                data_diff[dim_cnt] = data_in[data_addr + dim_cnt] - avg_out[dim_cnt];

            // generates covar
            for (var_ycnt = 0, var_yaddr = 0; var_ycnt < data_dim; var_ycnt++, var_yaddr += data_dim) {
                //
                for (var_xcnt = 0; var_xcnt < data_dim; var_xcnt++)
                    covar_out[var_yaddr + var_xcnt] += data_diff[var_ycnt] * data_diff[var_xcnt];
            }
            // done
        }

        // calculates E[(dataX-avgX)(dataX-avgX)^T]
        for (var_yaddr = 0; var_yaddr < data_dim * data_dim; var_yaddr += data_dim) {
            for (var_xcnt = 0; var_xcnt < data_dim; var_xcnt++)
                covar_out[var_yaddr + var_xcnt] /= data_size;
        }

        delete[] data_diff;
    }

    // function inputs/outputs: input matrix, data dimension, inverse matrix, matrix determinat
    template<typename FTYPE = float>
    int  inverse_matrix(FTYPE* matrix_in, int data_dim, FTYPE* matrix_out, FTYPE* det_out)
    {
        int     row_cnt, lin_cnt, col_cnt, row_addr, lin_addr;
        int     singular, perm_cnt; // permutation count
        FTYPE   data, * matrix_mir;  // matrix_in mirror

        singular = 0;
        perm_cnt = 0;
        *det_out = 1;
        matrix_mir = new FTYPE[data_dim * data_dim];

        // Identity matrix
        for (row_cnt = 0; row_cnt < data_dim; row_cnt++) {

            row_addr = row_cnt * data_dim;
            for (col_cnt = 0; col_cnt < data_dim; col_cnt++) {
                // copy input to working matrix
                matrix_mir[row_addr + col_cnt] = matrix_in[row_addr + col_cnt];
                // identity matrix
                matrix_out[row_addr + col_cnt] = (row_cnt == col_cnt) ? 1.0f : 0.0f;
            }
        }

        // calculates the Left Echelon
        for (row_cnt = 0; row_cnt < data_dim; row_cnt++) {

            lin_cnt = row_cnt;

            // searches for non-zero leading rows
            row_addr = row_cnt * data_dim;
            lin_addr = row_cnt * data_dim;
            while (!matrix_mir[lin_addr + row_cnt]) {
                lin_cnt++;
                if (lin_cnt >= data_dim) {
                    singular = 1;
                    break;
                }
                lin_addr = lin_cnt * data_dim;
            }

            // singularity occurs, exit
            if (singular) {
                delete[] matrix_mir;
                return 1;
            }

            // rows-switching transform
            if (lin_cnt != row_cnt) {

                // increases the permutation count
                perm_cnt++;

                // address 
                row_addr = row_cnt * data_dim;
                lin_addr = lin_cnt * data_dim;

                // element calc in a row
                for (col_cnt = 0; col_cnt < data_dim; col_cnt++) {
                    // matrix_mir
                    data = matrix_mir[lin_addr + col_cnt];
                    matrix_mir[lin_addr + col_cnt] = matrix_mir[row_addr + col_cnt];
                    matrix_mir[row_addr + col_cnt] = data;
                    // matrix_out
                    data = matrix_out[lin_addr + col_cnt];
                    matrix_out[lin_addr + col_cnt] = matrix_out[row_addr + col_cnt];
                    matrix_out[row_addr + col_cnt] = data;
                }
            }

            // normalizes the current row
            data = matrix_mir[row_addr + row_cnt];
            if (fabs(data) < PRECISION) {
                fprintf(stderr, "Error: almost singular\n");
            }

            *det_out *= data;
            for (col_cnt = 0; col_cnt < data_dim; col_cnt++) {
                matrix_mir[row_addr + col_cnt] /= data;
                matrix_out[row_addr + col_cnt] /= data;
            }

            // multiply-add transform
            for (lin_cnt = row_cnt + 1; lin_cnt < data_dim; lin_cnt++) {

                lin_addr = lin_cnt * data_dim;
                data = matrix_mir[lin_addr + row_cnt];

                if (data) {  // if the leading element is non-zero

                    for (col_cnt = 0; col_cnt < data_dim; col_cnt++) {

                        matrix_mir[lin_addr + col_cnt] -= matrix_mir[row_addr + col_cnt] * data;
                        matrix_out[lin_addr + col_cnt] -= matrix_out[row_addr + col_cnt] * data;
                    }
                }
            }
        }

        // checks the left-bottom corner element
        data = matrix_mir[data_dim * data_dim - 1];
        if (!data) {
            singular = 1;
            delete[] matrix_mir;
            return  1;
        }
        else if (fabs(data) < PRECISION) {
            fprintf(stderr, "Error: almost singular\n");
        }

        *det_out *= data;
        if (perm_cnt & 1)  *det_out *= -1;

        // normalizes the bottom row
        matrix_mir[data_dim * data_dim - 1] = 1;
        row_addr = (data_dim - 1) * data_dim;
        for (col_cnt = 0; col_cnt < data_dim; col_cnt++)
            matrix_out[row_addr + col_cnt] /= data;

        // calculates the Right Upper echelon
        for (row_cnt = data_dim - 1; row_cnt > 0; row_cnt--) {

            row_addr = row_cnt * data_dim;  // start at the last column

            for (lin_cnt = 0; lin_cnt < row_cnt; lin_cnt++) {

                lin_addr = lin_cnt * data_dim;
                data = matrix_mir[lin_addr + row_cnt];
                if (data) {
                    matrix_mir[lin_addr + row_cnt] = 0;
                    for (col_cnt = 0; col_cnt < data_dim; col_cnt++)
                        matrix_out[lin_addr + col_cnt] -= matrix_out[row_addr + col_cnt] * data;
                }
            }
        }

        delete[] matrix_mir;
        return 0;
    }


    template<typename FTYPE = float>
    void multiply_matrix(FTYPE* matrix_a, FTYPE* matrix_b, int* size, FTYPE* matrix_ab)
    {
        // size[0] = matrix A's row size,
        // size[1] = matrix B's col size
        // size[2] = matrix A's col size & B's row size
        int     row_cnt, col_cnt, inner_cnt, row_addr;

        for (row_cnt = 0; row_cnt < size[0]; row_cnt++) {

            row_addr = row_cnt * size[1];

            for (col_cnt = 0; col_cnt < size[1]; col_cnt++) {

                matrix_ab[row_addr + col_cnt] = 0.0f;
                for (inner_cnt = 0; inner_cnt < size[2]; inner_cnt++)
                    matrix_ab[row_addr + col_cnt] += matrix_a[row_cnt * size[2] + inner_cnt] * matrix_b[inner_cnt * size[1] + col_cnt];
            }
        }
    }

    template<typename FTYPE = float>
    void identity_matrix(FTYPE* matrix, int size)
    {
        int     row_cnt, col_cnt, row_addr;
        for (row_cnt = 0; row_cnt < size; row_cnt++) {
            row_addr = row_cnt * size;
            for (col_cnt = 0; col_cnt < size; col_cnt++)
                matrix[row_addr + col_cnt] = (row_cnt == col_cnt) ? 1.0f : 0.0f;
        }
    }

    //===========================================
    //
    //
    //===========================================
    template<typename FTYPE = float>
    void rotate_matrix(FTYPE* input_matrix, FTYPE angle, int axis, FTYPE* output_matrix)
    {
        FTYPE   cos_a = cos(angle);
        FTYPE   sin_a = sin(angle);
        FTYPE   rotate_matrix[4][4];    //*rotate_matrix = malloc( sizeof(FTYPE)*4*4);
        int     size[3] = { 4,4,4 };

        identity_matrix(&rotate_matrix[0][0], 4);

        if (axis == 0) {        // around x axis
            rotate_matrix[1][1] = cos_a;
            rotate_matrix[1][2] = -sin_a;
            rotate_matrix[2][1] = sin_a;
            rotate_matrix[2][2] = cos_a;
        }
        else if (axis == 1) {  // around y axis
            rotate_matrix[0][0] = cos_a;
            rotate_matrix[0][2] = sin_a;
            rotate_matrix[2][0] = -sin_a;
            rotate_matrix[2][2] = cos_a;
        }
        else if (axis == 2) {  // around z axis
            rotate_matrix[0][0] = cos_a;
            rotate_matrix[0][1] = -sin_a;
            rotate_matrix[1][0] = sin_a;
            rotate_matrix[1][1] = cos_a;
        }
        else {
            std::fprintf(stdout, "Error: rotation parameter overflow\n");
        }

        multiply_matrix(&rotate_matrix[0][0], input_matrix, size, output_matrix);

        //free(rotate_matrix);
    }

    template<typename FTYPE = float>
    void set_shift_matrix(FTYPE* matrix, FTYPE shift, int axis)
    {
        matrix[axis * 4 + 3] = shift;
    }

    template<typename FTYPE = float>
    void set_scale_matrix(FTYPE* matrix, FTYPE scale, int axis)
    {
        matrix[axis * 4 + axis] *= scale;
    }

    template<typename FTYPE = float>
    void perspective_matrix(FTYPE* input_matrix, FTYPE focus, FTYPE* output_matrix)
    {

        FTYPE   psp_matrix[4][4];
        int     size[3] = { 4,4,4 };

        identity_matrix(&psp_matrix[0][0], 4);
        psp_matrix[3][2] = -1.0f / focus;
        psp_matrix[2][2] = 0.0f;        // view plane always on z=0, see pp.445 on Computer Graphic C Version

        multiply_matrix(&psp_matrix[0][0], input_matrix, size, output_matrix);
    }

    template<typename FTYPE = float>
    void copy_matrix(FTYPE* input_matrix, int* size, FTYPE* output_matrix)
    {
        int     row_cnt, col_cnt, row_addr;
        for (row_cnt = 0; row_cnt < size[0]; row_cnt++) {
            row_addr = row_cnt * size[1];
            for (col_cnt = 0; col_cnt < size[1]; col_cnt++) {
                output_matrix[row_addr + col_cnt] = input_matrix[row_addr + col_cnt];
            }
        }
    }


    template<typename FTYPE = float>
    void covar_matrix_3d(int* data_x, int* data_y, int* data_z, int data_size)
    {
        int     data_cnt;
        FTYPE   avg_x, avg_y, avg_z, diff_x, diff_y, diff_z;
        FTYPE   covar_xx, covar_xy, covar_xz, covar_yy, covar_yz, covar_zz;

        avg_x = avg_y = avg_z = 0.0f;

        for (data_cnt = 0; data_cnt < data_size; data_cnt++) {

            avg_x += data_x[data_cnt];
            avg_y += data_y[data_cnt];
            avg_z += data_z[data_cnt];
        }

        avg_x /= data_size;
        avg_y /= data_size;
        avg_z /= data_size;

        covar_xx = covar_xy = covar_xz = covar_yy = covar_yz = covar_zz = 0.0f;

        for (data_cnt = 0; data_cnt < data_size; data_cnt++) {

            diff_x = data_x[data_cnt] - avg_x;
            diff_y = data_y[data_cnt] - avg_y;
            diff_z = data_z[data_cnt] - avg_z;

            covar_xx += diff_x * diff_x;
            covar_xy += diff_x * diff_y;
            covar_xz += diff_x * diff_z;
            covar_yy += diff_y * diff_y;
            covar_yz += diff_y * diff_z;
            covar_zz += diff_z * diff_z;
        }

        covar_xx /= data_size;
        covar_xy /= data_size;
        covar_xz /= data_size;
        covar_yy /= data_size;
        covar_yz /= data_size;
        covar_zz /= data_size;

        CovarMatrix[0][0] = covar_xx;
        CovarMatrix[0][1] = covar_xy;
        CovarMatrix[0][2] = covar_xz;
        CovarMatrix[1][0] = covar_xy;
        CovarMatrix[1][1] = covar_yy;
        CovarMatrix[1][2] = covar_yz;
        CovarMatrix[2][0] = covar_xz;
        CovarMatrix[2][1] = covar_yz;
        CovarMatrix[2][2] = covar_zz;
    }


    //===========================================
    //
    // covar = | covar 0, covar 1 | = | xx  xy |
    //         | covar 2, covar 3 |   | yx  yy |
    //
    //===========================================
    template<typename FTYPE = float>
    void gaussian_random_field(FTYPE* covar, FTYPE* avg, FTYPE det,
        int* x_rng, int* y_rng, char* path_name, char* file_name)
    {
        FILE* fp;
        char    file_buf[200];
        int     x_cnt, y_cnt;
        FTYPE   x_diff, y_diff, y_covar_yx, y_covar_yy, x_covar, y_covar;
        FTYPE   xMx, pdf, log_pdf, log_coef;    // xMx = vector * covarMatrix * vector^T

        log_coef = log(fabs(det)) / 2.0f + log(2.0f * F_PI);

        sprintf(file_buf, "%s/%s", path_name, file_name);

        fp = fopen(file_buf, "wb");

        for (y_cnt = y_rng[0]; y_cnt < y_rng[1]; y_cnt++) {

            y_diff = y_cnt - avg[1];
            y_covar_yx = y_diff * covar[2];
            y_covar_yy = y_diff * covar[3];

            for (x_cnt = x_rng[0]; x_cnt < x_rng[1]; x_cnt++) {

                // 
                x_diff = x_cnt - avg[0];
                x_covar = x_diff * covar[0] + y_covar_yx;
                y_covar = x_diff * covar[1] + y_covar_yy;

                //------------------------------------
                // xMx = (x-avg) * M^-1 * (x-avg)^-T
                //------------------------------------
                xMx = x_covar * x_diff + y_covar * y_diff;

                log_pdf = -xMx / 2.0f - log_coef;

                pdf = exp(log_pdf);

                fprintf(fp, "%d %d %f\n", x_cnt, y_cnt, pdf);
            }

            fprintf(fp, "\n");
        }

        fclose(fp);
    }

    template<typename FTYPE = float>
    void histogram(int* data_src, int data_dim, int data_size,
        char* file_path, char* file_name, char print_on)
    {
        int     data_cnt, data_addr, dim_cnt;
        char    file_buf[200];
        FILE* fp;
        FTYPE   tmp_data;

        sprintf(file_buf, "%s/%s", file_path, file_name);

        for (data_cnt = 0; data_cnt < 256; data_cnt++) {
            for (data_addr = 0; data_addr < 256; data_addr++) {
                for (dim_cnt = 0; dim_cnt < 256; dim_cnt++) {

                    HistBuf[array_3Dto1D(data_cnt, data_addr, dim_cnt)] = 0;
                }
            }
        }

        for (data_cnt = data_addr = 0; data_cnt < data_size; data_cnt++, data_addr += data_dim) {

            switch (data_dim) {

            case 1:
                HistBuf[array_3Dto1D(0, 0, data_src[data_addr])]++;
                break;

            case 2:
                HistBuf[array_3Dto1D(0, data_src[data_addr + 1], data_src[data_addr])]++;
                break;

            case 3:
                HistBuf[array_3Dto1D(data_src[data_addr + 2], data_src[data_addr + 1], data_src[data_addr])]++;
                break;

            default:
                fprintf(stderr, "Error: histogram source data dim %d (options: 1/2/3\n", data_dim);
                exit(0);

            }
        }

        if (print_on) {

            fp = fopen(file_buf, "wb");
            if (!fp) {
                fprintf(stderr, "Error: no file %s\n", file_buf);
                exit(0);
            }

            if (data_dim == 1) {

                for (data_cnt = 0; data_cnt < 256; data_cnt++)
                    fprintf(fp, "%f\n", (FTYPE)HistBuf[array_3Dto1D(0, 0, data_cnt)] / (FTYPE)data_size);
            }
            else {

                for (data_cnt = 0; data_cnt < 256; data_cnt++) {
                    for (data_addr = 0; data_addr < 256; data_addr++) {

                        tmp_data = (FTYPE)HistBuf[array_3Dto1D(0, data_cnt, data_addr)] / (FTYPE)data_size;
                        fprintf(fp, "%f\n", tmp_data);
                    }
                    fprintf(fp, "\n");
                }
            }

            fclose(fp);
        }

    }


    //===========================================================================
    //      measure: 0 = L1 norm, 1 = L2 norm, 2 = Linf norm, >2 = square norm
    //============================================================================
    template<typename FTYPE = float>
    int norm_calc(int* data_src, int* data_ref, FTYPE* color_wght, int data_dim, int measure)
    {
        int     dim_cnt;
        FTYPE   tmp, diff, sum;

        sum = 0;
        for (dim_cnt = 0; dim_cnt < data_dim; dim_cnt++) {

            diff = FTYPE(data_src[dim_cnt] - data_ref[dim_cnt]);
            diff *= color_wght[dim_cnt];

            switch (measure) {

            case 0:
                diff = fabs(diff);
                sum += diff;
                break;
            case 1:
                diff *= diff;
                sum += diff;
                break;

            case 2:
                diff = fabs(diff);
                if (diff > sum)
                    sum = diff;
                break;
            default:
                diff *= diff;
                sum += diff;
                break;
            }
        }

        if (measure == 1) {
            tmp = sqrt(sum / 3.0f);
            sum = tmp + 0.5f;
        }

        return (int)(sum + 0.5f);
    }

    //===========================================================
    // fast inverse square root
    //===========================================================
    template<typename FTYPE = float>
    float  fast_inv_sqrt(float number)
    {
        long        i;
        float       x2, y;
        const float threehalfs = 1.5F;

        x2 = number * 0.5F;
        y = number;
        i = *(long*)&y;                       // evil floating point bit level hacking
        i = 0x5f3759df - (i >> 1);               // what the fuck?
        // 0x5fe6ec85e7de30daL for double
        y = *(float*)&i;
        y = y * (threehalfs - (x2 * y * y));   // 1st iteration
        //      y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

        return y;
    }
}
