/*******************************************************************************************************************
 * FILE NAME   :    tools_math.h
 *
 * PROJECTION  :    DVE3G
 *
 * DESCRIPTION :    a file includes DVE related math calculations
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2013 JAN 09      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#ifndef __TOOLS_MATH_H__
#define __TOOLS_MATH_H__

using namespace std;

template <class FTYPE>
class mathCalc{
    typedef unsigned int    uint;
    typedef unsigned char   uchar;
    typedef unsigned short  ushort;


     FTYPE      CovarMatrix[3][3];
     FTYPE      InvCovMatrix[3][3];
     FTYPE      PRECISION;
     ushort     *HistBuf;   //HistBuf[256][256][256];

public:

    mathCalc( bool hist_on ){ PRECISION = 1e-10f; HistBuf = hist_on? (new ushort[256*256*256]) : NULL; }
    ~mathCalc( ){ if(HistBuf!=NULL) delete [] HistBuf; }

    void copy_matrix( FTYPE *matrix_in, int *size, FTYPE *matrix_out );
    void covar_matrix( int *data_in, int dim_size, int data_size, FTYPE* covar_out, FTYPE* avg_out );
    void covar_matrix_3d( int* data_x, int* data_y, int* data_z, int data_size );               // 3-dimension vector, using covarMatrix as output
    int  inverse_matrix( FTYPE* matrix_in, int dim_size, FTYPE* matrix_out, FTYPE* det_out );   // inverse matrix operation
    void multiply_matrix( FTYPE *matrix_a, FTYPE *matrix_b, int *size, FTYPE *matrix_ab );      // size[0]:A's row, size[1]:B's col, size[2]: A's col/B's row
    void identity_matrix( FTYPE *matrix, int size );                                            // square matrix
    void rotate_matrix( FTYPE *matrix_in, FTYPE angle, int axis, FTYPE *matrix_out );           // homogeneous matrix 4x4
    void set_shift_matrix( FTYPE *matrix, FTYPE shift, int axis );                              // homogeous matrix 4x4
    void set_scale_matrix( FTYPE *matrix, FTYPE scale, int axis );                              // homogeous matrix 4x4
    void perspective_matrix( FTYPE *matrix_in, FTYPE focus, FTYPE *matrix_out );                // homogeous matrix 4x4

    void image_histogram_rgb( ushort* input_r, ushort* input_g, ushort* input_b );
    void image_histogram_444( ushort* input_lm, short* input_cb, short* input_cr );
    void image_histogram_422( ushort* input_luma, short* input_chrm );

    void histogram( int *src_data, int data_dim, int data_size,
                    char *file_path, char *file_name, char print_on );

    void export_image_2dgram( char* file_path, char* file_name, int chnl, int smooth );

    void gaussian_random_field( FTYPE* covar, FTYPE* avg, FTYPE det,
                                int* x_rng, int* y_rng, char* path_name, char* file_name );


    int norm_calc( int *data_src, int *data_ref, FTYPE *color_wght, int data_dim, int measure );
    float fast_inv_sqrt( float number ); // fast inverse square root

private:    // internally used functions
    int array_3Dto1D( const int i, const int j, const int k ) const;

};
#endif
