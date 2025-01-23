/*******************************************************************************************************************
 * FILE NAME   :    video_inout.h
 *
 * PROJECTION  :    DVE3G
 *
 * DESCRIPTION :    a file defines a group of video data access functions.
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2013 JAN 09      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#ifndef __VIDEO_INOUT_H__
#define __VIDEO_INOUT_H__

#define TAPNUM_4TO2 32
static float    coef_source[TAPNUM_4TO2] = { -0.0011444091796875,
                                             -0.000213623046875, 
                                              0.00360107421875,  
                                              0.0000152587890625,
                                             -0.00775146484375,  
                                             -0.0001373291015625,
                                              0.0153350830078125,
                                              0,
                                             -0.027801513671875, 
                                              0.0000152587890625,
                                              0.049835205078125, 
                                             -0.0001220703125,   
                                             -0.0972442626953125,
                                              0.0001373291015625,
                                              0.315277099609375, 
                                              0.49981689453125,  
                                              0.315277099609375, 
                                              0.0001373291015625,
                                             -0.0972442626953125,
                                             -0.0001220703125,   
                                              0.049835205078125, 
                                              0.0000152587890625,
                                             -0.027801513671875, 
                                              0,
                                              0.0153350830078125,
                                             -0.0001373291015625,
                                             -0.00775146484375,  
                                              0.0000152587890625,
                                              0.00360107421875,  
                                             -0.000213623046875, 
                                             -0.0011444091796875};
using namespace std;

class vidInOut{

private:    // data members
    static const int   COEFlvl =1024*32;
    static const int   COEFbit =15;
    static const int   IS_HSL =1;

    short       luma_lowpass_buf[TAPNUM_4TO2];
    short       chrm_lowpass_buf[2][TAPNUM_4TO2];
    int         lowpass_coef[TAPNUM_4TO2];

private:    // internal use only
    void rgb2hsl( const short *color, short *lum, short *sat, short *hue );
    void hsl2rgb( const short luma, const short *chrm, short *color );
    FILE* ppm_header_read( const bool snt, const char *file_name );
    FILE* pgm_header_read( const bool snt, const char *file_name );
    FILE* ppm_src_header( const char *file_name );
    FILE* ppm_header_write( const char *file_name );
    FILE* pgm_header_write( const char *file_name );
    FILE* tga_header_write( const char *file_name );
    FILE* tga_src_header( const char *file_name );

public:     // processing interface
    
    vidInOut( ){
        std::cout<<"Object inst: Video in/out"<<std::endl;

        for(int tap_cnt=0; tap_cnt<TAPNUM_4TO2; tap_cnt++)
            lowpass_coef[tap_cnt] = int(coef_source[tap_cnt] * COEFlvl);
    }

    ~vidInOut( ){
        std::cout<<"Object done: Video in/out"<<std::endl;
    }

    //------------------- color space conversion ----------------------
    void rgb2yuv( const short color[3], short *luma, short chrm[2] );
    void yuv2rgb( const short luma, const short chrm[2], short color[3] );

    void luma_lowpass_filter( const short luma_input, short* luma_output );
    void chrm_lowpass_filter( const short* chrm_input, short* chrm_output );

    void chroma_interleave( const short cb_input[SRC_HEIGHT][SRC_WIDTH],
                            const short cr_input[SRC_HEIGHT][SRC_WIDTH],
                            short chrm_output[SRC_HEIGHT][SRC_WIDTH] );

    //-------------------- File Inputs ------------------------
    void ppm_rgb_input( const char *file_path, const char *file_name, const int frame_num, const bool anim_on,
                        short output_r[SRC_HEIGHT][SRC_WIDTH],
                        short output_g[SRC_HEIGHT][SRC_WIDTH],
                        short output_b[SRC_HEIGHT][SRC_WIDTH] );

    void ppm_tgt_input( const char *file_path, const char *file_name, const int frame_num, const bool anim_on,
                        short output_r[TGT_HEIGHT][TGT_WIDTH],
                        short output_g[TGT_HEIGHT][TGT_WIDTH],
                        short output_b[TGT_HEIGHT][TGT_WIDTH] );

    void ppm_444_input( const char *file_path, const char *file_name, const int frame_num, const bool anim_on,
                        short output_lm[SRC_HEIGHT][SRC_WIDTH],
                        short output_cb[SRC_HEIGHT][SRC_WIDTH],
                        short output_cr[SRC_HEIGHT][SRC_WIDTH] );

    void ppm_video_input( const char *file_path, const char *file_name, const int frame_num, const bool anim_on,
                          short output_luma[SRC_HEIGHT][SRC_WIDTH],
                          short output_chrm[SRC_HEIGHT][SRC_WIDTH] );

    void ppm_target_input( const char *file_path, const char *file_name, const int frame_num, const bool anim_on,
                           short output_luma[TGT_HEIGHT][TGT_WIDTH],
                           short output_chrm[TGT_HEIGHT][TGT_WIDTH] );

    //---------------------- File Output --------------------------
    void ppm_video_splitout( const char *file_path, const char *file_name, const int frame_num,
                             const short input_luma[TGT_HEIGHT][TGT_WIDTH],
                             const short input_chrm[TGT_HEIGHT][TGT_WIDTH] );

    void ppm_video_output( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                           const short input_luma[TGT_HEIGHT][TGT_WIDTH],
                           const short input_chrm[TGT_HEIGHT][TGT_WIDTH] );

    void ppm_444_output( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                         const short input_lm[TGT_HEIGHT][TGT_WIDTH],
                         const short input_cb[TGT_HEIGHT][TGT_WIDTH],
                         const short input_cr[TGT_HEIGHT][TGT_WIDTH] );

    void ppm_rgb_vidout( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                         const short input_r[SRC_HEIGHT][SRC_WIDTH],
                         const short input_g[SRC_HEIGHT][SRC_WIDTH],
                         const short input_b[SRC_HEIGHT][SRC_WIDTH] );

    void ppm_yuv_vidout( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                         const short input_lm[SRC_HEIGHT][SRC_WIDTH],
                         const short input_cb[SRC_HEIGHT][SRC_WIDTH],
                         const short input_cr[SRC_HEIGHT][SRC_WIDTH] );

    void ppm_src_vidout( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                         const short input_luma[SRC_HEIGHT][SRC_WIDTH],
                         const short input_chrm[SRC_HEIGHT][SRC_WIDTH] );

    void pgm_luma_output( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                          const short input_luma[TGT_HEIGHT][TGT_WIDTH] );

    void pgm_chrm_output( const char *file_path, const char *file_name, const int frame_num, const int split_chroma,
                          const short input_chrm[TGT_HEIGHT][TGT_WIDTH] );

    void tga_video_output( const char *file_path, const char *file_name, const int frame_num, const bool split_on, const bool is_ntsc,
                           const short input_luma[TGT_HEIGHT][TGT_WIDTH],
                           const short input_chrm[TGT_HEIGHT][TGT_WIDTH],
                           const short input_alph[TGT_HEIGHT][TGT_WIDTH] );
 
    void tga_src_vidout( const char *file_path, const char *file_name, const int frame_num, const bool split_on, const bool is_ntsc,
                         const short input_luma[SRC_HEIGHT][SRC_WIDTH],
                         const short input_chrm[SRC_HEIGHT][SRC_WIDTH],
                         const short input_alph[SRC_HEIGHT][SRC_WIDTH] );

    void tga_4444_vidout( const char *file_path, const char *file_name, const int frame_num, const bool is_ntsc,
                          const short input_lm[TGT_HEIGHT][TGT_WIDTH],
                          const short input_cb[TGT_HEIGHT][TGT_WIDTH],
                          const short input_cr[TGT_HEIGHT][TGT_WIDTH],
                          const short input_alph[TGT_HEIGHT][TGT_WIDTH] );

    void ppm_indx_vidout( const char *file_path, const char *file_name, const int frame_num,
                          const int *palette, const uchar color_num,
                          const uchar img_indx[TGT_HEIGHT][TGT_WIDTH] );

    void ppm_oddeven_swap( const char *file_path, const char *file_name, const int frame_num, const bool anim_on );

    void ppm_rg0_output( const char *file_path, const char *file_name, const int frame_num, const bool split_on,
                         const short img_input[TGT_HEIGHT][TGT_WIDTH*2] );

    void ppm_auto_keying( const char *file_path, const char *file_name, const char *bkgd_path, const char *bkgd_name,
                          const char *video_path, const char *video_name, const char *alpha_path, const char *alpha_name,
                          const int frame_num, const bool key_property );

    void ppm_pattern_load( const char *file_path, const char *file_name, const int hsize, const int vsize,
                           short *lm_dout, short *cb_dout, short *cr_dout );
    void pgm_alpha_load( const char *file_path, const char *file_name, const int hsize, const int vsize, short *alpha_dout );

};

#endif


/*==============================================================================

  TGA image header info


DATA TYPE 1: Color-mapped images 
________________________________________________________________________________
| Offset | Length |                     Description                            |
|--------|--------|------------------------------------------------------------|
|    0   |     1  |  Number of Characters in Identification Field.             |
|        |        |                                                            |
|        |        |  This field is a one-byte unsigned integer, specifying     |
|        |        |  the length of the Image Identification Field.  Its range  |
|        |        |  is 0 to 255.  A value of 0 means that no Image            |
|        |        |  Identification Field is included.                         |
|--------|--------|------------------------------------------------------------|
|    1   |     1  |  Color Map Type.                                           |
|        |        |                                                            |
|        |        |  This field contains a binary 1 for Data Type 1 images.    |
|--------|--------|------------------------------------------------------------|
|    2   |     1  |  Image Type Code.                                          |
|        |        |                                                            |
|        |        |  This field will always contain a binary 1.                |
|        |        |  ( That's what makes it Data Type 1 ).                     |
|--------|--------|------------------------------------------------------------|
|    3   |     5  |  Color Map Specification.                                  |
|        |        |                                                            |
|    3   |     2  |  Color Map Origin.                                         |
|        |        |  Integer ( lo-hi ) index of first color map entry.         |
|    5   |     2  |  Color Map Length.                                         |
|        |        |  Integer ( lo-hi ) count of color map entries.             |
|    7   |     1  |  Color Map Entry Size.                                     |
|        |        |  Number of bits in each color map entry.  16 for           |
|        |        |  the Targa 16, 24 for the Targa 24, 32 for the Targa 32.   |
|--------|--------|------------------------------------------------------------|
|    8   |    10  |  Image Specification.                                      |
|        |        |                                                            |
|    8   |     2  |  X Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) X coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   10   |     2  |  Y Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) Y coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   12   |     2  |  Width of Image.                                           |
|        |        |  Integer ( lo-hi ) width of the image in pixels.           |
|   14   |     2  |  Height of Image.                                          |
|        |        |  Integer ( lo-hi ) height of the image in pixels.          |
|   16   |     1  |  Image Pixel Size.                                         |
|        |        |  Number of bits in a stored pixel index.                   |
|   17   |     1  |  Image Descriptor Byte.                                    |
|        |        |  Bits 3-0 - number of attribute bits associated with each  |
|        |        |             pixel.                                         |
|        |        |  Bit 4    - reserved.  Must be set to 0.                   |
|        |        |  Bit 5    - screen origin bit.                             |
|        |        |             0 = Origin in lower left-hand corner.          |
|        |        |             1 = Origin in upper left-hand corner.          |
|        |        |             Must be 0 for Truevision images.               |
|        |        |  Bits 7-6 - Data storage interleaving flag.                |
|        |        |             00 = non-interleaved.                          |
|        |        |             01 = two-way (even/odd) interleaving.          |
|        |        |             10 = four way interleaving.                    |
|        |        |             11 = reserved.                                 |
|        |        |  This entire byte should be set to 0.  Don't ask me.       |
|--------|--------|------------------------------------------------------------|
|   18   | varies |  Image Identification Field.                               |
|        |        |                                                            |
|        |        |  Contains a free-form identification field of the length   |
|        |        |  specified in byte 1 of the image record.  It's usually    |
|        |        |  omitted ( length in byte 1 = 0 ), but can be up to 255    |
|        |        |  characters.  If more identification information is        |
|        |        |  required, it can be stored after the image data.          |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Color map data.                                           |
|        |        |                                                            |
|        |        |  The offset is determined by the size of the Image         |
|        |        |  Identification Field.  The length is determined by        |
|        |        |  the Color Map Specification, which describes the          |
|        |        |  size of each entry and the number of entries.             |
|        |        |  Each color map entry is 2, 3, or 4 bytes.                 |
|        |        |  Unused bits are assumed to specify attribute bits.        |
|        |        |  The 4 byte entry contains 1 byte for blue, 1 byte         |
|        |        |  for green, 1 byte for red, and 1 byte of attribute        |
|        |        |  information, in that order.                               |
|        |        |  The 3 byte entry contains 1 byte each of blue, green,     |
|        |        |  and red.                                                  |
|        |        |  The 2 byte entry is broken down as follows:               |
|        |        |  ARRRRRGG GGGBBBBB, where each letter represents a bit.    |
|        |        |  But, because of the lo-hi storage order, the first byte   |
|        |        |  coming from the file will actually be GGGBBBBB, and the   |
|        |        |  second will be ARRRRRGG. "A" represents an attribute bit. |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Image Data Field.                                         |
|        |        |                                                            |
|        |        |  This field specifies (width) x (height) color map         |
|        |        |  indices.  Each index is stored as an integral number      |
|        |        |  of bytes (typically 1 or 2).   All fields are unsigned.   |
|        |        |  The low-order byte of a two-byte field is stored first.   |
--------------------------------------------------------------------------------

DATA TYPE 2: Unmapped RGB 
________________________________________________________________________________
| Offset | Length |                     Description                            |
|--------|--------|------------------------------------------------------------|
|    0   |     1  |  Number of Characters in Identification Field.             |
|        |        |                                                            |
|        |        |  This field is a one-byte unsigned integer, specifying     |
|        |        |  the length of the Image Identification Field.  Its value  |
|        |        |  is 0 to 255.  A value of 0 means that no Image            |
|        |        |  Identification Field is included.                         |
|--------|--------|------------------------------------------------------------|
|    1   |     1  |  Color Map Type.                                           |
|        |        |                                                            |
|        |        |  This field contains either 0 or 1.  0 means no color map  |
|        |        |  is included.  1 means a color map is included, but since  |
|        |        |  this is an unmapped image it is usually ignored.  TIPS    |
|        |        |  ( a Targa paint system ) will set the border color        |
|        |        |  the first map color if it is present.                     |
|--------|--------|------------------------------------------------------------|
|    2   |     1  |  Image Type Code.                                          |
|        |        |                                                            |
|        |        |  This field will always contain a binary 2.                |
|        |        |  ( That's what makes it Data Type 2 ).                     |
|--------|--------|------------------------------------------------------------|
|    3   |     5  |  Color Map Specification.                                  |
|        |        |                                                            |
|        |        |  Ignored if Color Map Type is 0; otherwise, interpreted    |
|        |        |  as follows:                                               |
|    3   |     2  |  Color Map Origin.                                         |
|        |        |  Integer ( lo-hi ) index of first color map entry.         |
|    5   |     2  |  Color Map Length.                                         |
|        |        |  Integer ( lo-hi ) count of color map entries.             |
|    7   |     1  |  Color Map Entry Size.                                     |
|        |        |  Number of bits in color map entry.  16 for the Targa 16,  |
|        |        |  24 for the Targa 24, 32 for the Targa 32.                 |
|--------|--------|------------------------------------------------------------|
|    8   |    10  |  Image Specification.                                      |
|        |        |                                                            |
|    8   |     2  |  X Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) X coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   10   |     2  |  Y Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) Y coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   12   |     2  |  Width of Image.                                           |
|        |        |  Integer ( lo-hi ) width of the image in pixels.           |
|   14   |     2  |  Height of Image.                                          |
|        |        |  Integer ( lo-hi ) height of the image in pixels.          |
|   16   |     1  |  Image Pixel Size.                                         |
|        |        |  Number of bits in a pixel.  This is 16 for Targa 16,      |
|        |        |  24 for Targa 24, and .... well, you get the idea.         |
|   17   |     1  |  Image Descriptor Byte.                                    |
|        |        |  Bits 3-0 - number of attribute bits associated with each  |
|        |        |             pixel.  For the Targa 16, this would be 0 or   |
|        |        |             1.  For the Targa 24, it should be 0.  For     |
|        |        |             Targa 32, it should be 8.                      |
|        |        |  Bit 4    - reserved.  Must be set to 0.                   |
|        |        |  Bit 5    - screen origin bit.                             |
|        |        |             0 = Origin in lower left-hand corner.          |
|        |        |             1 = Origin in upper left-hand corner.          |
|        |        |             Must be 0 for Truevision images.               |
|        |        |  Bits 7-6 - Data storage interleaving flag.                |
|        |        |             00 = non-interleaved.                          |
|        |        |             01 = two-way (even/odd) interleaving.          |
|        |        |             10 = four way interleaving.                    |
|        |        |             11 = reserved.                                 |
|--------|--------|------------------------------------------------------------|
|   18   | varies |  Image Identification Field.                               |
|        |        |                                                            |
|        |        |  Contains a free-form identification field of the length   |
|        |        |  specified in byte 1 of the image record.  It's usually    |
|        |        |  omitted ( length in byte 1 = 0 ), but can be up to 255    |
|        |        |  characters.  If more identification information is        |
|        |        |  required, it can be stored after the image data.          |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Color map data.                                           |
|        |        |                                                            |
|        |        |  If the Color Map Type is 0, this field doesn't exist.     |
|        |        |  Otherwise, just read past it to get to the image.         |
|        |        |  The Color Map Specification describes the size of each    |
|        |        |  entry, and the number of entries you'll have to skip.     |
|        |        |  Each color map entry is 2, 3, or 4 bytes.                 |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Image Data Field.                                         |
|        |        |                                                            |
|        |        |  This field specifies (width) x (height) pixels.  Each     |
|        |        |  pixel specifies an RGB color value, which is stored as    |
|        |        |  an integral number of bytes.                              |
|        |        |  The 2 byte entry is broken down as follows:               |
|        |        |  ARRRRRGG GGGBBBBB, where each letter represents a bit.    |
|        |        |  But, because of the lo-hi storage order, the first byte   |
|        |        |  coming from the file will actually be GGGBBBBB, and the   |
|        |        |  second will be ARRRRRGG. "A" represents an attribute bit. |
|        |        |  The 3 byte entry contains 1 byte each of blue, green,     |
|        |        |  and red.                                                  |
|        |        |  The 4 byte entry contains 1 byte each of blue, green,     |
|        |        |  red, and attribute.  For faster speed (because of the     |
|        |        |  hardware of the Targa board itself), Targa 24 images are  |
|        |        |  sometimes stored as Targa 32 images.                      |
--------------------------------------------------------------------------------

DATA TYPE 9: Run Length Encoded, color-mapped images 
--------------------------------------------------------------------------------
| Offset | Length |                     Description                            |
|--------|--------|------------------------------------------------------------|
|    0   |     1  |  Number of Characters in Identification Field.             |
|        |        |                                                            |
|        |        |  This field is a one-byte unsigned integer, specifying     |
|        |        |  the length of the Image Identification Field.  Its value  |
|        |        |  is 0 to 255.  A value of 0 means that no Image            |
|        |        |  Identification Field is included.                         |
|--------|--------|------------------------------------------------------------|
|    1   |     1  |  Color Map Type.                                           |
|        |        |                                                            |
|        |        |  This field is always 1 for color-mapped images.           |
|--------|--------|------------------------------------------------------------|
|    2   |     1  |  Image Type Code.                                          |
|        |        |                                                            |
|        |        |  A binary 9 for this data type.                            |
|--------|--------|------------------------------------------------------------|
|    3   |     5  |  Color Map Specification.                                  |
|        |        |                                                            |
|    3   |     2  |  Color Map Origin.                                         |
|        |        |  Integer ( lo-hi ) index of first color map entry.         |
|    5   |     2  |  Color Map Length.                                         |
|        |        |  Integer ( lo-hi ) count of color map entries.             |
|    7   |     1  |  Color Map Entry Size.                                     |
|        |        |  Number of bits in each color map entry.  16 for the       |
|        |        |  Targa 16, 24 for the Targa 24, 32 for the Targa 32.       |
|--------|--------|------------------------------------------------------------|
|    8   |    10  |  Image Specification.                                      |
|        |        |                                                            |
|    8   |     2  |  X Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) X coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   10   |     2  |  Y Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) Y coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   12   |     2  |  Width of Image.                                           |
|        |        |  Integer ( lo-hi ) width of the image in pixels.           |
|   14   |     2  |  Height of Image.                                          |
|        |        |  Integer ( lo-hi ) height of the image in pixels.          |
|   16   |     1  |  Image Pixel Size.                                         |
|        |        |  Number of bits in a pixel.  This is 16 for Targa 16,      |
|        |        |  24 for Targa 24, and .... well, you get the idea.         |
|   17   |     1  |  Image Descriptor Byte.                                    |
|        |        |  Bits 3-0 - number of attribute bits associated with each  |
|        |        |             pixel.  For the Targa 16, this would be 0 or   |
|        |        |             1.  For the Targa 24, it should be 0.  For the |
|        |        |             Targa 32, it should be 8.                      |
|        |        |  Bit 4    - reserved.  Must be set to 0.                   |
|        |        |  Bit 5    - screen origin bit.                             |
|        |        |             0 = Origin in lower left-hand corner.          |
|        |        |             1 = Origin in upper left-hand corner.          |
|        |        |             Must be 0 for Truevision images.               |
|        |        |  Bits 7-6 - Data storage interleaving flag.                |
|        |        |             00 = non-interleaved.                          |
|        |        |             01 = two-way (even/odd) interleaving.          |
|        |        |             10 = four way interleaving.                    |
|        |        |             11 = reserved.                                 |
|--------|--------|------------------------------------------------------------|
|   18   | varies |  Image Identification Field.                               |
|        |        |                                                            |
|        |        |  Contains a free-form identification field of the length   |
|        |        |  specified in byte 1 of the image record.  It's usually    |
|        |        |  omitted ( length in byte 1 = 0 ), but can be up to 255    |
|        |        |  characters.  If more identification information is        |
|        |        |  required, it can be stored after the image data.          |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Color map data.                                           |
|        |        |                                                            |
|        |        |  The offset is determined by the size of the Image         |
|        |        |  Identification Field.  The length is determined by        |
|        |        |  the Color Map Specification, which describes the          |
|        |        |  size of each entry and the number of entries.             |
|        |        |  Each color map entry is 2, 3, or 4 bytes.                 |
|        |        |  Unused bits are assumed to specify attribute bits.        |
|        |        |  The 4 byte entry contains 1 byte for blue, 1 byte         |
|        |        |  for green, 1 byte for red, and 1 byte of attribute        |
|        |        |  information, in that order.                               |
|        |        |  The 3 byte entry contains 1 byte each of blue, green,     |
|        |        |  and red.                                                  |
|        |        |  The 2 byte entry is broken down as follows:               |
|        |        |  ARRRRRGG GGGBBBBB, where each letter represents a bit.    |
|        |        |  But, because of the lo-hi storage order, the first byte   |
|        |        |  coming from the file will actually be GGGBBBBB, and the   |
|        |        |  second will be ARRRRRGG. "A" represents an attribute bit. |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Image Data Field.                                         |
|        |        |                                                            |
|        |        |  This field specifies (width) x (height) color map         |
|        |        |  indices.  The indices are stored in packets.  There       |
|        |        |  two types of packets:  Run-length packets, and Raw        |
|        |        |  packets.  Both types of packets consist of a 1-byte       |
|        |        |  header, identifying the type of packet and specifying a   |
|        |        |  count, followed by a variable-length body.                |
|        |        |  The high-order bit of the header is "1" for the           |
|        |        |  run length packet, and "0" for the raw packet.            |
|        |        |                                                            |
|        |        |  For the run-length packet, the header consists of:        |
|        |        |      __________________________________________________    |
|        |        |      | 1 bit |   7 bit repetition count minus 1.      |    |
|        |        |      |   ID  |   Since the maximum value of this      |    |
|        |        |      |       |   field is 127, the largest possible   |    |
|        |        |      |       |   run size would be 128.               |    |
|        |        |      |-------|----------------------------------------|    |
|        |        |      |   1   |  C     C     C     C     C     C    C  |    |
|        |        |      --------------------------------------------------    |
|        |        |                                                            |
|        |        |  For the raw packet, the header consists of:               |
|        |        |      __________________________________________________    |
|        |        |      | 1 bit |   7 bit number of pixels minus 1.      |    |
|        |        |      |   ID  |   Since the maximum value of this      |    |
|        |        |      |       |   field is 127, there can never be     |    |
|        |        |      |       |   more than 128 pixels per packet.     |    |
|        |        |      |-------|----------------------------------------|    |
|        |        |      |   0   |  N     N     N     N     N     N    N  |    |
|        |        |      --------------------------------------------------    |
|        |        |                                                            |
|        |        |  For the run length packet, the header is followed by      |
|        |        |  a single color index, which is assumed to be repeated     |
|        |        |  the number of times specified in the header.  The RLE     |
|        |        |  packet may cross scan lines ( begin on one line and end   |
|        |        |  on the next ).                                            |
|        |        |  For the raw packet, the header is followed by the number  |
|        |        |  of color indices specified in the header.  The raw        |
|        |        |  packet may cross scan lines ( begin on one line and end   |
|        |        |  on the next ).                                            |
--------------------------------------------------------------------------------

DATA TYPE 10: Run Length Encoded, RGB images 
________________________________________________________________________________
| Offset | Length |                     Description                            |
|--------|--------|------------------------------------------------------------|
|    0   |     1  |  Number of Characters in Identification Field.             |
|        |        |                                                            |
|        |        |  This field is a one-byte unsigned integer, specifying     |
|        |        |  the length of the Image Identification Field.  Its range  |
|        |        |  is 0 to 255.  A value of 0 means that no Image            |
|        |        |  Identification Field is included.                         |
|--------|--------|------------------------------------------------------------|
|    1   |     1  |  Color Map Type.                                           |
|        |        |                                                            |
|        |        |  This field contains either 0 or 1.  0 means no color map  |
|        |        |  is included.  1 means a color map is included, but since  |
|        |        |  this is an unmapped image it is usually ignored.  TIPS    |
|        |        |  ( a Targa paint system ) will set the border color        |
|        |        |  the first map color if it is present.  Wowie zowie.       |
|--------|--------|------------------------------------------------------------|
|    2   |     1  |  Image Type Code.                                          |
|        |        |                                                            |
|        |        |  Binary 10 for this type of image.                         |
|--------|--------|------------------------------------------------------------|
|    3   |     5  |  Color Map Specification.                                  |
|        |        |                                                            |
|        |        |  Ignored if Color Map Type is 0; otherwise, interpreted    |
|        |        |  as follows:                                               |
|    3   |     2  |  Color Map Origin.                                         |
|        |        |  Integer ( lo-hi ) index of first color map entry.         |
|    5   |     2  |  Color Map Length.                                         |
|        |        |  Integer ( lo-hi ) count of color map entries.             |
|    7   |     1  |  Color Map Entry Size.                                     |
|        |        |  Number of bits in color map entry.  This value is 16 for  |
|        |        |  the Targa 16, 24 for the Targa 24, 32 for the Targa 32.   |
|--------|--------|------------------------------------------------------------|
|    8   |    10  |  Image Specification.                                      |
|        |        |                                                            |
|    8   |     2  |  X Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) X coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   10   |     2  |  Y Origin of Image.                                        |
|        |        |  Integer ( lo-hi ) Y coordinate of the lower left corner   |
|        |        |  of the image.                                             |
|   12   |     2  |  Width of Image.                                           |
|        |        |  Integer ( lo-hi ) width of the image in pixels.           |
|   14   |     2  |  Height of Image.                                          |
|        |        |  Integer ( lo-hi ) height of the image in pixels.          |
|   16   |     1  |  Image Pixel Size.                                         |
|        |        |  Number of bits in a pixel.  This is 16 for Targa 16,      |
|        |        |  24 for Targa 24, and .... well, you get the idea.         |
|   17   |     1  |  Image Descriptor Byte.                                    |
|        |        |  Bits 3-0 - number of attribute bits associated with each  |
|        |        |             pixel.  For the Targa 16, this would be 0 or   |
|        |        |             1.  For the Targa 24, it should be 0.  For the |
|        |        |             Targa 32, it should be 8.                      |
|        |        |  Bit 4    - reserved.  Must be set to 0.                   |
|        |        |  Bit 5    - screen origin bit.                             |
|        |        |             0 = Origin in lower left-hand corner.          |
|        |        |             1 = Origin in upper left-hand corner.          |
|        |        |             Must be 0 for Truevision images.               |
|        |        |  Bits 7-6 - Data storage interleaving flag.                |
|        |        |             00 = non-interleaved.                          |
|        |        |             01 = two-way (even/odd) interleaving.          |
|        |        |             10 = four way interleaving.                    |
|        |        |             11 = reserved.                                 |
|--------|--------|------------------------------------------------------------|
|   18   | varies |  Image Identification Field.                               |
|        |        |  Contains a free-form identification field of the length   |
|        |        |  specified in byte 1 of the image record.  It's usually    |
|        |        |  omitted ( length in byte 1 = 0 ), but can be up to 255    |
|        |        |  characters.  If more identification information is        |
|        |        |  required, it can be stored after the image data.          |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Color map data.                                           |
|        |        |                                                            |
|        |        |  If the Color Map Type is 0, this field doesn't exist.     |
|        |        |  Otherwise, just read past it to get to the image.         |
|        |        |  The Color Map Specification, describes the size of each   |
|        |        |  entry, and the number of entries you'll have to skip.     |
|        |        |  Each color map entry is 2, 3, or 4 bytes.                 |
|--------|--------|------------------------------------------------------------|
| varies | varies |  Image Data Field.                                         |
|        |        |                                                            |
|        |        |  This field specifies (width) x (height) pixels.  The      |
|        |        |  RGB color information for the pixels is stored in         |
|        |        |  packets.  There are two types of packets:  Run-length     |
|        |        |  encoded packets, and raw packets.  Both have a 1-byte     |
|        |        |  header, identifying the type of packet and specifying a   |
|        |        |  count, followed by a variable-length body.                |
|        |        |  The high-order bit of the header is "1" for the           |
|        |        |  run length packet, and "0" for the raw packet.            |
|        |        |                                                            |
|        |        |  For the run-length packet, the header consists of:        |
|        |        |      __________________________________________________    |
|        |        |      | 1 bit |   7 bit repetition count minus 1.      |    |
|        |        |      |   ID  |   Since the maximum value of this      |    |
|        |        |      |       |   field is 127, the largest possible   |    |
|        |        |      |       |   run size would be 128.               |    |
|        |        |      |-------|----------------------------------------|    |
|        |        |      |   1   |  C     C     C     C     C     C    C  |    |
|        |        |      --------------------------------------------------    |
|        |        |                                                            |
|        |        |  For the raw packet, the header consists of:               |
|        |        |      __________________________________________________    |
|        |        |      | 1 bit |   7 bit number of pixels minus 1.      |    |
|        |        |      |   ID  |   Since the maximum value of this      |    |
|        |        |      |       |   field is 127, there can never be     |    |
|        |        |      |       |   more than 128 pixels per packet.     |    |
|        |        |      |-------|----------------------------------------|    |
|        |        |      |   0   |  N     N     N     N     N     N    N  |    |
|        |        |      --------------------------------------------------    |
|        |        |                                                            |
|        |        |  For the run length packet, the header is followed by      |
|        |        |  a single color value, which is assumed to be repeated     |
|        |        |  the number of times specified in the header.  The         |
|        |        |  packet may cross scan lines ( begin on one line and end   |
|        |        |  on the next ).                                            |
|        |        |  For the raw packet, the header is followed by             |
|        |        |  the number of color values specified in the header.       |
|        |        |  The color entries themselves are two bytes, three bytes,  |
|        |        |  or four bytes ( for Targa 16, 24, and 32 ), and are       |
|        |        |  broken down as follows:                                   |
|        |        |  The 2 byte entry -                                        |
|        |        |  ARRRRRGG GGGBBBBB, where each letter represents a bit.    |
|        |        |  But, because of the lo-hi storage order, the first byte   |
|        |        |  coming from the file will actually be GGGBBBBB, and the   |
|        |        |  second will be ARRRRRGG. "A" represents an attribute bit. |
|        |        |  The 3 byte entry contains 1 byte each of blue, green,     |
|        |        |  and red.                                                  |
|        |        |  The 4 byte entry contains 1 byte each of blue, green,     |
|        |        |  red, and attribute.  For faster speed (because of the     |
|        |        |  hardware of the Targa board itself), Targa 24 image are   |
|        |        |  sometimes stored as Targa 32 images.                      |
--------------------------------------------------------------------------------

==================================================================================*/
