#pragma once

#define NUM_ULTRIX_WARP_CHANNELS 4
#define ACTIVE_PIXELS_2160p     (1920*2)
#define WARP_ALPHA_IDENTITY     255
#define WARP_LIGHT_IDENTITY     255
#define WARP_LIGHT_MAX          0xff
#define CUDART_PI_F             3.14159265359f
#define ALPH_MAX    255
#define CHRM_MID    128
#define Y_MIN       0
#define Y_MAX       255
#define CR_MIN      0
#define CR_MAX      255
#define CB_MIN      0
#define CB_MAX      255
#define PI          3.14159265359f


#define MOVIE_WARPPIX_ITER_MIN         (720)
#define MOVIE_WARPPIX_ITER_MAX         ACTIVE_PIXELS_2160p
#define MOVIE_WARPPIX_ITER_MULT        (16)
#define MAX_WARP_PARAMS     15      // Max number of parameters that a warp can
#define MOVIE_MAX_RANDOM    2048 //Ideally this should be 1, but we get a weird
#define MOVIE_MAX_PARAM  (MAX_WARP_PARAMS+1)
#define MOVIE_SOFTNESS_MAX  1024
#define MOVIE_A             16807u                        /* MULTIPLIER VALUE    */
#define MOVIE_JUMP_MAX      1024

// Quotient to reduce X defocus effect in "ultra-wide defocus"
#define MOVIE_DEFOCUS_RSHIFT_BITS  4

#define MAX_RANGE_BITS 12                // How many bits used for MAX_RANGE?
#define MAX_RANGE ((1<<MAX_RANGE_BITS)-1)    // Max value for pots and pot like objects
#define WARP_CHROMA_BITS 10
#define WARP_SOLARIZE_LUT_BITS   9
#define CHROMA_DC_BIAS          (0x200)
#define LUMA_DC_BIAS            (0x40)

// Warp Buffer Format
#define SUBPIX_BITS     (4)
#define X_INT_BITS      (11)
#define Y_INT_BITS      (11)
#define  MOVIE_TRIG_FRAB        14
#define  MOVIE_JUMP_SHIFT       12
#define  MOVIE_VLINE_MASK       0xF
#define  MOVIE_VLINE_SUB        8
#define  MOVIE_VLINE_INERTIA    8
// This warp was ported from sd, 480x720. Instead of redoing all the math
// we scale the output to fit the new formats
#define SD_Y_SIZE 480
#define SD_X_SIZE 720
#define RES_X_MAX 1920
#define RES_Y_MAX 1080
#define SHIFT_TO_LIGHTING 12
#define X_PIXEL_BITS 11
#define MAX_BITS_COORD 16   //x, y, and z are 11 bits plus 5 bit subpixel

// number of fields before random tables are refreshed
#ifdef WARP_SIMULATOR
#define  MOVIE_RANDOM_REFRESH    2
#else
#define  MOVIE_RANDOM_REFRESH    4  // was 4
#endif

//A linear congruential generator (pseudo random number)
// Yes this will evntually overflow, but we want it to.
#define GET_RAND    scratch->rand_Q32 * RNG_MULTIPLIER + RNG_OFFSET
#define RNG_SEED   1
#define RNG_MULTIPLIER 1103515245
#define RNG_OFFSET   12345


typedef struct {
    // warp params needed in kernel:
    float movie_radius;
    float noise;
    float film_jump;
    float presets;
    int16_t movieModel;
    float softness;
    float rand_scratch;

    float cols;
    float rows;

    // Color Tables:
    int16_t colorType;          // e_Movie_color_type colorType; // color type
    float hue, sat, lum;        // Panel input
    float red, blue;            // Panel input
    float luma_contrast;        // Panel input

    uint32_t rand_Q32;

    uint16_t scratch_max_width; // max width of the scratch
    uint32_t scratch_pos;       // position _pack2(y,x)
    uint16_t scratch_length;    // length of the scrath
    short  scratch_cosA;        // cosine of the angle of the scratch (w/ x-axis)
    short  scratch_sinA;        // sine of the angle of the scratch

    // Vertical Scratches
    int vert_pos1;     // current position of the 1st vertical scratch
    int vert_pos2;     // current position of the 2nd vertical scratch
    int old_vert_pos1; // "center value" of the 1st vertical scratch's position
    int old_vert_pos2; // "center value" of the 2nd vertical scratch's position
    uint32_t show_scratch1; // TRUE: show the 1st vertical scratch; FALSE: hide it
    uint32_t show_scratch2; // same for 2nd vertical scratch
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
    float ratio480iToCurrentX;
    float ratio480iToCurrentY;
    int32_t light_flash;
    int32_t line_error_max;
    int count;
    uint32_t scratch_max_light;
    uint16_t vert_scratch_light;
} s_movie_vars;

typedef struct {
    int16_t     vid_mode = 0;
    bool        isOddField = false;
    int         cols = 1920, rows = 1080;
} s_frame_info;
typedef struct {
    short* y, * u, * v;
}texObject_t;
typedef struct {
    int x, y, z, w;
} int4;
typedef struct {
    float x, y, z, w;
} float4;


