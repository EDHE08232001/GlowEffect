/*******************************************************************************************************************
 * FILE NAME   :    old_movies.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    old movies core
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 07      Yu Liu          translated DVE3D DSP code
 * 2022 OCT 10      Yu Liu          re-organized code structure
 *
 ********************************************************************************************************************/
#include "old_movies.hpp"
#include "old_movies.cuh"

#include <omp.h>

extern float    param_Fuzzy;
extern bool     button_State[5];

void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);
void filter_gaussian(const int width, const int height, uchar4* src_img, uchar4* dst_img);

void old_movies_image(const char* file_name, const bool is_cuda)
{
    constexpr char* display_winnm = "display window";
    cv::namedWindow(display_winnm, cv::WINDOW_AUTOSIZE);// WINDOW_NORMAL);

    set_sliders();
    initial_defocus(0.f);
    cv::setMouseCallback(display_winnm, mouse_cb);

    cv::Mat src_bgr = cv::imread(file_name, cv::ImreadModes::IMREAD_UNCHANGED);
    if (src_bgr.data == nullptr) {
        printf("file doesn't exist: %s\n", file_name);
        exit(0);
    }

    cv::Mat src_yuv;
    cv::cvtColor(src_bgr, src_yuv, cv::COLOR_BGR2YUV);

    cv::Mat dst_yuv = src_yuv.clone();
    cv::Mat dst_bgr = src_bgr.clone();
 
    int n_threads = omp_get_max_threads();
    omp_set_num_threads(n_threads);
    int key = 0;
    do {
#pragma omp parallel //num_threads(12)
        {
            color_polarizer(src_yuv, dst_yuv);
            if (!is_cuda)
                dynamic_defocus(dst_yuv);
            cv::cvtColor(dst_yuv, dst_bgr, cv::COLOR_YUV2BGR);
        }

        if (is_cuda)
        {
            cv::Mat dst_bgra;
            cv::cvtColor(dst_bgr, dst_bgra, cv::COLOR_BGR2BGRA);

            if (button_State[4])
                filter_gaussian(
                    dst_bgra.cols,
                    dst_bgra.rows,
                    (uchar4*)dst_bgra.data,     // source
                    (uchar4*)dst_bgra.data      // destination as well
                );
            else
                filter_mipmap(
                    dst_bgra.cols,
                    dst_bgra.rows,
                    param_Fuzzy * 200.f,
                    (uchar4*)dst_bgra.data,     // source
                    (uchar4*)dst_bgra.data      // destination as well
                );

            cv::cvtColor(dst_bgra, dst_bgr, cv::COLOR_BGRA2BGR);
        }

        set_buttons(dst_bgr);
        cv::imshow(display_winnm, dst_bgr);

        key = cv::waitKey(30); // 30 milliseconds
    } while (key < 0);// != 27); // 27 = escape

    cv::destroyAllWindows();
}


void old_movies_video(const char* video_nm, const bool is_cuda)
{
    constexpr char* display_winnm = "display";
    cv::namedWindow(display_winnm, cv::WINDOW_AUTOSIZE);

    set_sliders();
    initial_defocus(0.f);
    cv::setMouseCallback(display_winnm, mouse_cb);

    cv::VideoCapture    camera;

    bool ok = camera.open(video_nm, cv::VideoCaptureAPIs::CAP_ANY);

    double cam_val;
    cam_val = camera.get(cv::CAP_PROP_POS_MSEC);
    cam_val = camera.get(cv::CAP_PROP_POS_AVI_RATIO);
    cam_val = camera.get(cv::CAP_PROP_FRAME_WIDTH);         const int img_cols = (int)cam_val;
    cam_val = camera.get(cv::CAP_PROP_FRAME_HEIGHT);        const int img_rows = (int)cam_val;
    cam_val = camera.get(cv::CAP_PROP_FPS);
    cam_val = camera.get(cv::CAP_PROP_FOURCC);
    cam_val = camera.get(cv::CAP_PROP_FRAME_COUNT);         const int n_frame = (int)cam_val;
    cam_val = camera.get(cv::CAP_PROP_FORMAT);
    cam_val = camera.get(cv::CAP_PROP_MODE);
    cam_val = camera.get(cv::CAP_PROP_BRIGHTNESS);
    cam_val = camera.get(cv::CAP_PROP_CONTRAST);
    cam_val = camera.get(cv::CAP_PROP_SATURATION);
    cam_val = camera.get(cv::CAP_PROP_HUE);
    cam_val = camera.get(cv::CAP_PROP_GAIN);
    cam_val = camera.get(cv::CAP_PROP_EXPOSURE);
    cam_val = camera.get(cv::CAP_PROP_SHARPNESS);
    cam_val = camera.get(cv::CAP_PROP_GAMMA);
    cam_val = camera.get(cv::CAP_PROP_BACKEND);

    int frame_cnt = 0;
    while (camera.isOpened())
    {
        cv::Mat vid_cap;

        if (camera.read(vid_cap))
        {
            cv::Mat src_yuv;
            cv::cvtColor(vid_cap, src_yuv, cv::COLOR_BGR2YUV);

            cv::Mat dst_yuv(img_rows, img_cols, CV_8UC3);

            color_polarizer(src_yuv, dst_yuv);
            if (!is_cuda)
                dynamic_defocus(dst_yuv);

            cv::Mat dst_bgr;
            cv::cvtColor(dst_yuv, dst_bgr, cv::COLOR_YUV2BGR);

            if (is_cuda)
            {
                cv::Mat dst_bgra;
                cv::cvtColor(dst_bgr, dst_bgra, cv::COLOR_BGR2BGRA);

                if (button_State[4])
                    filter_gaussian(
                        dst_bgra.cols,
                        dst_bgra.rows,
                        (uchar4*)dst_bgra.data,     // source
                        (uchar4*)dst_bgra.data      // destination as well
                    );
                else
                    filter_mipmap(
                        dst_bgra.cols,
                        dst_bgra.rows,
                        param_Fuzzy * 200.f,
                        (uchar4*)dst_bgra.data,     // source
                        (uchar4*)dst_bgra.data      // destination as well
                    );

                cv::cvtColor(dst_bgra, dst_bgr, cv::COLOR_BGRA2BGR);
            }

            set_buttons(dst_bgr);
            cv::imshow(display_winnm, dst_bgr);
        }

        frame_cnt++;
        if (frame_cnt == n_frame)
            frame_cnt = 0,
            camera.set(cv::CAP_PROP_POS_FRAMES, 0);

        int key = cv::waitKey(1);
        if (key > 0)
            break;
    }

    cv::destroyAllWindows();
}

void old_movies(const int camera_id, const bool is_cuda)
{
    constexpr char* display_winnm = "display";
    cv::namedWindow(display_winnm, cv::WINDOW_AUTOSIZE);

    set_sliders();
    initial_defocus(0.f);
    cv::setMouseCallback(display_winnm, mouse_cb);

    cv::VideoCapture    camera;
    bool ok = camera.open(camera_id, cv::VideoCaptureAPIs::CAP_ANY);

    // normal properties of camera
    double cam_val;
    cam_val = camera.get(cv::CAP_PROP_POS_MSEC);
    cam_val = camera.get(cv::CAP_PROP_POS_AVI_RATIO);
    cam_val = camera.get(cv::CAP_PROP_FRAME_WIDTH);
    const int img_cols = (int)cam_val;
    cam_val = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
    const int img_rows = (int)cam_val;
    cam_val = camera.get(cv::CAP_PROP_FPS);
    cam_val = camera.get(cv::CAP_PROP_FOURCC);
    cam_val = camera.get(cv::CAP_PROP_FRAME_COUNT);
    cam_val = camera.get(cv::CAP_PROP_FORMAT);
    cam_val = camera.get(cv::CAP_PROP_MODE);
    cam_val = camera.get(cv::CAP_PROP_BRIGHTNESS);
    cam_val = camera.get(cv::CAP_PROP_CONTRAST);
    cam_val = camera.get(cv::CAP_PROP_SATURATION);
    cam_val = camera.get(cv::CAP_PROP_HUE);
    cam_val = camera.get(cv::CAP_PROP_GAIN);
    cam_val = camera.get(cv::CAP_PROP_EXPOSURE);
    cam_val = camera.get(cv::CAP_PROP_SHARPNESS);
    cam_val = camera.get(cv::CAP_PROP_GAMMA);
    cam_val = camera.get(cv::CAP_PROP_BACKEND);

    while (camera.isOpened())
    {
        cv::Mat vid_cap;

        if (camera.read(vid_cap))
        {
            cv::Mat src_yuv;
            cv::cvtColor(vid_cap, src_yuv, cv::COLOR_BGR2YUV);

            cv::Mat dst_yuv(img_rows, img_cols, CV_8UC3);

            color_polarizer(src_yuv, dst_yuv);
            if (!is_cuda)
                dynamic_defocus(dst_yuv);

            cv::Mat dst_bgr;
            cv::cvtColor(dst_yuv, dst_bgr, cv::COLOR_YUV2BGR);

            if(is_cuda)
            {
                cv::Mat dst_bgra;
                cv::cvtColor(dst_bgr, dst_bgra, cv::COLOR_BGR2BGRA);

                if (button_State[4])
                    filter_gaussian(
                        dst_bgra.cols,
                        dst_bgra.rows,
                        (uchar4*)dst_bgra.data,     // source
                        (uchar4*)dst_bgra.data      // destination as well
                    );
                else
                    filter_mipmap(
                        dst_bgra.cols,
                        dst_bgra.rows,
                        param_Fuzzy * 200.f,
                        (uchar4*)dst_bgra.data,     // source
                        (uchar4*)dst_bgra.data      // destination as well
                    );

                cv::cvtColor(dst_bgra, dst_bgr, cv::COLOR_BGRA2BGR);
            }

            set_buttons(dst_bgr);
            cv::imshow(display_winnm, dst_bgr);
        }

        int key = cv::waitKey(30);
        if (key > 0)
            break;
    }

    cv::destroyAllWindows();
}


