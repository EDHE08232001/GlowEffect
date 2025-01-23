/*******************************************************************************************************************
 * FILE NAME   :    gui_sliders.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    slider settings
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 07      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#include "old_movies.hpp"

float       param_Cangle;
float       param_Csat;
float       param_Uofs;
float       param_Vofs;
float       param_Yslope;
float       param_Yofs;
float       param_Fuzzy;
bool        button_State[5] = { false,false,false,false,false };

static const char* CTRL_WINNM = "control panel";

const char* bar_c_angle_winnm = "angle";
const int   bar_c_angle_range = 1024, bar_c_angle_center = bar_c_angle_range >> 1;
static int  bar_c_angle_value;
static void bar_c_angle_cb(int pos, void* user_data)
{
    param_Cangle = F_PI4 * (pos - bar_c_angle_center) / (float)bar_c_angle_center;
}

const char* bar_c_sat_winnm = "c_sat";
const int   bar_c_sat_range = 1024;
static int  bar_c_sat_value;
static void bar_c_sat_cb(int pos, void* user_data)
{
    param_Csat = float(pos) / bar_c_sat_range;
}

const char* bar_u_ofs_winnm = "u_ofs";
const int   bar_u_ofs_range = 1024, bar_u_ofs_center = bar_u_ofs_range >> 1;
static int  bar_u_ofs_value;
static void bar_u_ofs_cb(int pos, void* user_data)
{
    param_Uofs = (float)(pos - bar_u_ofs_center) / (float)bar_u_ofs_center * 0.1f;
}

const char* bar_v_ofs_winnm = "v_ofs";
const int   bar_v_ofs_range = 1024, bar_v_ofs_center = bar_v_ofs_range >> 1;
static int  bar_v_ofs_value;
static void bar_v_ofs_cb(int pos, void* user_data)
{
    param_Vofs = (float)(pos - bar_v_ofs_center) / (float)bar_v_ofs_center * 0.1f;
}

static int  bar_y_min_value;
static int  bar_y_max_value;

const char* bar_y_min_winnm = "y_min";
const int   bar_y_min_range = 255;
static void bar_y_min_cb(int pos, void* user_data)
{
    bar_y_min_value = pos < bar_y_max_value ? pos : bar_y_max_value - 1;
    param_Yslope = (bar_y_max_value - bar_y_min_value) / 255.f;
    param_Yofs = (float)bar_y_min_value;

    cv::setTrackbarPos(bar_y_min_winnm, CTRL_WINNM, bar_y_min_value);
}

const char* bar_y_max_winnm = "y_max";
const int   bar_y_max_range = 255;
static void bar_y_max_cb(int pos, void* user_data)
{
    bar_y_max_value = pos > bar_y_min_value ? pos : bar_y_min_value + 1;
    param_Yslope = (bar_y_max_value - bar_y_min_value) / 255.f;
    param_Yofs = (float)bar_y_min_value;

    cv::setTrackbarPos(bar_y_max_winnm, CTRL_WINNM, bar_y_max_value);
}

const char* bar_fuzzy_winnm = "fuzzy";
const int   bar_fuzzy_range = 255;
int         bar_fuzzy_value;
static void bar_fuzzy_cb(int pos, void* user_data)
{
    bar_fuzzy_value = pos;

    // normalize: 0.f ~ 1.f
    param_Fuzzy = (float)pos / (float)bar_fuzzy_range;
    //
    initial_defocus(param_Fuzzy * 25.f);

    cv::setTrackbarPos(bar_fuzzy_winnm, CTRL_WINNM, bar_fuzzy_value);
}

static const int BTTN_WIN_H = 100, BTTN_WIN_W = 500;
static const int BUTTON_H = 100, BUTTON_W = 100, BUTTON_M = 5, BUTTON_2M = 10;
static cv::Mat ButtonImage;
static void bar_button_image(const int id)
{
    const char* button_title[5] = {
        " Bilinear", " LayerOp", "readMode", " Optimal", "  Filter"
    };
    const char* button_state[5][2] = {
        {"  Off", "  On"},{"  Off", "  On"},{"Normal", "Element"},{"  Off", "  On"},{"Mipmap", " Gauss"}
    };
    const cv::Scalar bg_color[5] = {
        cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(0, 0, 128),
        cv::Scalar(0, 128, 128), cv::Scalar(128, 0, 128)
    };

    cv::Mat  word_arr(BUTTON_H - BUTTON_2M, BUTTON_W - BUTTON_2M, CV_8UC3, bg_color[id]);
    cv::Rect word_roi(BUTTON_W * id + BUTTON_M, BUTTON_M, BUTTON_W - BUTTON_2M, BUTTON_H - BUTTON_2M);

    // button title
    cv::putText(
        word_arr,
        button_title[id],
        cv::Point(0, 30),
        cv::FONT_HERSHEY_DUPLEX,
        0.6,
        cv::Scalar(200,200,200),
        1,
        cv::LINE_AA
    );

    // button state
    cv::putText(
        word_arr,
        button_state[id][button_State[id]],
        cv::Point(10, 70),
        cv::FONT_HERSHEY_DUPLEX,
        0.6,
        cv::Scalar(200, 200, 200),
        1,
        cv::LINE_AA
    );

    word_arr.copyTo(ButtonImage(word_roi));
}

static void bar_mouse_cb(int event, int x, int y, int flag, void* user_data)
{
    if (event == cv::EVENT_MOUSEMOVE) {}
    else if (event == cv::EVENT_LBUTTONDOWN) {
        int key = x / BUTTON_W;
        button_State[key] = !button_State[key];
        bar_button_image(key);
    }
    else if (event == cv::EVENT_RBUTTONDOWN) {}
    else if (event == cv::EVENT_MBUTTONDOWN) {}
    else if (event == cv::EVENT_LBUTTONUP) {}
    else if (event == cv::EVENT_RBUTTONUP) {}
    else if (event == cv::EVENT_MBUTTONUP) {}
    else if (event == cv::EVENT_LBUTTONDBLCLK) {}
    else if (event == cv::EVENT_RBUTTONDBLCLK) {}
    else if (event == cv::EVENT_MBUTTONDBLCLK) {}
    else if (event == cv::EVENT_MOUSEWHEEL) {}
    else if (event == cv::EVENT_MOUSEHWHEEL) {}

    cv::imshow(CTRL_WINNM, ButtonImage);
}

void set_sliders(void)
{
    // cv::WINDOW_NORMAL;cv::WINDOW_FREERATIO; cv::WINDOW_FULLSCREEN; cv::WINDOW_KEEPRATIO
    cv::namedWindow(CTRL_WINNM, cv::WINDOW_AUTOSIZE);// cv::WINDOW_NORMAL); // 
    //cv::resizeWindow(CTRL_WINNM, 500, 400);

    // creation of sliders
    cv::createTrackbar(bar_c_angle_winnm, CTRL_WINNM, nullptr, bar_c_angle_range, bar_c_angle_cb);
    cv::createTrackbar(bar_c_sat_winnm, CTRL_WINNM, nullptr, bar_c_sat_range, bar_c_sat_cb);
    cv::createTrackbar(bar_u_ofs_winnm, CTRL_WINNM, nullptr, bar_u_ofs_range, bar_u_ofs_cb);
    cv::createTrackbar(bar_v_ofs_winnm, CTRL_WINNM, nullptr, bar_v_ofs_range, bar_v_ofs_cb);
    cv::createTrackbar(bar_y_min_winnm, CTRL_WINNM, nullptr, bar_y_min_range, bar_y_min_cb);
    cv::createTrackbar(bar_y_max_winnm, CTRL_WINNM, nullptr, bar_y_max_range, bar_y_max_cb);
    cv::createTrackbar(bar_fuzzy_winnm, CTRL_WINNM, nullptr, bar_fuzzy_range, bar_fuzzy_cb);

    // default settings
    bar_c_angle_value = bar_c_angle_range >> 1;
    bar_c_sat_value = bar_c_sat_range >> 1;
    bar_u_ofs_value = bar_u_ofs_range >> 1;
    bar_v_ofs_value = bar_v_ofs_range >> 1;
    bar_y_min_value = 0;
    bar_y_max_value = bar_y_max_range;
    bar_fuzzy_value = 0;

    param_Cangle = 0.f;
    param_Csat = 0.5f;
    param_Uofs = 0.f;
    param_Vofs = 0.f;
    param_Yslope = (bar_y_max_value - bar_y_min_value) / 255.f;
    param_Yofs = (float)bar_y_min_value;
    param_Fuzzy = 0.f;

    cv::setTrackbarPos(bar_c_angle_winnm, CTRL_WINNM, bar_c_angle_value);
    cv::setTrackbarPos(bar_c_sat_winnm, CTRL_WINNM, bar_c_sat_value);
    cv::setTrackbarPos(bar_u_ofs_winnm, CTRL_WINNM, bar_u_ofs_value);
    cv::setTrackbarPos(bar_v_ofs_winnm, CTRL_WINNM, bar_v_ofs_value);
    cv::setTrackbarPos(bar_y_min_winnm, CTRL_WINNM, bar_y_min_value);
    cv::setTrackbarPos(bar_y_max_winnm, CTRL_WINNM, bar_y_max_value);
    cv::setTrackbarPos(bar_fuzzy_winnm, CTRL_WINNM, bar_fuzzy_value);

    ButtonImage = cv::Mat(BTTN_WIN_H, BTTN_WIN_W, CV_8UC3, cv::Scalar(80,80,80));
    cv::setMouseCallback(CTRL_WINNM, bar_mouse_cb);

    bar_button_image(0);
    bar_button_image(1);
    bar_button_image(2);
    bar_button_image(3);
    bar_button_image(4);
    cv::imshow(CTRL_WINNM, ButtonImage);
}
