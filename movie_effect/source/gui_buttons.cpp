/*******************************************************************************************************************
 * FILE NAME   :    gui_buttons.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    buttons setting made by mouse call back
 *                  button letters overlaid on top of video image, mouse event detects location and switch modes
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 08      Yu Liu          Creation
 *
 ********************************************************************************************************************/
#include "old_movies.hpp"

int param_Mode = 0;

void mouse_cb(int event, int x, int y, int flag, void* user_data)
{
    /*
    cv::EVENT_FLAG_ALTKEY;
    cv::EVENT_FLAG_CTRLKEY;
    cv::EVENT_FLAG_LBUTTON;
    cv::EVENT_FLAG_RBUTTON;
    cv::EVENT_FLAG_MBUTTON;
    cv::EVENT_FLAG_SHIFTKEY;

    if(flag== (cv::EVENT_FLAG_ALTKEY + cv::EVENT_FLAG_CTRLKEY + ...))
        ...
    */
    
    if (event == cv::EVENT_MOUSEMOVE) {}
    else if (event == cv::EVENT_LBUTTONDOWN) {}
    else if (event == cv::EVENT_RBUTTONDOWN) {}
    else if (event == cv::EVENT_MBUTTONDOWN) {}
    else if (event == cv::EVENT_LBUTTONUP) {}
    else if (event == cv::EVENT_RBUTTONUP) {}
    else if (event == cv::EVENT_MBUTTONUP) {}
    else if (event == cv::EVENT_LBUTTONDBLCLK) {
        if (x > ModePosX && x < ModePosX + 100 && y < ModePosY + 5 && y > ModePosY - 35)
        {
            param_Mode++;
            param_Mode %= 2;
        }
    }
    else if (event == cv::EVENT_RBUTTONDBLCLK) {}
    else if (event == cv::EVENT_MBUTTONDBLCLK) {}
    else if (event == cv::EVENT_MOUSEWHEEL) {}
    else if (event == cv::EVENT_MOUSEHWHEEL) {}
}

void set_buttons(cv::Mat& image)
{
    int bg_xsize = 100, bg_ysize = 40;
    cv::Mat  bg_color(bg_ysize, bg_xsize, CV_8UC3, cv::Scalar(128, 0, 0));
    cv::Rect bg_roi(ModePosX, ModePosY - bg_ysize, bg_xsize, bg_ysize);

    bg_color.copyTo(image(bg_roi));

    cv::putText(
        image,
        param_Mode == 0 ? "Sepia" : "Color",
        cv::Point(ModePosX + 5, ModePosY - 10),
        cv::FONT_HERSHEY_DUPLEX,
        1.0,
        CV_RGB(200, 200, 0),
        2,
        cv::LineTypes::LINE_AA
    );

    /*
    cv::circle(
        image,
        cv::Point(100, 100),
        20,
        cv::Scalar(200, 200, 0),
        2,
        cv::LineTypes::LINE_AA  // LINE_4: 4-connected; LINE_8: 8-connected
    );

    cv::rectangle(
        image,
        cv::Rect(80, 80, 40, 40),
        cv::Scalar(0, 0, 200),
        2,
        cv::LineTypes::FILLED
    );

    cv::ellipse(
        image,
        cv::RotatedRect(cv::Point2f(100, 200), cv::Size2f(80.f, 40.f), 45.f),
        cv::Scalar(0, 200, 200),
        2,
        cv::LineTypes::LINE_AA
    );
    */
}