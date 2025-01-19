/*******************************************************************************************************************
 * FILE NAME   :    control_gui_beautiful.cpp
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    General GUI setup using OpenCV with improved styling:
 *                  - Soft gradient background
 *                  - Slight shadow on button panel
 *                  - Rounded buttons with pastel flat design
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2025/JAN/17      Edward He       Updated GUI
 ********************************************************************************************************************/

#include "all_common.h"
#include "opencv2/opencv.hpp"
#include "glow_effect.hpp"

 // Global Variables
int button_id = 0; // Currently selected button ID.
int param_KeyScale = 600;  // Key scale parameter controlled by slider.
int param_KeyLevel = 96;   // Key level parameter controlled by slider.
int default_scale = 10;    // Default scale parameter controlled by slider.

static const std::string modelName[] = { "blow", "star", "na", "na", "na" }; // Button labels.
static const char* control_winnm = "control";  // Window name.
static const int button_container_cols = 500;  // Width of the button panel.
static const int button_container_rows = 50;   // Height of the button panel.
static const int button_num = 5;              // Number of buttons.
static const int button_width = button_container_cols / button_num; // Width of each button.
static const int button_height = button_container_rows;             // Height of each button.
static const int button_margin = 10;          // Margin around each button.

static cv::Mat button_container; // Container for the button panel.

// Updated Color Constants for Flat Design
// Slightly tweaked from your original
static const cv::Scalar BG_COLOR(245, 245, 250);           // Light background
static const cv::Scalar ACTIVE_COLOR(255, 223, 186);       // Warm pastel orange for active buttons
static const cv::Scalar INACTIVE_COLOR(220, 220, 240);     // Muted pastel blue for inactive buttons
static const cv::Scalar TEXT_COLOR_ACTIVE(40, 40, 40);     // Dark gray text for active buttons
static const cv::Scalar TEXT_COLOR_INACTIVE(120, 120, 120);// Medium gray text for inactive buttons

// Additional colors for gradient background
static const cv::Scalar TOP_GRADIENT_COLOR(240, 240, 245);     // near white
static const cv::Scalar BOTTOM_GRADIENT_COLOR(220, 230, 245);  // slightly bluish tint

/**
 * @brief Creates a vertical gradient in a Mat from top to bottom.
 *
 * @param img Reference to the Mat where gradient will be drawn.
 * @param topColor The color at the top of the gradient.
 * @param bottomColor The color at the bottom of the gradient.
 */
void createGradientBackground(cv::Mat& img, const cv::Scalar& topColor, const cv::Scalar& bottomColor)
{
    for (int y = 0; y < img.rows; y++) {
        double alpha = static_cast<double>(y) / img.rows;
        cv::Scalar rowColor = (1.0 - alpha) * topColor + alpha * bottomColor;
        for (int x = 0; x < img.cols; x++) {
            img.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(rowColor[0]);
            img.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(rowColor[1]);
            img.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(rowColor[2]);
        }
    }
}

/**
 * @brief Draws a rounded rectangle with a soft fill on a Mat object.
 *
 * @param img The target Mat object.
 * @param rect The rectangle region to draw.
 * @param fill_color The fill color of the rectangle.
 * @param radius The corner radius of the rectangle.
 */
void drawRoundedRectangle(cv::Mat& img, cv::Rect rect, const cv::Scalar& fill_color, int radius = 10) {
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    cv::rectangle(mask, rect, fill_color, -1, cv::LINE_AA);
    cv::GaussianBlur(mask, mask, cv::Size(radius * 2 + 1, radius * 2 + 1), radius);
    mask.copyTo(img, mask);
}

/**
 * @brief Updates the appearance of a button in the GUI.
 *
 * Each button is a rounded color block with clear text.
 *
 * @param id Index of the button to update.
 * @param on_off Boolean indicating the button's state (true for active, false for inactive).
 */
static void button_text(const int id, const bool on_off) {
    // Define the button region of interest (ROI).
    cv::Rect button_roi(
        id * button_width + button_margin / 2,
        button_margin / 2,
        button_width - button_margin,
        button_height - button_margin
    );

    // Draw a rounded rectangle for the button.
    drawRoundedRectangle(button_container, button_roi, on_off ? ACTIVE_COLOR : INACTIVE_COLOR, 15);

    // Put text on the button.
    cv::putText(
        button_container,
        modelName[id],
        cv::Point(button_roi.x + button_width / 6, button_roi.y + button_height / 1.8),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        on_off ? TEXT_COLOR_ACTIVE : TEXT_COLOR_INACTIVE,
        2,
        cv::LINE_AA
    );

    cv::imshow(control_winnm, button_container);
}

static bool button_changed = false;

/**
 * @brief Callback function for mouse events on the button panel.
 *
 * Updates the selected button when a button is clicked.
 *
 * @param event Mouse event type.
 * @param x X-coordinate of the mouse event.
 * @param y Y-coordinate of the mouse event.
 * @param flag Additional event flags.
 * @param user_data Pointer to user data (unused).
 */
static void button_cb(int event, int x, int y, int flag, void* user_data) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        for (int k = 0; k < button_num; k++) {
            if (x >= k * button_width && x < (k + 1) * button_width) {
                if (button_id != k) {
                    button_text(button_id, false); // Deactivate the previous button.
                    button_id = k;                 // Update selected button ID.
                    button_text(button_id, true);  // Activate the new button.
                    button_changed = true;
                }
                break;
            }
        }
    }
}

/**
 * @brief Callback function for the "Key Level" slider.
 */
static void bar_key_level_cb(int pos, void* user_data) {
    param_KeyLevel = pos;
    std::cout << "Key Level updated to: " << param_KeyLevel << std::endl;
}

/**
 * @brief Callback function for the "Key Scale" slider.
 */
static void bar_key_scale_cb(int pos, void* user_data) {
    param_KeyScale = pos;
    std::cout << "Key Scale updated to: " << param_KeyScale << std::endl;
}

/**
 * @brief Callback function for the "Default Scale" slider.
 */
static void bar_default_scale_cb(int pos, void* user_data) {
    default_scale = pos;
    std::cout << "Default Scale updated to: " << default_scale << std::endl;
}

/**
 * @brief Initializes the GUI control panel with a gradient background, sliders, and a button panel.
 */
void set_control(void) {
    // Create the control window.
    cv::namedWindow(control_winnm, cv::WindowFlags::WINDOW_NORMAL);
    cv::resizeWindow(control_winnm, button_container_cols, button_container_rows + 120);
    cv::setMouseCallback(control_winnm, button_cb);

    // Create the background with a gentle gradient.
    cv::Mat full_bg(button_container_rows + 120, button_container_cols, CV_8UC3);
    createGradientBackground(full_bg, TOP_GRADIENT_COLOR, BOTTOM_GRADIENT_COLOR);

    // A subtle "shadow" rectangle to house the button container.
    // This can give an elevated look to the button panel.
    cv::Rect shadowROI(5, 5, button_container_cols - 10, button_container_rows);
    drawRoundedRectangle(full_bg, shadowROI, cv::Scalar(200, 200, 200), 20);

    // Create the button_container as a region on top of the gradient for clarity.
    button_container = full_bg(cv::Rect(5, 5, button_container_cols - 10, button_container_rows)).clone();
    button_text(0, true); // Activate the first button by default.
    for (int i = 1; i < button_num; ++i) {
        button_text(i, false); // Deactivate the other buttons.
    }
    // Show it once for initialization.
    cv::imshow(control_winnm, full_bg);

    // Create Trackbars
    cv::createTrackbar("Key Level", control_winnm, &param_KeyLevel, 255, bar_key_level_cb);
    cv::createTrackbar("Key Scale", control_winnm, &param_KeyScale, 1000, bar_key_scale_cb);
    cv::createTrackbar("Default Scale", control_winnm, &default_scale, 100, bar_default_scale_cb);

    // Move sliders down for better spacing and set initial positions.
    cv::setTrackbarPos("Key Level", control_winnm, param_KeyLevel);
    cv::setTrackbarPos("Key Scale", control_winnm, param_KeyScale);
    cv::setTrackbarPos("Default Scale", control_winnm, default_scale);

    std::cout << "Control panel successfully initialized with a stylish and clean look." << std::endl;
}
