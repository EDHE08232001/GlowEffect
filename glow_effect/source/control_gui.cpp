/*******************************************************************************************************************
 * FILE NAME   :    control_gui.cpp
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
#include "./wx_gui.h"
#include <thread>

 // Global Variables
int button_id = 0; // Currently selected button ID.
int param_KeyScale = 600;  // Key scale parameter controlled by slider.
int param_KeyLevel = 96;   // Key level parameter controlled by slider.
int default_scale = 10;    // Default scale parameter controlled by slider.

/**
* wxwidgets gui
*/
void set_control() {
	// Create a new wxWidgets application instance
	wxApp::SetInstance(new MyApp());

	// Initialize the wxWidgets application
	if (!wxEntryStart(0, nullptr)) {
		std::cerr << "Failed to initialize wxWidgets application." << std::endl;
		return;
	}

	// Call the OnInit function of the wxWidgets application
	if (!wxTheApp->OnInit()) {
		std::cerr << "wxWidgets application initialization failed in OnInit." << std::endl;
		wxEntryCleanup();
		return;
	}

	// Start the main event loop for the GUI
	wxTheApp->OnRun();

	// Call the OnExit function to clean up
	wxTheApp->OnExit();

	// Cleanup wxWidgets application resources
	wxEntryCleanup();
}