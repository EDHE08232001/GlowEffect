#include "all_common.h"
#include "./all_main.h"
#include "./wx_gui.h"
#include "./glow_effect.hpp"

// ** Application Entry Point **
/**
 * @brief Initializes the application and creates the main frame.
 *
 * This method is automatically called when the wxWidgets application starts.
 * @return true if initialization is successful; false otherwise.
 */
bool MyApp::OnInit() {
	MainFrame* frame = new MainFrame("Glow Effect GUI");
	frame->Show(true); // Display the main frame
	return true;
}

// ** Main Frame Constructor **
/**
 * @brief Constructs the MainFrame and initializes the UI components.
 *
 * @param title The title displayed on the window.
 */
MainFrame::MainFrame(const wxString& title)
	: wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(600, 500)),
	selectedButtonIndex(-1) { // Initialize selectedButtonIndex to -1 (no button selected)

  // Set the minimum and maximum window sizes
	SetMinSize(wxSize(600, 500));
	SetMaxSize(wxSize(800, 600));

	// ** Main Panel with Layout **
	mainPanel = new wxPanel(this, wxID_ANY);
	wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);
	mainPanel->SetSizer(mainSizer);

	// ** Button Panel **
	buttonPanel = new wxPanel(mainPanel, wxID_ANY);
	SetupButtonPanel(); // Setup and style the buttons
	mainSizer->Add(buttonPanel, 0, wxEXPAND | wxALL, 10);

	// ** Sliders Section **
	SetupSliders(); // Setup and style the sliders

	// Center the window on the screen
	this->Centre();
}

// ** Setup Button Panel **
/**
 * @brief Initializes and styles the button panel.
 *
 * Creates five buttons, assigns default styles, and binds click events to a common handler.
 */
void MainFrame::SetupButtonPanel() {
	wxBoxSizer* buttonSizer = new wxBoxSizer(wxHORIZONTAL);
	const wxString buttonLabels[5] = { "Blow", "Star", "NA", "NA", "NA" };

	for (int i = 0; i < 5; ++i) {
		buttons[i] = new wxButton(buttonPanel, wxID_ANY, buttonLabels[i], wxDefaultPosition, wxSize(100, 40));

		// Default appearance
		buttons[i]->SetBackgroundColour(wxColour(220, 220, 240)); // Inactive background color
		buttons[i]->SetForegroundColour(wxColour(120, 120, 120)); // Inactive text color

		// Bind click event to the OnButtonClicked handler
		buttons[i]->Bind(wxEVT_BUTTON, &MainFrame::OnButtonClicked, this);

		buttonSizer->Add(buttons[i], 1, wxEXPAND | wxALL, 5);
	}

	buttonPanel->SetSizer(buttonSizer);
}

// ** Update Button Appearance **
/**
 * @brief Updates the appearance of the buttons to reflect the selected state.
 *
 * Highlights the selected button and resets the appearance of others.
 *
 * @param selectedIndex The index of the button to highlight. Pass -1 to reset all buttons.
 */
void MainFrame::UpdateButtonAppearance(int selectedIndex) {
	for (int i = 0; i < 5; ++i) {
		if (i == selectedIndex) {
			// Highlight the selected button
			buttons[i]->SetBackgroundColour(wxColour(255, 223, 186)); // Active background color
			buttons[i]->SetForegroundColour(wxColour(40, 40, 40));   // Active text color
		}
		else {
			// Reset appearance for other buttons
			buttons[i]->SetBackgroundColour(wxColour(220, 220, 240)); // Default background color
			buttons[i]->SetForegroundColour(wxColour(120, 120, 120)); // Default text color
		}
	}

	// Refresh the button panel to apply the changes
	buttonPanel->Refresh();
	buttonPanel->Update();
}

// ** Setup Sliders **
/**
 * @brief Initializes and styles the sliders and their associated labels.
 *
 * Adds three sliders ("Key Level", "Key Scale", "Default Scale") and their min/max labels to the main layout.
 */
void MainFrame::SetupSliders() {
	wxBoxSizer* mainSizer = dynamic_cast<wxBoxSizer*>(mainPanel->GetSizer());

	// ** Key Level Slider **
	keyLevelLabel = new wxStaticText(mainPanel, wxID_ANY, "Key Level:");
	keyLevelSlider = new wxSlider(mainPanel, wxID_ANY, 96, 0, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL);
	keyLevelMinLabel = new wxStaticText(mainPanel, wxID_ANY, "0");
	keyLevelMaxLabel = new wxStaticText(mainPanel, wxID_ANY, "255");

	keyLevelSlider->Bind(wxEVT_SLIDER, &MainFrame::OnKeyLevelChange, this); // Bind slider event

	mainSizer->Add(keyLevelLabel, 0, wxLEFT | wxTOP, 10);
	wxBoxSizer* keyLevelSizer = new wxBoxSizer(wxHORIZONTAL);
	keyLevelSizer->Add(keyLevelMinLabel, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, 10);
	keyLevelSizer->Add(keyLevelSlider, 1, wxEXPAND | wxALL, 5);
	keyLevelSizer->Add(keyLevelMaxLabel, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
	mainSizer->Add(keyLevelSizer, 0, wxEXPAND | wxALL, 5);

	// ** Key Scale Slider **
	keyScaleLabel = new wxStaticText(mainPanel, wxID_ANY, "Key Scale:");
	keyScaleSlider = new wxSlider(mainPanel, wxID_ANY, 600, 0, 1000, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL);
	keyScaleMinLabel = new wxStaticText(mainPanel, wxID_ANY, "0");
	keyScaleMaxLabel = new wxStaticText(mainPanel, wxID_ANY, "1000");

	keyScaleSlider->Bind(wxEVT_SLIDER, &MainFrame::OnKeyScaleChange, this); // Bind slider event

	mainSizer->Add(keyScaleLabel, 0, wxLEFT | wxTOP, 10);
	wxBoxSizer* keyScaleSizer = new wxBoxSizer(wxHORIZONTAL);
	keyScaleSizer->Add(keyScaleMinLabel, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, 10);
	keyScaleSizer->Add(keyScaleSlider, 1, wxEXPAND | wxALL, 5);
	keyScaleSizer->Add(keyScaleMaxLabel, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
	mainSizer->Add(keyScaleSizer, 0, wxEXPAND | wxALL, 5);

	// ** Default Scale Slider **
	defaultScaleLabel = new wxStaticText(mainPanel, wxID_ANY, "Default Scale:");
	defaultScaleSlider = new wxSlider(mainPanel, wxID_ANY, 10, 0, 100, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL);
	defaultScaleMinLabel = new wxStaticText(mainPanel, wxID_ANY, "0");
	defaultScaleMaxLabel = new wxStaticText(mainPanel, wxID_ANY, "100");

	defaultScaleSlider->Bind(wxEVT_SLIDER, &MainFrame::OnDefaultScaleChange, this); // Bind slider event

	mainSizer->Add(defaultScaleLabel, 0, wxLEFT | wxTOP, 10);
	wxBoxSizer* defaultScaleSizer = new wxBoxSizer(wxHORIZONTAL);
	defaultScaleSizer->Add(defaultScaleMinLabel, 0, wxALIGN_CENTER_VERTICAL | wxLEFT, 10);
	defaultScaleSizer->Add(defaultScaleSlider, 1, wxEXPAND | wxALL, 5);
	defaultScaleSizer->Add(defaultScaleMaxLabel, 0, wxALIGN_CENTER_VERTICAL | wxRIGHT, 10);
	mainSizer->Add(defaultScaleSizer, 0, wxEXPAND | wxALL, 5);
}

/**
* Button Event Handler
* 
* So far in the development process, these buttons don't need any functionalities
*/
void MainFrame::OnButtonClicked(wxCommandEvent& event) {
	for (int i = 0; i < 5; ++i) {
		if (event.GetEventObject() == buttons[i]) {
			button_id = i; // Update the selected button index atomically
			UpdateButtonAppearance(i); // Update button appearance
			break;
		}
	}
}

// ** Slider Handlers **
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
* Callback function for KeyLevelSlider
*/
void MainFrame::OnKeyLevelChange(wxCommandEvent& event) {
	// 1) Read the current slider value
	int value = keyLevelSlider->GetValue();

	// 2) Log something
	std::clog << "Key level updated to: " << value << std::endl;

	// 3) update
	bar_key_level_cb(value);
}

/**
* Callback function for keyScaleSlider
*/
void MainFrame::OnKeyScaleChange(wxCommandEvent& event) {
	// 1) Read the current slider value
	int value = keyScaleSlider->GetValue();

	// 2) Log something
	std::clog << "Key Scale updated to: " << param_KeyScale << std::endl;

	// 3) update
	bar_key_scale_cb(value);
}

/**
* Callback function for defaultScaleSlider
*/
void MainFrame::OnDefaultScaleChange(wxCommandEvent& event) {
	// 1) Read the current slider value
	int value = defaultScaleSlider->GetValue();

	// 2) Log something
	std::clog << "Default scale updated to: " << default_scale << std::endl;

	// 3) update
	bar_default_scale_cb(value);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////