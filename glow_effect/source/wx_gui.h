#pragma once

#ifndef WX_GUI_H
#define WX_GUI_H

#include <wx/wx.h>
#include <wx/slider.h>
#include <wx/panel.h>
#include <wx/button.h>

/**
 * @brief The main application class for initializing the wxWidgets application.
 *
 * This class inherits from wxApp and serves as the entry point of the program.
 */
class MyApp : public wxApp {
public:
	/**
	 * @brief Initializes the application and creates the main frame.
	 *
	 * @return true if initialization is successful, false otherwise.
	 */
	virtual bool OnInit();
};

/**
 * @brief The main frame class representing the primary GUI window.
 *
 * This class inherits from wxFrame and contains the main user interface elements.
 */
class MainFrame : public wxFrame {
public:
	/**
	 * @brief Constructs the MainFrame window and initializes UI components.
	 *
	 * @param title The title displayed on the window.
	 */
	MainFrame(const wxString& title);

private:
	// ** UI Components **

	/**
	 * @brief Main panel containing all other components.
	 *
	 * This panel acts as the parent container for all UI elements in the application.
	 */
	wxPanel* mainPanel;

	/**
	 * @brief Panel containing the button elements.
	 *
	 * This panel is used to group and layout the buttons in a horizontal row.
	 */
	wxPanel* buttonPanel;

	/**
	 * @brief Array of buttons for user interaction.
	 *
	 * Buttons represent different actions the user can select.
	 */
	wxButton* buttons[5];

	/**
	 * @brief Tracks the index of the currently selected button.
	 *
	 * Used to highlight the selected button and reset the appearance of others.
	 */
	int selectedButtonIndex;

	/**
	 * @brief Static text labels and sliders for the three adjustable parameters.
	 *
	 * These labels and sliders are used for controlling application parameters.
	 */
	wxStaticText* keyLevelLabel;       ///< Label for the "Key Level" slider.
	wxStaticText* keyScaleLabel;       ///< Label for the "Key Scale" slider.
	wxStaticText* defaultScaleLabel;   ///< Label for the "Default Scale" slider.

	wxSlider* keyLevelSlider;          ///< Slider for adjusting "Key Level".
	wxSlider* keyScaleSlider;          ///< Slider for adjusting "Key Scale".
	wxSlider* defaultScaleSlider;      ///< Slider for adjusting "Default Scale".

	wxStaticText* keyLevelMinLabel;    ///< Min value label for "Key Level".
	wxStaticText* keyLevelMaxLabel;    ///< Max value label for "Key Level".
	wxStaticText* keyScaleMinLabel;    ///< Min value label for "Key Scale".
	wxStaticText* keyScaleMaxLabel;    ///< Max value label for "Key Scale".
	wxStaticText* defaultScaleMinLabel; ///< Min value label for "Default Scale".
	wxStaticText* defaultScaleMaxLabel; ///< Max value label for "Default Scale".

	// ** Helper Methods **

	/**
	 * @brief Sets up the button panel with styled buttons.
	 *
	 * Adds buttons to the `buttonPanel` and binds click events to a common handler.
	 */
	void SetupButtonPanel();

	/**
	 * @brief Sets up sliders and their associated labels.
	 *
	 * Adds three sliders ("Key Level", "Key Scale", "Default Scale") and their min/max labels to the main layout.
	 */
	void SetupSliders();

	/**
	 * @brief Updates the appearance of the buttons.
	 *
	 * Highlights the selected button and resets the appearance of the others.
	 *
	 * @param selectedIndex The index of the button to highlight. Pass -1 to reset all buttons.
	 */
	void UpdateButtonAppearance(int selectedIndex);

	// ** Event Handlers **

	/**
	 * @brief Handles button click events.
	 *
	 * Updates the selected button index and triggers a UI refresh to reflect the selection.
	 *
	 * @param event The event object representing the button click.
	 */
	void OnButtonClicked(wxCommandEvent& event);

	/**
	 * @brief Handles changes to the "Key Level" slider.
	 *
	 * Updates the application's Key Level parameter based on the slider value.
	 *
	 * @param event The event object representing the slider change.
	 */
	void OnKeyLevelChange(wxCommandEvent& event);

	/**
	 * @brief Handles changes to the "Key Scale" slider.
	 *
	 * Updates the application's Key Scale parameter based on the slider value.
	 *
	 * @param event The event object representing the slider change.
	 */
	void OnKeyScaleChange(wxCommandEvent& event);

	/**
	 * @brief Handles changes to the "Default Scale" slider.
	 *
	 * Updates the application's Default Scale parameter based on the slider value.
	 *
	 * @param event The event object representing the slider change.
	 */
	void OnDefaultScaleChange(wxCommandEvent& event);
};

#endif // WX_GUI_H