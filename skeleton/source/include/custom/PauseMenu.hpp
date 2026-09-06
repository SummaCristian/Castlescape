// ***** CUSTOM *****

// A modal pause menu: a full-screen dim overlay plus three centered buttons,
// Resume / Settings / Quit. Same TextMaker + UiQuad approach as CheatHud, but
// the open/close transition is owned by main.cpp -- opening the menu also has
// to freeze GameLogic(), so main.cpp drives it either way (setOpen()).
//
// Header-only, implementation gated behind PAUSEMENU_IMPLEMENTATION (Libs.cpp).
// Assumes Starter.hpp, TextMaker.hpp and UiQuad.hpp were included first.

#include <algorithm>
#include <string>
#include <vector>

struct PauseMenu {
	void init(TextMaker *txt, UiQuad *quads);
	bool isOpen() const { return open; }

	// Opens/closes the menu and re-renders. Called from main.cpp -- from the
	// ESC handling (toggle) and from GameLogic() on a Resume click (close).
	// One place writes "open", called from both.
	void setOpen(bool isOpen, int screenW, int screenH);

	// Reads input, moves hover/selection, re-renders if changed. Once per
	// frame from GameLogic(), BEFORE getSixAxis so a click isn't also a
	// drag-look.
	void update(GLFWwindow *window, int screenW, int screenH);

	// True for the frame a button was clicked or Enter-confirmed. The actual
	// effects (Resume closes, Settings opens SettingsMenu) live in main.cpp,
	// so one place flips "open". Cleared by clearRequests(), not automatically.
	bool resumeClicked() const { return wantsResume; }
	bool settingsClicked() const { return wantsSettings; }
	bool quitClicked() const { return wantsQuit; }
	void clearRequests() { wantsResume = false; wantsSettings = false; wantsQuit = false; }

	private:
	// Not owned: the game's shared TextMaker/UiQuad instances.
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;

	bool open = false;
	bool wantsResume = false;
	bool wantsSettings = false;
	bool wantsQuit = false;

	static constexpr int NUM_BUTTONS = 3;
	int selectedIndex = 0; // 0 = Resume, 1 = Settings, 2 = Quit

	// Last screen size render() ran at -- a resize needs a forced re-render
	// with fresh anchors, not just resizeScreen(). See CheatHud.
	int lastScreenW = -1;
	int lastScreenH = -1;

	bool upKeyWasPressed = false;
	bool downKeyWasPressed = false;
	bool enterKeyWasPressed = false;
	bool leftMouseWasPressed = false;

	bool dirty = true;

	// Layout in pixels, centered. Fixed-WIDTH buttons, but not fixed-height
	// (buttonHeight() below): a guessed constant let buttons overlap where
	// the rendered text ran taller.
	static constexpr float BUTTON_WIDTH = 240.0f;
	static constexpr float BUTTON_GAP = 20.0f;
	static constexpr float TITLE_GAP = 50.0f; // title baseline to first button
	static constexpr float TITLE_SCALE = 1.6f;
	static constexpr float BUTTON_TEXT_SCALE = 1.0f;
	static constexpr float BUTTON_LINE_GAP = 16.0f;

	// Text-block ids. Past CheatHud's range, though only one is ever open.
	static constexpr int TITLE_TEXT_ID = 200;
	static constexpr int FIRST_BUTTON_TEXT_ID = 201;

	static const std::string &buttonLabel(int i) {
		static const std::string labels[NUM_BUTTONS] = {"Resume", "Settings", "Quit"};
		return labels[i];
	}

	float measureTextHeight(int fontId, float scale) const;
	// A button's real height, measured. fontId 10 is "SS" Bold, the variant a
	// SELECTED button prints with -- a bitmap bold face can run taller, so
	// sizing every button off it stops the selected one overflowing its row.
	float buttonHeight() const { return measureTextHeight(10, BUTTON_TEXT_SCALE) + BUTTON_LINE_GAP; }

	// Top of the centered block (title + gap + buttons), one anchor for both.
	float contentTop(int screenH) const;
	float buttonTop(int i, int screenH) const {
		return contentTop(screenH) + measureTextHeight(2, TITLE_SCALE) + TITLE_GAP
			 + i * (buttonHeight() + BUTTON_GAP);
	}
	float buttonLeft(int screenW) const { return (float)screenW / 2.0f - BUTTON_WIDTH / 2.0f; }

	void render(int screenW, int screenH);
	void hide();
	// Top-left pixel coord -> TextMaker::print's NDC-ish anchor.
	static void pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay);
};

#ifdef PAUSEMENU_IMPLEMENTATION

void PauseMenu::init(TextMaker *_txt, UiQuad *_quads) {
	txt = _txt;
	quads = _quads;
}

void PauseMenu::pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay) {
	ax = (px / (float)screenW) * 2.0f - 1.0f;
	ay = (py / (float)screenH) * 2.0f - 1.0f;
}

float PauseMenu::measureTextHeight(int fontId, float scale) const {
	int w, h, nlines, totChars;
	std::vector<int> linew;
	std::vector<std::string> lines;
	// Content doesn't matter, only fontId/nlines do.
	txt->measureText("Ag", fontId, w, h, nlines, totChars, linew, lines);
	return (float)h * scale;
}

float PauseMenu::contentTop(int screenH) const {
	float totalH = measureTextHeight(2, TITLE_SCALE) + TITLE_GAP
				 + NUM_BUTTONS * buttonHeight() + (NUM_BUTTONS - 1) * BUTTON_GAP;
	return (float)screenH / 2.0f - totalH / 2.0f;
}

void PauseMenu::setOpen(bool isOpen, int screenW, int screenH) {
	if(open == isOpen) {
		return;
	}
	open = isOpen;
	wantsResume = false;
	wantsSettings = false;
	wantsQuit = false;
	if(open) {
		selectedIndex = 0;
		lastScreenW = screenW;
		lastScreenH = screenH;
		render(screenW, screenH);
	} else {
		hide();
	}
	dirty = false;
}

void PauseMenu::update(GLFWwindow *window, int screenW, int screenH) {
	if(!open) {
		return;
	}

	// A resize leaves TextMaker showing text baked for the old size.
	if(screenW != lastScreenW || screenH != lastScreenH) {
		lastScreenW = screenW;
		lastScreenH = screenH;
		dirty = true;
	}

	// Up/Down move the selection, Enter confirms.
	bool upPressed = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
	if(upPressed && !upKeyWasPressed) {
		selectedIndex = (selectedIndex - 1 + NUM_BUTTONS) % NUM_BUTTONS;
		dirty = true;
	}
	upKeyWasPressed = upPressed;

	bool downPressed = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
	if(downPressed && !downKeyWasPressed) {
		selectedIndex = (selectedIndex + 1) % NUM_BUTTONS;
		dirty = true;
	}
	downKeyWasPressed = downPressed;

	bool enterPressed = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS;
	if(enterPressed && !enterKeyWasPressed) {
		if(selectedIndex == 0) {
			wantsResume = true;
		} else if(selectedIndex == 1) {
			wantsSettings = true;
		} else {
			wantsQuit = true;
		}
	}
	enterKeyWasPressed = enterPressed;

	// Mouse: hover selects, click selects and confirms.
	double mx, my;
	glfwGetCursorPos(window, &mx, &my);
	int hoveredIndex = -1;
	for(int i = 0; i < NUM_BUTTONS; i++) {
		float top = buttonTop(i, screenH);
		float left = buttonLeft(screenW);
		if(mx >= left && mx <= left + BUTTON_WIDTH && my >= top && my <= top + buttonHeight()) {
			hoveredIndex = i;
			break;
		}
	}
	if(hoveredIndex != -1 && hoveredIndex != selectedIndex) {
		selectedIndex = hoveredIndex;
		dirty = true;
	}

	bool leftMousePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	if(leftMousePressed && !leftMouseWasPressed && hoveredIndex != -1) {
		if(hoveredIndex == 0) {
			wantsResume = true;
		} else if(hoveredIndex == 1) {
			wantsSettings = true;
		} else {
			wantsQuit = true;
		}
	}
	leftMouseWasPressed = leftMousePressed;

	if(dirty) {
		render(screenW, screenH);
		dirty = false;
	}
}

void PauseMenu::render(int screenW, int screenH) {
	float ax, ay;

	// Full-screen dim overlay, then one quad per button, the selected one
	// lighter -- in that order so the buttons composite on top of the dim.
	std::vector<UiRect> rects;
	rects.push_back({0.0f, 0.0f, (float)screenW, (float)screenH, {0.0f, 0.0f, 0.0f, 0.55f}});
	for(int i = 0; i < NUM_BUTTONS; i++) {
		bool selected = (i == selectedIndex);
		glm::vec4 color = selected ? glm::vec4(1.0f, 1.0f, 1.0f, 0.28f)
									: glm::vec4(1.0f, 1.0f, 1.0f, 0.14f);
		rects.push_back({buttonLeft(screenW), buttonTop(i, screenH), BUTTON_WIDTH, buttonHeight(), color});
	}
	quads->setQuads(rects);

	pixelToAnchor((float)screenW / 2.0f, contentTop(screenH), screenW, screenH, ax, ay);
	txt->print(ax, ay, "PAUSED", TITLE_TEXT_ID, "CO", false, true, false,
			   TAL_CENTER, TRH_CENTER, TRV_TOP,
			   {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f},
			   TITLE_SCALE, TITLE_SCALE);

	for(int i = 0; i < NUM_BUTTONS; i++) {
		bool selected = (i == selectedIndex);
		float top = buttonTop(i, screenH);
		// TRV_TOP, not TRV_MIDDLE: anchor at the button's top edge rather than
		// trusting TextMaker to center around a computed midpoint.
		pixelToAnchor((float)screenW / 2.0f, top, screenW, screenH, ax, ay);
		glm::vec4 labelColor = selected ? glm::vec4(1.0f, 1.0f, 0.3f, 1.0f)
										 : glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
		txt->print(ax, ay, buttonLabel(i), FIRST_BUTTON_TEXT_ID + i, "SS", false, selected, false,
				   TAL_CENTER, TRH_CENTER, TRV_TOP, labelColor,
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, BUTTON_TEXT_SCALE, BUTTON_TEXT_SCALE);
	}
}

void PauseMenu::hide() {
	txt->removeText(TITLE_TEXT_ID);
	for(int i = 0; i < NUM_BUTTONS; i++) {
		txt->removeText(FIRST_BUTTON_TEXT_ID + i);
	}
	quads->setQuads({});
}

#endif
