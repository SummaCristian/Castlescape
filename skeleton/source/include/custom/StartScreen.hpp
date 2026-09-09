// ***** CUSTOM *****

// Launch screen: opaque backdrop, Play/Settings/Quit buttons. Open from the
// first frame; Play hands control to GameLogic(). Near-identical to
// PauseMenu.hpp but fully opaque and Quit really exits.
// Header-only, gated behind STARTSCREEN_IMPLEMENTATION (Libs.cpp).

#include <algorithm>
#include <string>
#include <vector>

struct StartScreen {
	// title: shown as-is; main.cpp passes its own windowTitle.
	void init(TextMaker *txt, UiQuad *quads, const std::string &title);
	bool isOpen() const { return open; }

	// Opens/closes the screen and re-renders. Like PauseMenu::setOpen.
	void setOpen(bool isOpen, int screenW, int screenH);

	// Reads input, moves selection, re-renders if changed. Once per frame from
	// GameLogic(), BEFORE getSixAxis.
	void update(GLFWwindow *window, int screenW, int screenH);

	// True for the frame a button was clicked/Enter-confirmed. Effects live in
	// main.cpp: Play closes this, Settings opens SettingsMenu, Quit closes the window.
	bool playClicked() const { return wantsPlay; }
	bool settingsClicked() const { return wantsSettings; }
	bool quitClicked() const { return wantsQuit; }
	void clearRequests() { wantsPlay = false; wantsSettings = false; wantsQuit = false; }

	private:
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;
	std::string title;

	bool open = false;
	bool wantsPlay = false;
	bool wantsSettings = false;
	bool wantsQuit = false;

	static constexpr int NUM_BUTTONS = 3;
	int selectedIndex = 0; // 0 = Play, 1 = Settings, 2 = Quit

	int lastScreenW = -1;
	int lastScreenH = -1;

	bool upKeyWasPressed = false;
	bool downKeyWasPressed = false;
	bool enterKeyWasPressed = false;
	bool leftMouseWasPressed = false;

	bool dirty = true;

	// Layout in pixels, centered -- same as PauseMenu.
	static constexpr float BUTTON_WIDTH = 240.0f;
	static constexpr float BUTTON_GAP = 20.0f;
	static constexpr float TITLE_GAP = 60.0f;
	static constexpr float TITLE_SCALE = 2.0f;
	static constexpr float BUTTON_TEXT_SCALE = 1.0f;
	static constexpr float BUTTON_LINE_GAP = 16.0f;

	// Text-block ids, past CheatHud's and PauseMenu's ranges.
	static constexpr int TITLE_TEXT_ID = 300;
	static constexpr int FIRST_BUTTON_TEXT_ID = 301;

	static const std::string &buttonLabel(int i) {
		static const std::string labels[NUM_BUTTONS] = {"Play", "Settings", "Quit"};
		return labels[i];
	}

	float measureTextHeight(int fontId, float scale) const;
	// Measured, not guessed -- see PauseMenu::buttonHeight().
	float buttonHeight() const { return measureTextHeight(10, BUTTON_TEXT_SCALE) + BUTTON_LINE_GAP; }
	float contentTop(int screenH) const;
	float buttonTop(int i, int screenH) const {
		return contentTop(screenH) + measureTextHeight(2, TITLE_SCALE) + TITLE_GAP
			 + i * (buttonHeight() + BUTTON_GAP);
	}
	float buttonLeft(int screenW) const { return (float)screenW / 2.0f - BUTTON_WIDTH / 2.0f; }

	void render(int screenW, int screenH);
	void hide();
	static void pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay);
};

#ifdef STARTSCREEN_IMPLEMENTATION

void StartScreen::init(TextMaker *_txt, UiQuad *_quads, const std::string &_title) {
	txt = _txt;
	quads = _quads;
	title = _title;
}

void StartScreen::pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay) {
	ax = (px / (float)screenW) * 2.0f - 1.0f;
	ay = (py / (float)screenH) * 2.0f - 1.0f;
}

float StartScreen::measureTextHeight(int fontId, float scale) const {
	int w, h, nlines, totChars;
	std::vector<int> linew;
	std::vector<std::string> lines;
	txt->measureText("Ag", fontId, w, h, nlines, totChars, linew, lines);
	return (float)h * scale;
}

float StartScreen::contentTop(int screenH) const {
	float totalH = measureTextHeight(2, TITLE_SCALE) + TITLE_GAP
				 + NUM_BUTTONS * buttonHeight() + (NUM_BUTTONS - 1) * BUTTON_GAP;
	return (float)screenH / 2.0f - totalH / 2.0f;
}

void StartScreen::setOpen(bool isOpen, int screenW, int screenH) {
	if(open == isOpen) {
		return;
	}
	open = isOpen;
	wantsPlay = false;
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

void StartScreen::update(GLFWwindow *window, int screenW, int screenH) {
	if(!open) {
		return;
	}

	if(screenW != lastScreenW || screenH != lastScreenH) {
		lastScreenW = screenW;
		lastScreenH = screenH;
		dirty = true;
	}

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
			wantsPlay = true;
		} else if(selectedIndex == 1) {
			wantsSettings = true;
		} else {
			wantsQuit = true;
		}
	}
	enterKeyWasPressed = enterPressed;

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
			wantsPlay = true;
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

void StartScreen::render(int screenW, int screenH) {
	float ax, ay;

	// Fully opaque, unlike PauseMenu's deliberate dim.
	std::vector<UiRect> rects;
	rects.push_back({0.0f, 0.0f, (float)screenW, (float)screenH, {0.03f, 0.03f, 0.05f, 1.0f}});
	for(int i = 0; i < NUM_BUTTONS; i++) {
		bool selected = (i == selectedIndex);
		glm::vec4 color = selected ? glm::vec4(1.0f, 1.0f, 1.0f, 0.28f)
									: glm::vec4(1.0f, 1.0f, 1.0f, 0.14f);
		rects.push_back({buttonLeft(screenW), buttonTop(i, screenH), BUTTON_WIDTH, buttonHeight(), color});
	}
	quads->setQuads(rects);

	pixelToAnchor((float)screenW / 2.0f, contentTop(screenH), screenW, screenH, ax, ay);
	txt->print(ax, ay, title, TITLE_TEXT_ID, "CO", false, true, false,
			   TAL_CENTER, TRH_CENTER, TRV_TOP,
			   {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f},
			   TITLE_SCALE, TITLE_SCALE);

	for(int i = 0; i < NUM_BUTTONS; i++) {
		bool selected = (i == selectedIndex);
		float top = buttonTop(i, screenH);
		// TRV_TOP: anchor at the button's top edge, not a computed midpoint.
		pixelToAnchor((float)screenW / 2.0f, top, screenW, screenH, ax, ay);
		glm::vec4 labelColor = selected ? glm::vec4(1.0f, 1.0f, 0.3f, 1.0f)
										 : glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
		txt->print(ax, ay, buttonLabel(i), FIRST_BUTTON_TEXT_ID + i, "SS", false, selected, false,
				   TAL_CENTER, TRH_CENTER, TRV_TOP, labelColor,
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, BUTTON_TEXT_SCALE, BUTTON_TEXT_SCALE);
	}
}

void StartScreen::hide() {
	txt->removeText(TITLE_TEXT_ID);
	for(int i = 0; i < NUM_BUTTONS; i++) {
		txt->removeText(FIRST_BUTTON_TEXT_ID + i);
	}
	quads->setQuads({});
}

#endif
