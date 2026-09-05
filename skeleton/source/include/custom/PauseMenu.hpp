// ***** CUSTOM *****

// A simple modal pause menu: a full-screen dim overlay plus two centered
// buttons, Resume and Quit. Reuses the same TextMaker + UiQuad overlay
// approach as CheatHud (see that file's header comment for why UiQuad exists
// at all), but the open/close *transition* is owned by main.cpp rather than
// by a key baked into this struct: unlike CheatHud's own L toggle, opening
// the pause menu also has to freeze GameLogic()'s movement/physics from
// updateUniformBuffer(), so main.cpp needs to drive that transition itself
// either way -- see setOpen() below.
//
// Quit is intentionally a dead button for now: there is no main-menu/start
// screen anywhere in this project to return to yet, so quitClicked() only
// reports that the click happened; main.cpp decides (currently: does
// nothing but log it) what that means.
//
// Same header-only "module" pattern as TextMaker/CheatHud/UiQuad:
// declarations + implementation in this one file, implementation gated
// behind PAUSEMENU_IMPLEMENTATION (defined once in Libs.cpp). Like those
// modules, this file assumes "modules/Starter.hpp", "modules/TextMaker.hpp"
// and "custom/UiQuad.hpp" are already included by whoever includes this one.

#include <algorithm>
#include <string>
#include <vector>

struct PauseMenu {
	void init(TextMaker *txt, UiQuad *quads);
	bool isOpen() const { return open; }

	// Opens or closes the menu and (re)renders it immediately to match.
	// Called from main.cpp: once from the top-level ESC handling (toggle),
	// and once from GameLogic() when update() below reports Resume was
	// clicked (close). Keeping the open/close write in one place, called
	// from both spots, is simpler than main.cpp and this struct each
	// keeping their own copy of "open" in sync.
	void setOpen(bool isOpen, int screenW, int screenH);

	// Reads keyboard/mouse input, moves the hover/selection between the two
	// buttons, and re-renders if anything changed. Does nothing while
	// closed. Called once per frame from GameLogic(), same spot and same
	// reasoning as CheatHud::update (BEFORE getSixAxis, so a click here
	// isn't also consumed as a drag-look by getSixAxis's sticky mouse
	// buttons).
	void update(GLFWwindow *window, int screenW, int screenH);

	// True for exactly the frame Resume/Settings/Quit was clicked or
	// Enter-confirmed. Resume's actual effect (closing the menu, via
	// setOpen) lives in main.cpp rather than in here, same as Settings'
	// (opening SettingsMenu, remembering to come back here), so that
	// main.cpp has a single place that flips "open" no matter which of the
	// three callers triggered it. Cleared by main.cpp once handled
	// (clearRequests()), not automatically, so a caller that checks all
	// three flags in sequence can't miss one to a stray leftover from
	// update() re-arming it first.
	bool resumeClicked() const { return wantsResume; }
	bool settingsClicked() const { return wantsSettings; }
	bool quitClicked() const { return wantsQuit; }
	void clearRequests() { wantsResume = false; wantsSettings = false; wantsQuit = false; }

	private:
	// Not owned: point at the game's shared TextMaker/UiQuad instances.
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;

	bool open = false;
	bool wantsResume = false;
	bool wantsSettings = false;
	bool wantsQuit = false;

	static constexpr int NUM_BUTTONS = 3;
	int selectedIndex = 0; // 0 = Resume, 1 = Settings, 2 = Quit

	// Last screen size render() was called with -- see CheatHud's identical
	// fields for why this is needed (TextMaker bakes pixel positions into a
	// fixed NDC anchor per print() call, so a resize needs a forced
	// re-render with fresh anchors, not just a resizeScreen()).
	int lastScreenW = -1;
	int lastScreenH = -1;

	bool upKeyWasPressed = false;
	bool downKeyWasPressed = false;
	bool enterKeyWasPressed = false;
	bool leftMouseWasPressed = false;

	// Re-render only when something actually changed (open/close, hover,
	// selection), same idiom as CheatHud's own dirty flag.
	bool dirty = true;

	// Layout, in pixels, centered on screen. Unlike CheatHud there's no
	// growing/shrinking row list to fit against screen height, just three
	// fixed-WIDTH buttons -- but NOT fixed-height, see buttonHeight() below:
	// that used to be a guessed constant here (56.0f), which is what let
	// buttons overlap on displays where the actual rendered text turned out
	// taller than the guess.
	static constexpr float BUTTON_WIDTH = 240.0f;
	static constexpr float BUTTON_GAP = 20.0f;
	static constexpr float TITLE_GAP = 50.0f; // title baseline to first button
	static constexpr float TITLE_SCALE = 1.6f;
	static constexpr float BUTTON_TEXT_SCALE = 1.0f;
	// Breathing room added on top of the measured glyph height, same idiom
	// as CheatHud's LINE_GAP.
	static constexpr float BUTTON_LINE_GAP = 16.0f;

	// Text-block ids handed to TextMaker::print/removeText. Start past
	// CheatHud's own range (100 + its toggle count) so the two can never
	// collide even though only one of the two is ever open at a time.
	static constexpr int TITLE_TEXT_ID = 200;
	static constexpr int FIRST_BUTTON_TEXT_ID = 201;

	static const std::string &buttonLabel(int i) {
		static const std::string labels[NUM_BUTTONS] = {"Resume", "Settings", "Quit"};
		return labels[i];
	}

	float measureTextHeight(int fontId, float scale) const;
	// A button's real height, measured rather than guessed: fontId 10 is
	// "SS" Bold (8 + 2, TextMaker::print's own formula), the BOLD variant
	// specifically because that's what a SELECTED button prints with, and a
	// bitmap font's bold face can render measurably taller than its regular
	// one -- sizing every button (selected or not) off the larger of the
	// two is what stops the selected one from overflowing past its own row.
	// See SettingsMenu's identical rowHeight() for the fuller version of
	// this same comment.
	float buttonHeight() const { return measureTextHeight(10, BUTTON_TEXT_SCALE) + BUTTON_LINE_GAP; }

	// Top of the whole centered block (title + gap + buttons), so title and
	// buttons are laid out from the same anchor and can't drift apart.
	float contentTop(int screenH) const;
	float buttonTop(int i, int screenH) const {
		return contentTop(screenH) + measureTextHeight(2, TITLE_SCALE) + TITLE_GAP
			 + i * (buttonHeight() + BUTTON_GAP);
	}
	float buttonLeft(int screenW) const { return (float)screenW / 2.0f - BUTTON_WIDTH / 2.0f; }

	void render(int screenW, int screenH);
	void hide();
	// Converts a top-left-origin pixel coordinate into the NDC-ish anchor
	// TextMaker::print expects (mirrors TextMaker::pixelToScr, private to
	// that struct -- same duplication CheatHud already carries).
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
	// Content doesn't matter, only fontId/nlines do -- see CheatHud's
	// identical helper.
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

	// A resize while open leaves TextMaker showing text baked for the old
	// screen size (see lastScreenW/H's comment above), so force a
	// re-render with fresh anchors whenever the size actually changes.
	if(screenW != lastScreenW || screenH != lastScreenH) {
		lastScreenW = screenW;
		lastScreenH = screenH;
		dirty = true;
	}

	// Keyboard navigation: Up/Down move the selection, Enter confirms it --
	// same idiom as CheatHud.
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

	// Mouse: hovering a button selects it, a click both selects and
	// confirms it -- same idiom as CheatHud.
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

	// Full-screen dim overlay, then one quad per button (background) with
	// the selected/hovered one drawn lighter -- drawn in that order so the
	// buttons composite on top of the dim rather than under it.
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
		// TRV_TOP, not TRV_MIDDLE -- same reasoning as SettingsMenu's
		// identical change: anchor at the button's own top edge, matching
		// CheatHud's rows, rather than trusting TextMaker to center text
		// around a computed midpoint.
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
