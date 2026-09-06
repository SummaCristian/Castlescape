// ***** CUSTOM *****

// A full-screen settings menu: opaque backdrop, centered "SETTINGS" title, one
// Left/Right row per adjustable value, and a "Back" row. Reachable from both
// StartScreen and PauseMenu; main.cpp remembers which to reopen on Back.
//
// Its OWN struct, not a reused CheatHud: CheatHud is a small corner panel over
// live gameplay, laid out around whatever height is left; this is a centered
// full screen in the PauseMenu/StartScreen family, with nothing behind it.
//
// Header-only, implementation gated behind SETTINGSMENU_IMPLEMENTATION
// (Libs.cpp). Assumes Starter.hpp, TextMaker.hpp and UiQuad.hpp came first.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// One adjustable row. Left/Right change *value by step, clamped to [min, max];
// onChange fires once per press that moved it (edge-triggered, so a held key
// can't spam a rebuild). format turns the value into the right-hand text;
// default is "< NN% >" of value*100, for a 0..1 range. Same as CheatHud.
struct SettingsRow {
	std::string label;
	float *value = nullptr;
	float min = 0.0f;
	float max = 1.0f;
	float step = 0.05f;
	std::function<void()> onChange;
	std::function<std::string(float)> format;
};

struct SettingsMenu {
	void init(TextMaker *txt, UiQuad *quads);
	void addSlider(const std::string &label, float *value, float min, float max,
				   float step, std::function<void()> onChange = nullptr,
				   std::function<std::string(float)> format = nullptr);
	bool isOpen() const { return open; }

	// Opens/closes the screen and re-renders. Like PauseMenu::setOpen.
	void setOpen(bool isOpen, int screenW, int screenH);

	// Reads input, moves selection, adjusts the selected row on Left/Right,
	// re-renders if changed. Once per frame from GameLogic(), BEFORE getSixAxis.
	void update(GLFWwindow *window, int screenW, int screenH);

	// True for the frame "Back" was clicked or Enter-confirmed. Its effect
	// (close, reopen whichever menu sent the player here) lives in main.cpp.
	bool backClicked() const { return wantsBack; }
	void clearRequests() { wantsBack = false; }

	private:
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;

	std::vector<SettingsRow> rows;
	bool open = false;
	bool wantsBack = false;

	// Selection index into [rows..., Back]; rows.size() means "Back".
	int selectedIndex = 0;
	int backIndex() const { return (int)rows.size(); }

	int lastScreenW = -1;
	int lastScreenH = -1;

	bool upKeyWasPressed = false;
	bool downKeyWasPressed = false;
	bool enterKeyWasPressed = false;
	bool leftArrowKeyWasPressed = false;
	bool rightArrowKeyWasPressed = false;
	bool leftMouseWasPressed = false;

	bool dirty = true;

	// Layout in pixels, centered. Like PauseMenu, except panelWidth is measured
	// from content, not a fixed guess: a fixed 460px fit "MSAA" + "< 4x >" but
	// not "Render Scale" + its value text (which also shows the live
	// resolution), and label and value are independently anchored, so they
	// collided in the middle regardless of screen size. CheatHud measures its
	// panel for the same reason.
	//
	// TITLE_SCALE/ROW_TEXT_SCALE are ASKED-for; titleScale/rowScale are what's
	// used, shrunk by computeLayout()'s fit factor when the content wouldn't
	// fit the screen height -- CheatHud's technique.
	static constexpr float TITLE_GAP = 50.0f;
	static constexpr float TITLE_SCALE = 1.6f;
	static constexpr float ROW_TEXT_SCALE = 1.0f;
	static constexpr float MIN_FIT_SCALE = 0.4f;   // floor for the fit shrink
	static constexpr float SCREEN_MARGIN = 40.0f;  // clear space above/below the block
	static constexpr float PANEL_PADDING = 20.0f;
	static constexpr float LABEL_VALUE_GAP = 30.0f;
	static constexpr float BACK_GAP = 16.0f;       // extra gap before "Back", so it reads apart
	static constexpr float ROW_LINE_GAP = 12.0f;   // extra vertical room per row

	static constexpr int TITLE_TEXT_ID = 400;
	static constexpr int FIRST_LABEL_TEXT_ID = 401;
	// Fixed spacing between the two text-id ranges, wide enough for any row count.
	static constexpr int MAX_ROWS = 32;
	static constexpr int FIRST_VALUE_TEXT_ID = FIRST_LABEL_TEXT_ID + MAX_ROWS;
	static constexpr int BACK_TEXT_ID = FIRST_VALUE_TEXT_ID + MAX_ROWS;

	// Populated by computeLayout(), invalidated on a resize. Cached so
	// contentTop()/rowTop() and computeLayout() can't disagree about the fit.
	bool layoutComputed = false;
	float titleScale = TITLE_SCALE;
	float rowScale = ROW_TEXT_SCALE;
	float titleRowHeight = 0.0f;    // measured, at titleScale, + TITLE_GAP
	float computedRowHeight = 0.0f; // measured, at rowScale, + ROW_LINE_GAP
	float panelWidth = 0.0f;        // widest row's label+value at rowScale

	float measureTextHeight(int fontId, float scale) const;
	float measureTextWidth(const std::string &s, int fontId, float scale) const;
	// Shrinks titleScale/rowScale to fit screenH, and measures panelWidth.
	// CheatHud::computeLayout()'s technique.
	void computeLayout(int screenH);

	float contentTop(int screenH) const;
	// Top of row i (a slider row if i < rows.size(), else "Back").
	float rowTop(int i, int screenH) const;
	float rowLeft(int screenW) const { return (float)screenW / 2.0f - panelWidth / 2.0f; }

	void render(int screenW, int screenH);
	void hide();
	static void pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay);
};

#ifdef SETTINGSMENU_IMPLEMENTATION

void SettingsMenu::init(TextMaker *_txt, UiQuad *_quads) {
	txt = _txt;
	quads = _quads;
}

void SettingsMenu::addSlider(const std::string &label, float *value, float min, float max,
							 float step, std::function<void()> onChange,
							 std::function<std::string(float)> format) {
	SettingsRow row;
	row.label = label;
	row.value = value;
	row.min = min;
	row.max = max;
	row.step = step;
	row.onChange = std::move(onChange);
	row.format = std::move(format);
	rows.push_back(row);
}

void SettingsMenu::pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay) {
	ax = (px / (float)screenW) * 2.0f - 1.0f;
	ay = (py / (float)screenH) * 2.0f - 1.0f;
}

float SettingsMenu::measureTextHeight(int fontId, float scale) const {
	int w, h, nlines, totChars;
	std::vector<int> linew;
	std::vector<std::string> lines;
	txt->measureText("Ag", fontId, w, h, nlines, totChars, linew, lines);
	return (float)h * scale;
}

float SettingsMenu::measureTextWidth(const std::string &s, int fontId, float scale) const {
	int w, h, nlines, totChars;
	std::vector<int> linew;
	std::vector<std::string> lines;
	txt->measureText(s, fontId, w, h, nlines, totChars, linew, lines);
	return (float)w * scale;
}

void SettingsMenu::computeLayout(int screenH) {
	if(layoutComputed) {
		return;
	}

	// fontId 2 is "CO" Bold (the title); fontId 10 is "SS" Bold (a selected
	// row -- measure Bold so the layout isn't too narrow once selected).
	const int titleFontId = 2;
	const int rowFontId = 10;

	// Heights at the ASKED-for scales first: whether they fit is the question.
	float titleH = measureTextHeight(titleFontId, TITLE_SCALE) + TITLE_GAP;
	float rowH = measureTextHeight(rowFontId, ROW_TEXT_SCALE) + ROW_LINE_GAP;
	int numRows = (int)rows.size() + 1; // +1 for Back
	float contentH = titleH + rowH * (float)numRows + BACK_GAP;
	float availableH = (float)screenH - SCREEN_MARGIN * 2.0f;

	// One factor for title and row alike, so the screen keeps its proportions.
	float fit = 1.0f;
	if(contentH > availableH && contentH > 0.0f) {
		fit = std::max(availableH / contentH, MIN_FIT_SCALE);
	}
	titleScale = TITLE_SCALE * fit;
	rowScale = ROW_TEXT_SCALE * fit;
	titleRowHeight = titleH * fit;
	computedRowHeight = rowH * fit;

	// Width at the settled scales: the widest of the title and every row's
	// label+gap+value. Rows measure their CURRENT value text, not a
	// placeholder, so a value wider than "< 100% >" is still accounted for.
	float widest = measureTextWidth("SETTINGS", titleFontId, titleScale);
	for(const SettingsRow &row : rows) {
		std::string valueText = row.format ? row.format(*row.value)
											: [&] {
												char buf[16];
												snprintf(buf, sizeof(buf), "< %d%% >",
														 (int)std::lround(*row.value * 100.0f));
												return std::string(buf);
											  }();
		float rowWidth = measureTextWidth(row.label, rowFontId, rowScale) + LABEL_VALUE_GAP
						+ measureTextWidth(valueText, rowFontId, rowScale);
		widest = std::max(widest, rowWidth);
	}
	widest = std::max(widest, measureTextWidth("Back", rowFontId, rowScale));
	panelWidth = widest + PANEL_PADDING * 2.0f;

	layoutComputed = true;
}

float SettingsMenu::contentTop(int screenH) const {
	int numRows = (int)rows.size() + 1; // +1 for Back
	float totalH = titleRowHeight + numRows * computedRowHeight + BACK_GAP;
	return (float)screenH / 2.0f - totalH / 2.0f;
}

float SettingsMenu::rowTop(int i, int screenH) const {
	float top = contentTop(screenH) + titleRowHeight + i * computedRowHeight;
	if(i == backIndex()) {
		top += BACK_GAP;
	}
	return top;
}

void SettingsMenu::setOpen(bool isOpen, int screenW, int screenH) {
	if(open == isOpen) {
		return;
	}
	open = isOpen;
	wantsBack = false;
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

void SettingsMenu::update(GLFWwindow *window, int screenW, int screenH) {
	if(!open) {
		return;
	}

	if(screenW != lastScreenW || screenH != lastScreenH) {
		lastScreenW = screenW;
		lastScreenH = screenH;
		// The cached layout is only valid for the height it was fitted to.
		layoutComputed = false;
		dirty = true;
	}

	int numSelectable = (int)rows.size() + 1; // rows + Back

	bool upPressed = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
	if(upPressed && !upKeyWasPressed) {
		selectedIndex = (selectedIndex - 1 + numSelectable) % numSelectable;
		dirty = true;
	}
	upKeyWasPressed = upPressed;

	bool downPressed = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
	if(downPressed && !downKeyWasPressed) {
		selectedIndex = (selectedIndex + 1) % numSelectable;
		dirty = true;
	}
	downKeyWasPressed = downPressed;

	// Enter confirms "Back"; nothing on a slider row (Left/Right own those).
	bool enterPressed = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS;
	if(enterPressed && !enterKeyWasPressed && selectedIndex == backIndex()) {
		wantsBack = true;
	}
	enterKeyWasPressed = enterPressed;

	// Left/Right: adjust the selected row's value, if it's a slider row
	// (no-op on "Back").
	auto adjustSelected = [&](float sign) {
		if(selectedIndex >= (int)rows.size()) {
			return;
		}
		SettingsRow &row = rows[selectedIndex];
		float before = *row.value;
		float after = std::clamp(before + sign * row.step, row.min, row.max);
		if(after != before) {
			*row.value = after;
			dirty = true;
			if(row.onChange) {
				row.onChange();
			}
		}
	};

	bool leftArrowPressed = glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS;
	if(leftArrowPressed && !leftArrowKeyWasPressed) {
		adjustSelected(-1.0f);
	}
	leftArrowKeyWasPressed = leftArrowPressed;

	bool rightArrowPressed = glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;
	if(rightArrowPressed && !rightArrowKeyWasPressed) {
		adjustSelected(1.0f);
	}
	rightArrowKeyWasPressed = rightArrowPressed;

	// Mouse: hover selects; a click on "Back" confirms it. A slider is
	// adjusted by keyboard only -- no click-and-drag.
	double mx, my;
	glfwGetCursorPos(window, &mx, &my);
	int hoveredIndex = -1;
	for(int i = 0; i <= backIndex(); i++) {
		float top = rowTop(i, screenH);
		float left = rowLeft(screenW);
		if(mx >= left && mx <= left + panelWidth && my >= top && my <= top + computedRowHeight) {
			hoveredIndex = i;
			break;
		}
	}
	if(hoveredIndex != -1 && hoveredIndex != selectedIndex) {
		selectedIndex = hoveredIndex;
		dirty = true;
	}

	bool leftMousePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	if(leftMousePressed && !leftMouseWasPressed && hoveredIndex == backIndex()) {
		wantsBack = true;
	}
	leftMouseWasPressed = leftMousePressed;

	if(dirty) {
		render(screenW, screenH);
		dirty = false;
	}
}

void SettingsMenu::render(int screenW, int screenH) {
	computeLayout(screenH);

	float ax, ay;

	// Fully opaque backdrop: reachable from StartScreen or mid-game from
	// PauseMenu, and it should read the same either way.
	std::vector<UiRect> rects;
	rects.push_back({0.0f, 0.0f, (float)screenW, (float)screenH, {0.03f, 0.03f, 0.05f, 1.0f}});
	for(int i = 0; i <= backIndex(); i++) {
		bool selected = (i == selectedIndex);
		glm::vec4 color = selected ? glm::vec4(1.0f, 1.0f, 1.0f, 0.28f)
									: glm::vec4(1.0f, 1.0f, 1.0f, 0.14f);
		rects.push_back({rowLeft(screenW), rowTop(i, screenH), panelWidth, computedRowHeight, color});
	}
	quads->setQuads(rects);

	pixelToAnchor((float)screenW / 2.0f, contentTop(screenH), screenW, screenH, ax, ay);
	txt->print(ax, ay, "SETTINGS", TITLE_TEXT_ID, "CO", false, true, false,
			   TAL_CENTER, TRH_CENTER, TRV_TOP,
			   {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f},
			   titleScale, titleScale);

	float left = rowLeft(screenW);
	for(int i = 0; i < (int)rows.size(); i++) {
		const SettingsRow &row = rows[i];
		bool selected = (i == selectedIndex);
		float top = rowTop(i, screenH);
		glm::vec4 labelColor = selected ? glm::vec4(1.0f, 1.0f, 0.3f, 1.0f)
										 : glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);

		// TRV_TOP, not TRV_MIDDLE: anchor each row at its own top edge (like
		// CheatHud) rather than trusting TextMaker to center around a midpoint
		// -- rows were seen overlapping otherwise.
		pixelToAnchor(left + PANEL_PADDING, top, screenW, screenH, ax, ay);
		txt->print(ax, ay, row.label, FIRST_LABEL_TEXT_ID + i, "SS", false, selected, false,
				   TAL_LEFT, TRH_LEFT, TRV_TOP, labelColor,
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, rowScale, rowScale);

		std::string valueText;
		if(row.format) {
			valueText = row.format(*row.value);
		} else {
			char buf[16];
			snprintf(buf, sizeof(buf), "< %d%% >", (int)std::lround(*row.value * 100.0f));
			valueText = buf;
		}
		pixelToAnchor(left + panelWidth - PANEL_PADDING, top, screenW, screenH, ax, ay);
		txt->print(ax, ay, valueText, FIRST_VALUE_TEXT_ID + i, "SS", false, false, false,
				   TAL_RIGHT, TRH_RIGHT, TRV_TOP, glm::vec4(0.4f, 0.8f, 1.0f, 1.0f),
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, rowScale, rowScale);
	}

	{
		bool selected = (selectedIndex == backIndex());
		float top = rowTop(backIndex(), screenH);
		pixelToAnchor((float)screenW / 2.0f, top, screenW, screenH, ax, ay);
		glm::vec4 labelColor = selected ? glm::vec4(1.0f, 1.0f, 0.3f, 1.0f)
										 : glm::vec4(0.9f, 0.9f, 0.9f, 1.0f);
		txt->print(ax, ay, "Back", BACK_TEXT_ID, "SS", false, selected, false,
				   TAL_CENTER, TRH_CENTER, TRV_TOP, labelColor,
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, rowScale, rowScale);
	}
}

void SettingsMenu::hide() {
	txt->removeText(TITLE_TEXT_ID);
	for(int i = 0; i < (int)rows.size(); i++) {
		txt->removeText(FIRST_LABEL_TEXT_ID + i);
		txt->removeText(FIRST_VALUE_TEXT_ID + i);
	}
	txt->removeText(BACK_TEXT_ID);
	quads->setQuads({});
}

#endif
