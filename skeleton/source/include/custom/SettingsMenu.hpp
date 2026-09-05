// ***** CUSTOM *****

// A full-screen settings menu: an opaque backdrop, a centered "SETTINGS"
// title, one row per adjustable value (Left/Right to change it, same idiom
// as CheatHud's sliders), and a "Back" row at the end. Reachable from both
// StartScreen and PauseMenu (each grew a third "Settings" button for it, see
// their own settingsClicked()) -- main.cpp is what remembers which of those
// two to reopen when Back is pressed here, this struct only knows it was
// opened and that it can be closed.
//
// Deliberately its OWN struct rather than reusing CheatHud, even though the
// row/slider mechanics are the same idea: CheatHud is a small top-left
// corner panel meant to float over live gameplay, laid out to fit whatever
// screen height is left over; this is a full centered screen in the same
// family as PauseMenu/StartScreen, opaque, with no gameplay visible behind
// it and no need to fit around anything else. Sharing one widget between
// "debug corner overlay" and "the game's own settings screen" would have
// meant one of the two constantly fighting the other's layout assumptions.
//
// Same header-only "module" pattern as TextMaker/CheatHud/PauseMenu/
// StartScreen: declarations + implementation gated behind
// SETTINGSMENU_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// "modules/Starter.hpp", "modules/TextMaker.hpp" and "custom/UiQuad.hpp" are
// already included by whoever includes this one.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// One adjustable row. Left/Right change *value by step, clamped to
// [min, max]; onChange fires once per press that actually moved it (not
// once per frame held -- same edge-triggered reasoning as CheatHud's
// identical field, since this can be an expensive operation, a render-target
// rebuild). format turns the raw value into the text shown on the row's
// right; nullptr defaults to "< NN% >" of value*100, which only reads
// sensibly for a 0..1-fraction range -- a row over some other kind of range
// passes its own, the same convention CheatHud's slider rows use.
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

	// Opens or closes the screen and (re)renders it immediately to match --
	// same reasoning as PauseMenu::setOpen/StartScreen::setOpen.
	void setOpen(bool isOpen, int screenW, int screenH);

	// Reads keyboard/mouse input, moves the hover/selection between rows,
	// adjusts the selected row's value on Left/Right, and re-renders if
	// anything changed. Does nothing while closed. Called once per frame
	// from GameLogic(), same spot/reasoning as CheatHud/PauseMenu/
	// StartScreen's own update() (BEFORE getSixAxis).
	void update(GLFWwindow *window, int screenW, int screenH);

	// True for exactly the frame "Back" was clicked or Enter-confirmed.
	// Its actual effect (closing this screen, reopening whichever of
	// PauseMenu/StartScreen sent the player here) lives in main.cpp, not in
	// here -- same division of responsibility as PauseMenu/StartScreen's
	// own wantsX flags.
	bool backClicked() const { return wantsBack; }
	void clearRequests() { wantsBack = false; }

	private:
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;

	std::vector<SettingsRow> rows;
	bool open = false;
	bool wantsBack = false;

	// Selection index into [rows..., Back] -- rows.size() itself means
	// "Back" is selected, one past the last real row.
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

	// Layout, in pixels, centered on screen -- same idiom as PauseMenu's own
	// layout section, except the panel WIDTH is measured from the actual
	// content (panelWidth below), not a fixed guess: an earlier version
	// fixed it at 460px, which fit "MSAA" + "< 4x >" fine but was nowhere
	// near wide enough for "Render Scale" + its own value text (a genuinely
	// longer string, since it also shows the live pixel resolution) --
	// label and value are independently left/right-anchored within the row,
	// so whenever their combined width exceeds the row's, they collide in
	// the middle REGARDLESS of screen size or resize state, which is exactly
	// what made this one persistent rather than the earlier (real, but
	// different) bugs that only showed up on some displays or mid-resize.
	// CheatHud already measures its own panel width from content for the
	// identical reason; this now does the same instead of guessing a second
	// time.
	//
	// TITLE_SCALE/ROW_TEXT_SCALE below are the ASKED-for sizes -- what's
	// actually used is titleScale/rowScale further down, which computeLayout()
	// shrinks by a "fit" factor (same technique as CheatHud's own
	// computeLayout()) whenever the content -- title, every row, Back --
	// wouldn't otherwise fit within the screen's real height. Without this a
	// tall enough row list on a short window would just run off the
	// top/bottom; CheatHud already solves this for its own corner panel, so
	// this reuses the identical idea for a centered full-screen one instead
	// of inventing a second way to do it.
	static constexpr float TITLE_GAP = 50.0f;
	static constexpr float TITLE_SCALE = 1.6f;
	static constexpr float ROW_TEXT_SCALE = 1.0f;
	// Floor for the fit shrink, same reasoning and same value as CheatHud's
	// MIN_FIT_SCALE: past this the font stops being readable, so content is
	// allowed to run off screen instead of a menu nobody can read.
	static constexpr float MIN_FIT_SCALE = 0.4f;
	// How much clear space is kept above and below the whole centered block
	// before the fit shrink kicks in at all.
	static constexpr float SCREEN_MARGIN = 40.0f;
	// Horizontal padding kept clear on each side of the panel's content, and
	// the minimum gap kept between a row's label and its value so they can
	// never visually run into each other even if panelWidth (below) were
	// somehow measured a little short.
	static constexpr float PANEL_PADDING = 20.0f;
	static constexpr float LABEL_VALUE_GAP = 30.0f;
	// Extra gap between the last slider row and "Back", so it doesn't read
	// as just another setting. Scaled by fit along with everything else in
	// computeLayout(), same as TITLE_GAP/ROW_LINE_GAP.
	static constexpr float BACK_GAP = 16.0f;
	// Breathing room added on top of the measured glyph height for each row,
	// same idiom as CheatHud's LINE_GAP.
	static constexpr float ROW_LINE_GAP = 12.0f;

	static constexpr int TITLE_TEXT_ID = 400;
	static constexpr int FIRST_LABEL_TEXT_ID = 401;
	// Generous fixed spacing between the two text-id ranges below, wide
	// enough for any realistic row count -- see FIRST_VALUE_TEXT_ID's use.
	static constexpr int MAX_ROWS = 32;
	static constexpr int FIRST_VALUE_TEXT_ID = FIRST_LABEL_TEXT_ID + MAX_ROWS;
	static constexpr int BACK_TEXT_ID = FIRST_VALUE_TEXT_ID + MAX_ROWS;

	// Populated by computeLayout(), invalidated (see update()) whenever the
	// screen size actually changes -- same idiom as CheatHud's own
	// layoutComputed.
	bool layoutComputed = false;
	float titleScale = TITLE_SCALE;
	float rowScale = ROW_TEXT_SCALE;
	// Title height (measured, at titleScale, plus TITLE_GAP) and row height
	// (measured, at rowScale, plus ROW_LINE_GAP) at the CURRENT fit -- cached
	// here rather than recomputed from the constants on every call, so
	// contentTop()/rowTop() and computeLayout() can't disagree with each
	// other about what fit was actually applied.
	float titleRowHeight = 0.0f;
	float computedRowHeight = 0.0f;
	// The panel's width, measured in computeLayout() from the widest row's
	// actual label+value content at the current rowScale (see its own
	// comment there) -- this is what was a fixed, too-narrow guess before.
	float panelWidth = 0.0f;

	float measureTextHeight(int fontId, float scale) const;
	float measureTextWidth(const std::string &s, int fontId, float scale) const;
	// screenH: the panel shrinks itself (titleScale/rowScale) to fit inside
	// it if the unshrunk content wouldn't -- see the layout members' own
	// comments above and CheatHud::computeLayout(), the same technique. Also
	// where panelWidth (above) gets measured.
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

	// fontId 2 is "CO" Bold (the title's own print() call below); fontId 10
	// is "SS" Bold (a SELECTED row's print() call -- see rowScale's own
	// comment history for why the bold variant specifically, preserved from
	// the earlier fixed-height version of this file).
	const int titleFontId = 2;
	const int rowFontId = 10;

	// Heights at the ASKED-for scales first, since whether they fit is
	// exactly the question -- same structure as CheatHud::computeLayout().
	float titleH = measureTextHeight(titleFontId, TITLE_SCALE) + TITLE_GAP;
	float rowH = measureTextHeight(rowFontId, ROW_TEXT_SCALE) + ROW_LINE_GAP;
	int numRows = (int)rows.size() + 1; // +1 for Back
	float contentH = titleH + rowH * (float)numRows + BACK_GAP;
	float availableH = (float)screenH - SCREEN_MARGIN * 2.0f;

	// One factor for title and row alike, so the whole screen shrinks as a
	// unit and keeps its proportions instead of squeezing rows onto text
	// that stayed full-size.
	float fit = 1.0f;
	if(contentH > availableH && contentH > 0.0f) {
		fit = std::max(availableH / contentH, MIN_FIT_SCALE);
	}
	titleScale = TITLE_SCALE * fit;
	rowScale = ROW_TEXT_SCALE * fit;
	titleRowHeight = titleH * fit;
	computedRowHeight = rowH * fit;

	// Width, measured at the scales just settled on above: the widest of the
	// title text, and every row's own label+gap+value combined -- see this
	// member's own comment for why a fixed guess doesn't work here (a
	// row's value text isn't a fixed length, e.g. Render Scale's shows a
	// live resolution alongside its percentage). Rows are measured by their
	// CURRENT actual value/format text, not a placeholder, so a value that
	// happens to render wider than "< 100% >" (this project's only two
	// sliders both stay well under that, but a future one might not) still
	// gets accounted for correctly instead of assuming a fixed reference
	// string the way CheatHud's own stateWidth does for its short ON/OFF
	// pairs.
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
		// The cached layout (titleScale/rowScale/*RowHeight) is only valid
		// for the height it was fitted to -- see computeLayout() -- so a
		// resize throws it away too, same as CheatHud's identical
		// invalidation.
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

	// Enter confirms "Back"; on a slider row it does nothing (Left/Right own
	// those, same division CheatHud's Enter/Left/Right split carries).
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

	// Mouse: hovering a row selects it; a click on "Back" confirms it (a
	// click on a slider row just selects it, same as CheatHud -- adjusting a
	// slider is keyboard-only, there's no click-and-drag here).
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

	// Fully opaque backdrop, same reasoning as StartScreen's: reachable from
	// either StartScreen (nothing behind it yet) or PauseMenu (mid-game),
	// and reading as one consistent "you're in the settings screen now"
	// rather than looking different depending on where you came from is
	// simpler than carrying an extra "which style" flag through from
	// main.cpp for a purely cosmetic difference.
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

		// TRV_TOP, not TRV_MIDDLE -- see the module-level note above this
		// function (added after rows were seen overlapping): every row is
		// anchored at its own TOP edge, matching CheatHud's own rows (the
		// one widget in this project that has never mispositioned itself),
		// rather than trusting TextMaker to center it around a computed
		// midpoint.
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
