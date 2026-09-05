// ***** CUSTOM *****

// A small keyboard/mouse-navigable HUD for flipping the game's boolean cheat
// flags at runtime. Text rows come from TextMaker; the background panel and
// selection highlight come from UiQuad, since TextMaker's font atlas has no
// filled rectangle to fake a backdrop with.
//
// Follows the same header-only "module" pattern as TextMaker/Scene/Animations:
// declarations + implementation in this one file, implementation gated behind
// CHEATHUD_IMPLEMENTATION (defined once in Libs.cpp). Like those modules, none
// of these headers are self-guarded against double inclusion, so this file
// assumes "modules/Starter.hpp", "modules/TextMaker.hpp" and
// "custom/UiQuad.hpp" are already included by whoever includes this one.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// One row of the panel: either an on/off toggle bound to a live bool, or a
// Left/Right-adjustable slider bound to a live float. Exactly one of
// toggleValue/sliderValue is set, deciding which. The HUD never copies/owns
// the value, it just changes the real one in place, so there's nothing to
// keep in sync -- except for a slider whose effect needs more than "read
// this value" to take hold (RENDER_SCALE's render-target rebuild in
// main.cpp, say), which is what onChange is for.
struct CheatRow {
	std::string label;
	bool *toggleValue = nullptr;
	float *sliderValue = nullptr;
	float sliderMin = 0.0f;
	float sliderMax = 1.0f;
	float sliderStep = 0.05f;
	// Fires once per Left/Right press that actually changed the slider's
	// value -- edge-triggered the same way Up/Down/Enter below are, not
	// once per frame the key is held, since this can be an expensive
	// operation (a swapchain/render-target rebuild) that a held key must
	// not spam. Unused for a toggle row: the caller already reads
	// *toggleValue live every frame.
	std::function<void()> onChange;
	// How *sliderValue is turned into the text shown on the right of the
	// row. Defaults (nullptr) to "<  NN% >" of value*100, which only reads
	// sensibly for a slider whose range is meant as a 0..1 fraction (like
	// renderScale) -- anything else (main.cpp's MSAA level, an exponent
	// rather than a fraction) passes its own, e.g. "<  4x >".
	std::function<std::string(float)> format;
};

struct CheatHud {
	void init(TextMaker *txt, UiQuad *quads);
	void addToggle(const std::string &label, bool *value);
	// step: how much one Left/Right press changes *value by. Clamped to
	// [min, max] after every change. onChange, if set, fires after the
	// clamp, once per press that actually moved the value (not at the
	// clamped ends when already there) -- see CheatRow::onChange. format,
	// if set, overrides the default "<  NN% >" display -- see
	// CheatRow::format.
	void addSlider(const std::string &label, float *value, float min, float max,
				   float step, std::function<void()> onChange = nullptr,
				   std::function<std::string(float)> format = nullptr);
	bool isOpen() const { return open; }

	// Reads keyboard/mouse input, updates the open/selected/toggled state,
	// and re-renders the panel if anything changed. Called once per frame
	// from GameLogic(), BEFORE getSixAxis
	void update(GLFWwindow *window, int screenW, int screenH);

	private:
	// Not owned: point at the game's TextMaker/UiQuad instances, used only to
	// print/remove the panel's text blocks and background/highlight quads.
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;

	std::vector<CheatRow> options;
	int selectedIndex = 0;
	bool open = false;

	// Last screen size renderRows() was called with. renderRows bakes each
	// row's pixel position into a fixed NDC anchor before handing it to
	// TextMaker::print. UiQuad's rects, by contrast, stay in pixel space and
	// get reconverted on every mesh rebuild, but TextMaker::resizeScreen just
	// rebuilds from that already-baked anchor, so it has no way to notice a
	// resize on its own. Tracked here so update() can force a re-render
	// (with freshly baked anchors) whenever the screen size actually changes.
	int lastScreenW = -1;
	int lastScreenH = -1;

	// Edge-detection for each input that should trigger only once per
	// physical press, not once per frame it's held. Same idiom as
	// jumpKeyWasPressed in main.cpp.
	bool lKeyWasPressed = false;
	bool upKeyWasPressed = false;
	bool downKeyWasPressed = false;
	bool enterKeyWasPressed = false;
	bool leftMouseWasPressed = false;
	// Adjust the selected row's slider (no-op on a toggle row).
	bool leftArrowKeyWasPressed = false;
	bool rightArrowKeyWasPressed = false;

	// True whenever the panel needs to be re-printed (just opened/closed,
	// selection or hover moved, a toggle flipped). Avoids rebuilding
	// TextMaker's mesh/command buffer every single frame the HUD merely sits
	// open and idle, the same way the FPS counter only calls txt.print once
	// a second instead of every frame.
	bool dirty = true;

	// Panel layout, in pixels, anchored to the top-left corner. Mouse
	// hit-testing, the highlight quad, and the background panel all derive
	// from the same numbers below, so they can never drift out of sync.
	static constexpr float PANEL_X = 24.0f;
	static constexpr float PANEL_Y = 24.0f;
	// Inset between the background panel's edge and the text/hit-rects
	// inside it, so text isn't flush against the panel border.
	static constexpr float PADDING = 10.0f;
	// Minimum horizontal gap between a row's label and its right-aligned
	// [ON]/[OFF] state, so they can't visually run into each other.
	static constexpr float LABEL_STATE_GAP = 20.0f;
	// Extra vertical breathing room added on top of the measured glyph
	// height for each row/title, so lines aren't packed edge-to-edge.
	static constexpr float LINE_GAP = 8.0f;
	// Text scale (TextMaker::print's sx/sy). Kept close to the font's native
	// size (1.0), small and tight rather than the panel dominating the screen.
	// These are the sizes ASKED for; what actually gets drawn is titleScale/
	// rowScale below, which may be smaller. See computeLayout().
	static constexpr float TITLE_SCALE = 1.0f;
	static constexpr float ROW_SCALE = 0.85f;
	// Floor for that shrinking. Past this the font stops being readable, so
	// the panel is allowed to run off the bottom instead: a menu you can't
	// read is no better than one you can't see all of.
	static constexpr float MIN_FIT_SCALE = 0.4f;

	// Text-block ids handed to TextMaker::print/removeText. Start well past
	// the FPS counter's id (1) so the two can never collide.
	static constexpr int TITLE_TEXT_ID = 100;
	static constexpr int FIRST_ROW_TEXT_ID = 101;

	// Panel width, per-row heights and the text scales actually used, all in
	// pixels. Computed from the real rendered text size (via
	// TextMaker::measureText) instead of guessed constants, so the panel fits
	// its content exactly even after changing the font scale or a label.
	// Populated lazily by computeLayout(), and invalidated on a resize, since
	// how much room there is to fit into is part of what it computes.
	bool layoutComputed = false;
	float panelWidth = 0.0f;
	float titleRowHeight = 0.0f;
	float rowHeight = 0.0f;
	float titleScale = TITLE_SCALE;
	float rowScale = ROW_SCALE;

	// screenH: the panel shrinks itself to fit inside it. The row list grows
	// every time a cheat is added and the window is only 600px tall by
	// default, so at some point a fixed row size runs off the bottom, taking
	// the rows with it (there is no scrolling, and a row you can't see is a
	// row you can't click).
	void computeLayout(int screenH);
	// Width/height, in pixels, of "s" as TextMaker would render it at the
	// given fontId/scale. fontId must match the same (FontFace, Bold,
	// Italic, Small) combination passed to the corresponding print() call.
	// See TextMaker::print's own fontId formula, duplicated here since
	// measureText takes a fontId rather than those flags directly.
	float measureTextWidth(const std::string &s, int fontId, float scale) const;
	float measureTextHeight(int fontId, float scale) const;

	// Row rectangle in pixel space (row 0 sits right below the title row,
	// which may have a different height than the toggle rows).
	float rowTop(int i) const { return PANEL_Y + PADDING + titleRowHeight + rowHeight * i; }
	float panelHeight() const { return PADDING * 2.0f + titleRowHeight + rowHeight * (int)options.size(); }

	void renderRows(int screenW, int screenH);
	void hideRows();
	// Converts a top-left-origin pixel coordinate into the NDC-ish anchor
	// TextMaker::print expects (mirrors TextMaker::pixelToScr, which is
	// private to that struct).
	static void pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay);
};

#ifdef CHEATHUD_IMPLEMENTATION

void CheatHud::init(TextMaker *_txt, UiQuad *_quads) {
	txt = _txt;
	quads = _quads;
}

void CheatHud::addToggle(const std::string &label, bool *value) {
	CheatRow row;
	row.label = label;
	row.toggleValue = value;
	options.push_back(row);
}

void CheatHud::addSlider(const std::string &label, float *value, float min, float max,
						 float step, std::function<void()> onChange,
						 std::function<std::string(float)> format) {
	CheatRow row;
	row.label = label;
	row.sliderValue = value;
	row.sliderMin = min;
	row.sliderMax = max;
	row.sliderStep = step;
	row.onChange = std::move(onChange);
	row.format = std::move(format);
	options.push_back(row);
}

void CheatHud::pixelToAnchor(float px, float py, int screenW, int screenH, float &ax, float &ay) {
	ax = (px / (float)screenW) * 2.0f - 1.0f;
	ay = (py / (float)screenH) * 2.0f - 1.0f;
}

void CheatHud::update(GLFWwindow *window, int screenW, int screenH) {
	// A resize while the HUD is open leaves TextMaker showing rows baked for
	// the old screen size (see lastScreenW/H's comment above), so force a
	// re-render with fresh anchors whenever the size actually changes.
	// Harmless to also run this while closed: renderRows() never gets called
	// until something (here or the L-key toggle below) is actually open.
	if(screenW != lastScreenW || screenH != lastScreenH) {
		lastScreenW = screenW;
		lastScreenH = screenH;
		// The cached layout is only valid for the height it was fitted to
		// (see computeLayout), so a resize throws it away as well.
		layoutComputed = false;
		dirty = true;
	}

	// Open/close toggle, always polled regardless of current state.
	bool lPressed = glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS;
	if(lPressed && !lKeyWasPressed) {
		open = !open;
		dirty = true;
		if(!open) {
			hideRows();
			lKeyWasPressed = lPressed;
			return;
		}
	}
	lKeyWasPressed = lPressed;

	if(!open) {
		return;
	}

	// Keyboard navigation: Up/Down move the selection, Enter flips it.
	bool upPressed = glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS;
	if(upPressed && !upKeyWasPressed) {
		selectedIndex = (selectedIndex - 1 + (int)options.size()) % (int)options.size();
		dirty = true;
	}
	upKeyWasPressed = upPressed;

	bool downPressed = glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS;
	if(downPressed && !downKeyWasPressed) {
		selectedIndex = (selectedIndex + 1) % (int)options.size();
		dirty = true;
	}
	downKeyWasPressed = downPressed;

	bool enterPressed = glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS;
	if(enterPressed && !enterKeyWasPressed && options[selectedIndex].toggleValue != nullptr) {
		bool *v = options[selectedIndex].toggleValue;
		*v = !(*v);
		dirty = true;
	}
	enterKeyWasPressed = enterPressed;

	// Left/Right: adjust the selected row's slider, if it is one. No-op on
	// a toggle row (Enter/click cover those).
	auto adjustSelectedSlider = [&](float sign) {
		CheatRow &row = options[selectedIndex];
		if(row.sliderValue == nullptr) {
			return;
		}
		float before = *row.sliderValue;
		float after = std::clamp(before + sign * row.sliderStep, row.sliderMin, row.sliderMax);
		if(after != before) {
			*row.sliderValue = after;
			dirty = true;
			if(row.onChange) {
				row.onChange();
			}
		}
	};

	bool leftArrowPressed = glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS;
	if(leftArrowPressed && !leftArrowKeyWasPressed) {
		adjustSelectedSlider(-1.0f);
	}
	leftArrowKeyWasPressed = leftArrowPressed;

	bool rightArrowPressed = glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS;
	if(rightArrowPressed && !rightArrowKeyWasPressed) {
		adjustSelectedSlider(1.0f);
	}
	rightArrowKeyWasPressed = rightArrowPressed;

	// Mouse: hovering a row selects it (hover and keyboard selection share
	// the same "selectedIndex", there's only one highlighted concept), and
	// a click both selects and flips the hovered row.
	double mx, my;
	glfwGetCursorPos(window, &mx, &my);
	int hoveredIndex = -1;
	for(int i = 0; i < (int)options.size(); i++) {
		float top = rowTop(i);
		if(mx >= PANEL_X && mx <= PANEL_X + panelWidth && my >= top && my <= top + rowHeight) {
			hoveredIndex = i;
			break;
		}
	}
	if(hoveredIndex != -1 && hoveredIndex != selectedIndex) {
		selectedIndex = hoveredIndex;
		dirty = true;
	}

	bool leftMousePressed = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	if(leftMousePressed && !leftMouseWasPressed && hoveredIndex != -1
	   && options[hoveredIndex].toggleValue != nullptr) {
		bool *v = options[hoveredIndex].toggleValue;
		*v = !(*v);
		dirty = true;
	}
	leftMouseWasPressed = leftMousePressed;

	if(dirty) {
		renderRows(screenW, screenH);
		dirty = false;
	}
}

float CheatHud::measureTextWidth(const std::string &s, int fontId, float scale) const {
	int w, h, nlines, totChars;
	std::vector<int> linew;
	std::vector<std::string> lines;
	txt->measureText(s, fontId, w, h, nlines, totChars, linew, lines);
	return (float)w * scale;
}

float CheatHud::measureTextHeight(int fontId, float scale) const {
	int w, h, nlines, totChars;
	std::vector<int> linew;
	std::vector<std::string> lines;
	// Any non-empty single-line string measures the face's line height;
	// content doesn't matter here, only fontId/nlines do.
	txt->measureText("Ag", fontId, w, h, nlines, totChars, linew, lines);
	return (float)h * scale;
}

void CheatHud::computeLayout(int screenH) {
	if(layoutComputed) {
		return;
	}

	// fontId values matching TextMaker::print's own formula for the
	// (FontFace, Bold, Italic, Small) combinations used in renderRows below:
	// "SS" family is offset 8, Bold adds 2. Rows go bold only when selected;
	// measuring with Bold keeps the layout from being too narrow once a row
	// becomes selected.
	const int titleFontId = 8 + 2;
	const int rowFontId = 8 + 2;
	const int stateFontId = 8;

	// Heights at the asked-for scales first, since whether they fit is exactly
	// the question. The panel is anchored at PANEL_Y and given the same margin
	// at the bottom, so that is twice PANEL_Y gone before any content.
	float titleH = measureTextHeight(titleFontId, TITLE_SCALE) + LINE_GAP;
	float rowH = measureTextHeight(rowFontId, ROW_SCALE) + LINE_GAP;
	float contentH = titleH + rowH * (float)options.size();
	float availableH = (float)screenH - PANEL_Y * 2.0f - PADDING * 2.0f;

	// One factor for text and row heights alike, so the panel shrinks as a
	// whole and keeps its proportions instead of squeezing the rows onto text
	// that stayed big. PADDING is left alone, it's a margin, not content.
	float fit = 1.0f;
	if(contentH > availableH && contentH > 0.0f) {
		fit = std::max(availableH / contentH, MIN_FIT_SCALE);
	}
	titleScale = TITLE_SCALE * fit;
	rowScale = ROW_SCALE * fit;
	titleRowHeight = titleH * fit;
	rowHeight = rowH * fit;

	// Width is measured at the fitted scales, so a shrunk panel is narrower
	// too rather than a short list of tiny text in a full-width box.
	float maxContentWidth = measureTextWidth("CHEATS (L to close)", titleFontId, titleScale);
	// "< 100% (9999x9999) >" stands in for a slider row's value text: a
	// generously wide reference (comfortably past any real window
	// resolution) rather than the exact current text, independent of any
	// row's actual current value -- so the panel's width doesn't shift as a
	// slider is adjusted (matching how it already doesn't shift as a toggle
	// flips between its two fixed-width strings), and doesn't need
	// recomputing every time a row's custom format() text changes length,
	// only layoutComputed's usual resize-triggered invalidation.
	float stateWidth = std::max({measureTextWidth("[ON]", stateFontId, rowScale),
								  measureTextWidth("[OFF]", stateFontId, rowScale),
								  measureTextWidth("< 100% (9999x9999) >", stateFontId, rowScale)});
	for(const auto &opt : options) {
		float rowWidth = measureTextWidth(opt.label, rowFontId, rowScale) + LABEL_STATE_GAP + stateWidth;
		maxContentWidth = std::max(maxContentWidth, rowWidth);
	}
	panelWidth = maxContentWidth + PADDING * 2.0f;

	layoutComputed = true;
}

void CheatHud::renderRows(int screenW, int screenH) {
	computeLayout(screenH);

	float ax, ay;

	// Background panel, sized to exactly fit the title + all rows + padding
	// (see computeLayout), and a highlight bar behind the selected row.
	// Drawn in that order so the (translucent) highlight composites on top
	// of the background rather than the other way around.
	// This is used to show hover and selection
	std::vector<UiRect> panelQuads;
	panelQuads.push_back({PANEL_X, PANEL_Y, panelWidth, panelHeight(),
						   {0.05f, 0.05f, 0.08f, 0.8f}});
	panelQuads.push_back({PANEL_X + PADDING * 0.5f, rowTop(selectedIndex),
						   panelWidth - PADDING, rowHeight,
						   {1.0f, 1.0f, 1.0f, 0.12f}});
	quads->setQuads(panelQuads);

	pixelToAnchor(PANEL_X + PADDING, PANEL_Y + PADDING, screenW, screenH, ax, ay);
	txt->print(ax, ay, "CHEATS (L to close)", TITLE_TEXT_ID, "SS", false, true, false,
			   TAL_LEFT, TRH_LEFT, TRV_TOP,
			   {1.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f},
			   titleScale, titleScale);

	for(int i = 0; i < (int)options.size(); i++) {
		const CheatRow &row = options[i];
		bool selected = (i == selectedIndex);
		float top = rowTop(i);

		// Label, left-aligned.
		pixelToAnchor(PANEL_X + PADDING, top, screenW, screenH, ax, ay);
		glm::vec4 labelColor = selected ? glm::vec4(1.0f, 1.0f, 0.3f, 1.0f)
										 : glm::vec4(0.85f, 0.85f, 0.85f, 1.0f);
		txt->print(ax, ay, row.label, FIRST_ROW_TEXT_ID + i, "SS", false, selected, false,
				   TAL_LEFT, TRH_LEFT, TRV_TOP, labelColor,
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, rowScale, rowScale);

		// State/value, right-aligned to the panel's (padded) right edge --
		// [ON]/[OFF] for a toggle row, the slider's own formatted value (or
		// the default "< NN% >" percentage) for a slider row -- see
		// CheatRow::format.
		std::string stateText;
		glm::vec4 stateColor;
		if(row.toggleValue != nullptr) {
			bool enabled = *row.toggleValue;
			stateText = enabled ? "[ON]" : "[OFF]";
			stateColor = enabled ? glm::vec4(0.3f, 1.0f, 0.3f, 1.0f) : glm::vec4(1.0f, 0.3f, 0.3f, 1.0f);
		} else if(row.format) {
			stateText = row.format(*row.sliderValue);
			stateColor = glm::vec4(0.4f, 0.8f, 1.0f, 1.0f);
		} else {
			char buf[16];
			snprintf(buf, sizeof(buf), "< %d%% >", (int)std::lround(*row.sliderValue * 100.0f));
			stateText = buf;
			stateColor = glm::vec4(0.4f, 0.8f, 1.0f, 1.0f);
		}
		pixelToAnchor(PANEL_X + panelWidth - PADDING, top, screenW, screenH, ax, ay);
		txt->print(ax, ay, stateText, FIRST_ROW_TEXT_ID + (int)options.size() + i,
				   "SS", false, false, false,
				   TAL_RIGHT, TRH_RIGHT, TRV_TOP, stateColor,
				   {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, rowScale, rowScale);
	}
}

void CheatHud::hideRows() {
	txt->removeText(TITLE_TEXT_ID);
	for(int i = 0; i < (int)options.size(); i++) {
		txt->removeText(FIRST_ROW_TEXT_ID + i);
		txt->removeText(FIRST_ROW_TEXT_ID + (int)options.size() + i);
	}
	quads->setQuads({});
}

#endif
