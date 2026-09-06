// ***** CUSTOM *****

// A keyboard/mouse-navigable HUD for flipping cheat flags at runtime. Text
// rows come from TextMaker; the panel and selection highlight from UiQuad,
// since the font atlas has no filled rectangle.
//
// Header-only, implementation gated behind CHEATHUD_IMPLEMENTATION (Libs.cpp).
// Assumes Starter.hpp, TextMaker.hpp and UiQuad.hpp were included first.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// One row: either an on/off toggle bound to a live bool, or a Left/Right
// slider bound to a live float. Exactly one of toggleValue/sliderValue is set.
// The HUD changes the real value in place. onChange is for a slider whose
// effect needs more than a read to take hold (a render-target rebuild, say).
struct CheatRow {
	std::string label;
	bool *toggleValue = nullptr;
	float *sliderValue = nullptr;
	float sliderMin = 0.0f;
	float sliderMax = 1.0f;
	float sliderStep = 0.05f;
	// Fires once per Left/Right press that moved the value -- edge-triggered,
	// so a held key can't spam an expensive rebuild. Unused for a toggle.
	std::function<void()> onChange;
	// Turns *sliderValue into the right-hand text. Default: "< NN% >" of
	// value*100, for a 0..1 fraction; anything else (MSAA level, an exponent)
	// passes its own, e.g. "< 4x >".
	std::function<std::string(float)> format;
};

struct CheatHud {
	void init(TextMaker *txt, UiQuad *quads);
	void addToggle(const std::string &label, bool *value);
	// step: how much one Left/Right press moves *value, clamped to [min, max].
	// onChange/format: see CheatRow.
	void addSlider(const std::string &label, float *value, float min, float max,
				   float step, std::function<void()> onChange = nullptr,
				   std::function<std::string(float)> format = nullptr);
	bool isOpen() const { return open; }

	// Reads input, updates state, re-renders the panel if anything changed.
	// Once per frame from GameLogic(), BEFORE getSixAxis.
	void update(GLFWwindow *window, int screenW, int screenH);

	private:
	// Not owned: the game's TextMaker/UiQuad instances.
	TextMaker *txt = nullptr;
	UiQuad *quads = nullptr;

	std::vector<CheatRow> options;
	int selectedIndex = 0;
	bool open = false;

	// Last screen size renderRows() ran at. renderRows bakes each row's pixel
	// position into a fixed NDC anchor, and TextMaker can't notice a resize on
	// its own, so update() forces a re-render when the size changes.
	int lastScreenW = -1;
	int lastScreenH = -1;

	// Edge-detection: trigger once per physical press, not per frame held.
	bool lKeyWasPressed = false;
	bool upKeyWasPressed = false;
	bool downKeyWasPressed = false;
	bool enterKeyWasPressed = false;
	bool leftMouseWasPressed = false;
	// Adjust the selected row's slider (no-op on a toggle row).
	bool leftArrowKeyWasPressed = false;
	bool rightArrowKeyWasPressed = false;

	// Set whenever the panel needs re-printing (opened/closed, selection moved,
	// a toggle flipped). Avoids rebuilding TextMaker's mesh every idle frame.
	bool dirty = true;

	// Panel layout in pixels, anchored top-left. Hit-testing, the highlight
	// quad and the background all derive from these, so they can't drift.
	static constexpr float PANEL_X = 24.0f;
	static constexpr float PANEL_Y = 24.0f;
	static constexpr float PADDING = 10.0f;          // panel edge to text
	static constexpr float LABEL_STATE_GAP = 20.0f;  // min gap, label to [ON]/[OFF]
	static constexpr float LINE_GAP = 8.0f;          // extra vertical room per row
	// Text scales ASKED for; computeLayout() may shrink to titleScale/rowScale.
	static constexpr float TITLE_SCALE = 1.0f;
	static constexpr float ROW_SCALE = 0.85f;
	// Floor for that shrinking: past this the panel runs off the bottom
	// instead, since unreadable is no better than partly off-screen.
	static constexpr float MIN_FIT_SCALE = 0.4f;

	// Text-block ids for TextMaker. Past the FPS counter's id (1).
	static constexpr int TITLE_TEXT_ID = 100;
	static constexpr int FIRST_ROW_TEXT_ID = 101;

	// Panel width, row heights and the scales actually used, in pixels.
	// Computed from real rendered text size, so the panel fits its content
	// exactly. Lazy via computeLayout(), invalidated on a resize.
	bool layoutComputed = false;
	float panelWidth = 0.0f;
	float titleRowHeight = 0.0f;
	float rowHeight = 0.0f;
	float titleScale = TITLE_SCALE;
	float rowScale = ROW_SCALE;

	// The panel shrinks to fit screenH: the row list grows with every cheat
	// added, and there's no scrolling.
	void computeLayout(int screenH);
	// Pixel size of "s" as TextMaker would render it. fontId must match the
	// (FontFace, Bold, Italic, Small) combo of the print() call.
	float measureTextWidth(const std::string &s, int fontId, float scale) const;
	float measureTextHeight(int fontId, float scale) const;

	// Row rectangle in pixel space (row 0 sits below the title row).
	float rowTop(int i) const { return PANEL_Y + PADDING + titleRowHeight + rowHeight * i; }
	float panelHeight() const { return PADDING * 2.0f + titleRowHeight + rowHeight * (int)options.size(); }

	void renderRows(int screenW, int screenH);
	void hideRows();
	// Top-left pixel coord -> the NDC-ish anchor TextMaker::print expects
	// (mirrors TextMaker::pixelToScr, which is private).
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
	// A resize leaves TextMaker showing rows baked for the old size, so force
	// a re-render (and drop the cached layout, fitted to the old height).
	if(screenW != lastScreenW || screenH != lastScreenH) {
		lastScreenW = screenW;
		lastScreenH = screenH;
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

	// Mouse: hovering a row selects it (shares selectedIndex with the
	// keyboard), a click selects and flips it.
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

	// fontIds matching TextMaker::print's formula: "SS" family is 8, Bold
	// adds 2. Rows measure Bold so the layout isn't too narrow once selected.
	const int titleFontId = 8 + 2;
	const int rowFontId = 8 + 2;
	const int stateFontId = 8;

	// Heights at the asked-for scales first: whether they fit is the question.
	// PANEL_Y margin top and bottom.
	float titleH = measureTextHeight(titleFontId, TITLE_SCALE) + LINE_GAP;
	float rowH = measureTextHeight(rowFontId, ROW_SCALE) + LINE_GAP;
	float contentH = titleH + rowH * (float)options.size();
	float availableH = (float)screenH - PANEL_Y * 2.0f - PADDING * 2.0f;

	// One factor for text and row heights alike, so the panel keeps its
	// proportions. PADDING is a margin, left alone.
	float fit = 1.0f;
	if(contentH > availableH && contentH > 0.0f) {
		fit = std::max(availableH / contentH, MIN_FIT_SCALE);
	}
	titleScale = TITLE_SCALE * fit;
	rowScale = ROW_SCALE * fit;
	titleRowHeight = titleH * fit;
	rowHeight = rowH * fit;

	// Width at the fitted scales, so a shrunk panel is narrower too.
	float maxContentWidth = measureTextWidth("CHEATS (L to close)", titleFontId, titleScale);
	// "< 100% (9999x9999) >" is a generous stand-in for a slider's value text,
	// so the panel width doesn't shift as a slider is adjusted or a format()
	// changes length -- only on a resize.
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

	// Background panel sized to fit title + rows + padding, then a highlight
	// bar behind the selected row -- in that order so the translucent
	// highlight composites on top.
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

		// State/value, right-aligned: [ON]/[OFF] for a toggle, the formatted
		// value (or default "< NN% >") for a slider. See CheatRow::format.
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
