// ***** CUSTOM *****

// The hunt cycle clock. Three phases, looping: Calm (patrols, ignore player),
// Warning (telegraph: flames bleed/pulse, still patrolling), Hunt (all ghosts
// chase, touch ends the run). Colour eases back over recoverDuration AFTER the
// phase returns to Calm -- danger ends on the phase flip, fade is cosmetic only.
// Owns only the clock and the colour blend; never touches a torch/light/ghost.
// Header-only, gated behind HUNTCYCLE_IMPLEMENTATION (Libs.cpp).

#include <algorithm>
#include <cmath>
#include <string>

enum class HuntPhase {
	Calm,
	Warning,
	Hunt
};

// Tunable, overridable from gameplay.json's "hunt" block.
struct HuntConfig {
	float calmDuration = 45.0f;     // quiet between hunts
	float warningDuration = 3.0f;   // telegraph before a hunt
	float huntDuration = 18.0f;     // how long ghosts chase
	float recoverDuration = 4.0f;   // flame fade-back after a hunt (not a phase, see header)

	glm::vec3 huntColor = glm::vec3(0.42f, 0.10f, 1.0f); // violet: reads "wrong" without killing visibility
	// Hunt light output vs calm; below 1 but not much -- can't dodge what you can't see.
	float huntLightScale = 0.8f;

	float warningPulseDepth = 0.35f; // extra Warning flicker, fraction of flame brightness
	float warningPulseHz = 4.0f;     // pulses/sec
};

class HuntCycle {
	public:
	// Reads "hunt" from an already-parsed gameplay.json; missing keys keep their default.
	void init(const nlohmann::json &js);

	// Advances the clock. deltaT: same clamped value the rest of GameLogic() uses.
	void update(float deltaT);

	// Resets to a fresh Calm phase, colour included. Called on run restart.
	void reset();

	HuntPhase phase() const { return current; }
	// Only true in Hunt: Warning still has ghosts patrolling.
	bool hunting() const { return current == HuntPhase::Hunt; }

	// True for the one frame the phase changed; phase() gives the new one.
	bool phaseJustChanged() const { return justChanged; }

	// Seconds left in the current phase; feeds the warning countdown.
	float timeLeftInPhase() const { return std::max(0.0f, phaseTimer); }

	// 0 = calm/orange, 1 = hunting/violet. Ramps across Warning, holds through Hunt,
	// eases down across recoverDuration.
	float colorBlend() const { return blend; }

	// Pure functions of colorBlend(), so the blend curve lives in one place.
	glm::vec3 flameColor(const glm::vec3 &base) const;
	float lightScale() const;

	// Warning-pulse brightness multiplier, 1.0 outside Warning. On top of the flame's own flicker.
	float warningPulse() const;

	// Skips into Warning. Wired to the debug menu.
	void triggerHunt();

	// Holds the cycle in Hunt while set. Written in place by the debug HUD.
	bool forceHunt = false;

	const HuntConfig &config() const { return cfg; }

	private:
	HuntConfig cfg;
	HuntPhase current = HuntPhase::Calm;
	float phaseTimer = 0.0f;	// counts down through the current phase
	float blend = 0.0f;
	float elapsed = 0.0f;		// free-running seconds, drives the Warning pulse sine
	bool justChanged = false;
	// Latches forceHunt so turning it on is an edge, not a level re-entering Hunt every frame.
	bool forceHuntWasSet = false;

	void enter(HuntPhase p);

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);
};

#ifdef HUNTCYCLE_IMPLEMENTATION

// JSON array -> vec3.
glm::vec3 HuntCycle::readVec3(const nlohmann::json &js, const glm::vec3 &fallback) {
	if(js.size() != 3) {
		std::cout << "HuntCycle: expected 3 values, got " << js.size()
				  << ", using the default\n";
		return fallback;
	}
	// Explicit get<float>(): glm::vec3 has multiple 3-arg ctors, implicit conversion can pick the wrong one.
	return glm::vec3(js[0].get<float>(), js[1].get<float>(), js[2].get<float>());
}

// Load config + reset.
void HuntCycle::init(const nlohmann::json &js) {
	if(js.contains("calmDuration"))      cfg.calmDuration = js["calmDuration"].get<float>();
	if(js.contains("warningDuration"))   cfg.warningDuration = js["warningDuration"].get<float>();
	if(js.contains("huntDuration"))      cfg.huntDuration = js["huntDuration"].get<float>();
	if(js.contains("recoverDuration"))   cfg.recoverDuration = js["recoverDuration"].get<float>();
	if(js.contains("huntColor"))         cfg.huntColor = readVec3(js["huntColor"], cfg.huntColor);
	if(js.contains("huntLightScale"))    cfg.huntLightScale = js["huntLightScale"].get<float>();
	if(js.contains("warningPulseDepth")) cfg.warningPulseDepth = js["warningPulseDepth"].get<float>();
	if(js.contains("warningPulseHz"))    cfg.warningPulseHz = js["warningPulseHz"].get<float>();

	// Clamped, not rejected: 0 or negative would either never exit the phase or
	// divide by zero in the blend ramp below.
	cfg.calmDuration = std::max(0.1f, cfg.calmDuration);
	cfg.warningDuration = std::max(0.1f, cfg.warningDuration);
	cfg.huntDuration = std::max(0.1f, cfg.huntDuration);
	cfg.recoverDuration = std::max(0.1f, cfg.recoverDuration);

	reset();
}

// Back to Calm, blend 0.
void HuntCycle::reset() {
	current = HuntPhase::Calm;
	phaseTimer = cfg.calmDuration;
	blend = 0.0f;
	elapsed = 0.0f;
	justChanged = false;
	// forceHunt itself is NOT cleared (debug HUD holds a pointer to it); clearing
	// only the latch makes an already-on toggle read as a fresh edge next update().
	forceHuntWasSet = false;
}

// Switch phase, reset its timer.
void HuntCycle::enter(HuntPhase p) {
	current = p;
	justChanged = true;
	switch(p) {
		case HuntPhase::Calm:    phaseTimer = cfg.calmDuration; break;
		case HuntPhase::Warning: phaseTimer = cfg.warningDuration; break;
		case HuntPhase::Hunt:    phaseTimer = cfg.huntDuration; break;
	}
}

// Force-skip Calm -> Warning.
void HuntCycle::triggerHunt() {
	if(current == HuntPhase::Calm) {
		enter(HuntPhase::Warning);
	}
}

// Per-frame tick: debug toggle, timer, blend ramp.
void HuntCycle::update(float deltaT) {
	justChanged = false;
	elapsed += deltaT;

	// Debug toggle, handled before the clock. ON: jumps straight to Hunt, no
	// Warning (debug switch). OFF: ends the hunt immediately, back to Calm.
	if(forceHunt != forceHuntWasSet) {
		enter(forceHunt ? HuntPhase::Hunt : HuntPhase::Calm);
		forceHuntWasSet = forceHunt;
	}

	if(forceHunt) {
		// Held at full danger: don't let the timer run the phase out.
		phaseTimer = cfg.huntDuration;
	} else {
		phaseTimer -= deltaT;
		if(phaseTimer <= 0.0f) {
			switch(current) {
				case HuntPhase::Calm:    enter(HuntPhase::Warning); break;
				case HuntPhase::Warning: enter(HuntPhase::Hunt); break;
				case HuntPhase::Hunt:    enter(HuntPhase::Calm); break;
			}
		}
	}

	// Ramp rate makes the blend arrive exactly on the phase change: 1/warningDuration
	// up means it hits 1.0 the frame Warning ends. Fall uses recoverDuration instead
	// (deliberately unrelated to any phase length, see header).
	float target = (current == HuntPhase::Calm) ? 0.0f : 1.0f;
	float rate = (target > blend) ? (1.0f / cfg.warningDuration)
								  : (1.0f / cfg.recoverDuration);
	float step = rate * deltaT;
	if(blend < target)      blend = std::min(blend + step, target);
	else if(blend > target) blend = std::max(blend - step, target);
}

// base -> huntColor by blend.
glm::vec3 HuntCycle::flameColor(const glm::vec3 &base) const {
	return glm::mix(base, cfg.huntColor, blend);
}

// 1.0 -> huntLightScale by blend.
float HuntCycle::lightScale() const {
	return glm::mix(1.0f, cfg.huntLightScale, blend);
}

// Warning-only flicker, sine-based.
float HuntCycle::warningPulse() const {
	if(current != HuntPhase::Warning) {
		return 1.0f;
	}
	// Kept at or below 1.0: reads as the fire being snuffed and recovering, never brighter.
	float s = 0.5f + 0.5f * std::sin(elapsed * cfg.warningPulseHz * 2.0f * (float)M_PI);
	return 1.0f - cfg.warningPulseDepth * (1.0f - s);
}

#endif
