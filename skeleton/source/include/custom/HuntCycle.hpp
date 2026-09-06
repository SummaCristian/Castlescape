// ***** CUSTOM *****

// The hunt cycle: the clock the whole game is built around. Three phases,
// looping forever:
//
//   Calm      torches burn orange, ghosts walk their patrols and ignore the player.
//   Warning   a short telegraph: the flames bleed toward the hunt colour and
//             pulse, but the ghosts are still patrolling. So being hunted is
//             never a surprise -- a couple of seconds to reach a door first.
//   Hunt      flames fully changed, every ghost drops its patrol and comes for
//             the player. Touching one ends the run.
//
// After a hunt the colour eases back to orange over `recoverDuration` while the
// phase is already Calm: the danger ends the instant the phase flips, the fade
// is only cosmetic. That's why the blend is its own value, not derived from
// the phase.
//
// This class owns only the clock and that blend -- it never touches a torch, a
// light or a ghost. main.cpp asks it "what colour" and "hunting?" and does the
// work, same split as SceneLights.
//
// Header-only, implementation gated behind HUNTCYCLE_IMPLEMENTATION (Libs.cpp).

#include <algorithm>
#include <cmath>
#include <string>

enum class HuntPhase {
	Calm,
	Warning,
	Hunt
};

// Everything tunable, all overridable from gameplay.json's "hunt" block so
// the pacing can change without a recompile.
struct HuntConfig {
	float calmDuration = 45.0f;     // quiet between hunts; most of the game, when the player explores
	float warningDuration = 3.0f;   // the telegraph: alarming, but long enough to reach a door
	float huntDuration = 18.0f;     // how long the ghosts chase; an interruption, not the norm
	float recoverDuration = 4.0f;   // flame fade-back after a hunt; not a phase, see the header

	// Violet: reads as "wrong" against stone and firelight without going so
	// dark the room stops being navigable.
	glm::vec3 huntColor = glm::vec3(0.42f, 0.10f, 1.0f);
	// Hunt light output vs calm. Below 1 so the room is harder to read, but not
	// much -- a player who can't see the walls can't dodge either.
	float huntLightScale = 0.8f;

	float warningPulseDepth = 0.35f; // extra Warning flicker, fraction of flame brightness
	float warningPulseHz = 4.0f;     // pulses/sec, fast enough to read as an alarm
};

class HuntCycle {
	public:
	// Reads the "hunt" object out of an already-parsed gameplay.json. Missing
	// keys keep their default, so the file can override one number and leave
	// the rest alone.
	void init(const nlohmann::json &js);

	// Advances the clock. deltaT is the same clamped value the rest of
	// GameLogic() runs on.
	void update(float deltaT);

	// Puts the cycle back to the start of a fresh Calm phase, colour and all.
	// Called when a run restarts.
	void reset();

	HuntPhase phase() const { return current; }
	// The one question the ghosts ask. Only true in Hunt: during Warning they
	// are still patrolling, which is the whole point of Warning.
	bool hunting() const { return current == HuntPhase::Hunt; }

	// True for the one frame the phase changed; phase() gives the new one. The
	// hook a music system would hang off.
	bool phaseJustChanged() const { return justChanged; }

	// Seconds left in the current phase. Feeds the on-screen warning countdown.
	float timeLeftInPhase() const { return std::max(0.0f, phaseTimer); }

	// 0 = fully calm/orange, 1 = fully hunting/violet. Ramps up across Warning,
	// sits at 1 through Hunt, eases back down across recoverDuration.
	float colorBlend() const { return blend; }

	// The colour a flame authored as `base` should be now, and the multiplier
	// its light should carry. Both pure functions of colorBlend(), kept here
	// so the blend curve lives in one place.
	glm::vec3 flameColor(const glm::vec3 &base) const;
	float lightScale() const;

	// Warning-pulse brightness multiplier, 1.0 outside Warning. On TOP of the
	// flame's own flicker: the fire keeps behaving like fire, it just throbs.
	float warningPulse() const;

	// Skips the rest of the current phase into Warning. Wired to the cheat menu.
	void triggerHunt();

	// Holds the cycle in Hunt while set. A bool* the cheat HUD writes in place.
	bool forceHunt = false;

	const HuntConfig &config() const { return cfg; }

	private:
	HuntConfig cfg;
	HuntPhase current = HuntPhase::Calm;
	// Counts DOWN through the current phase.
	float phaseTimer = 0.0f;
	float blend = 0.0f;
	// Free-running seconds, only used to drive the Warning pulse's sine.
	float elapsed = 0.0f;
	bool justChanged = false;
	// Latches forceHunt's previous value so turning it on is an edge (enter
	// Hunt now) rather than a level that would re-enter Hunt every frame.
	bool forceHuntWasSet = false;

	void enter(HuntPhase p);

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);
};

#ifdef HUNTCYCLE_IMPLEMENTATION

glm::vec3 HuntCycle::readVec3(const nlohmann::json &js, const glm::vec3 &fallback) {
	if(js.size() != 3) {
		std::cout << "HuntCycle: expected 3 values, got " << js.size()
				  << ", using the default\n";
		return fallback;
	}
	// Explicit get<float>(), same reason as SceneLights::readVec3: glm::vec3
	// has several 3-arg constructors and the implicit conversion can pick the
	// wrong one.
	return glm::vec3(js[0].get<float>(), js[1].get<float>(), js[2].get<float>());
}

void HuntCycle::init(const nlohmann::json &js) {
	if(js.contains("calmDuration"))      cfg.calmDuration = js["calmDuration"].get<float>();
	if(js.contains("warningDuration"))   cfg.warningDuration = js["warningDuration"].get<float>();
	if(js.contains("huntDuration"))      cfg.huntDuration = js["huntDuration"].get<float>();
	if(js.contains("recoverDuration"))   cfg.recoverDuration = js["recoverDuration"].get<float>();
	if(js.contains("huntColor"))         cfg.huntColor = readVec3(js["huntColor"], cfg.huntColor);
	if(js.contains("huntLightScale"))    cfg.huntLightScale = js["huntLightScale"].get<float>();
	if(js.contains("warningPulseDepth")) cfg.warningPulseDepth = js["warningPulseDepth"].get<float>();
	if(js.contains("warningPulseHz"))    cfg.warningPulseHz = js["warningPulseHz"].get<float>();

	// A zero or negative duration would leave the phase it belongs to
	// unexitable (the timer never counts down past 0 into the next phase) or,
	// worse, divide by zero in the blend ramp below. Clamped rather than
	// rejected: someone writing 0 clearly meant "as short as possible".
	cfg.calmDuration = std::max(0.1f, cfg.calmDuration);
	cfg.warningDuration = std::max(0.1f, cfg.warningDuration);
	cfg.huntDuration = std::max(0.1f, cfg.huntDuration);
	cfg.recoverDuration = std::max(0.1f, cfg.recoverDuration);

	reset();
}

void HuntCycle::reset() {
	current = HuntPhase::Calm;
	phaseTimer = cfg.calmDuration;
	blend = 0.0f;
	elapsed = 0.0f;
	justChanged = false;
	// forceHunt is NOT cleared: it's a cheat-menu toggle the HUD holds a
	// pointer to, and a restart silently switching a menu row off (while the
	// row still reads ON, since nothing tells the HUD to re-render) is worse
	// than either honouring it or ignoring it. Clearing the latch instead means
	// a toggle that's still on reads as a fresh edge on the next update() and
	// puts the cycle straight back into Hunt, which is what leaving it on asked
	// for.
	forceHuntWasSet = false;
}

void HuntCycle::enter(HuntPhase p) {
	current = p;
	justChanged = true;
	switch(p) {
		case HuntPhase::Calm:    phaseTimer = cfg.calmDuration; break;
		case HuntPhase::Warning: phaseTimer = cfg.warningDuration; break;
		case HuntPhase::Hunt:    phaseTimer = cfg.huntDuration; break;
	}
}

void HuntCycle::triggerHunt() {
	if(current == HuntPhase::Calm) {
		enter(HuntPhase::Warning);
	}
}

void HuntCycle::update(float deltaT) {
	justChanged = false;
	elapsed += deltaT;

	// The cheat toggle, handled before the clock so the phase it forces is
	// what the rest of this function then sees. Turning it ON jumps straight
	// into Hunt with no Warning -- it's a debug switch, the telegraph is for
	// the player. Turning it OFF hands control back to the normal clock by
	// ending the hunt immediately.
	if(forceHunt != forceHuntWasSet) {
		enter(forceHunt ? HuntPhase::Hunt : HuntPhase::Calm);
		forceHuntWasSet = forceHunt;
	}

	if(forceHunt) {
		// Held at full danger: don't let the timer run the phase out from
		// under the toggle.
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

	// The colour blend, ramped at a rate that makes it arrive exactly when the
	// phase does: 1/warningDuration on the way up means it hits 1.0 on the
	// frame Warning ends, so the flames are fully violet the moment the ghosts
	// start moving. The fall uses recoverDuration instead, which is
	// deliberately unrelated to any phase length -- see the file header.
	float target = (current == HuntPhase::Calm) ? 0.0f : 1.0f;
	float rate = (target > blend) ? (1.0f / cfg.warningDuration)
								  : (1.0f / cfg.recoverDuration);
	float step = rate * deltaT;
	if(blend < target)      blend = std::min(blend + step, target);
	else if(blend > target) blend = std::max(blend - step, target);
}

glm::vec3 HuntCycle::flameColor(const glm::vec3 &base) const {
	return glm::mix(base, cfg.huntColor, blend);
}

float HuntCycle::lightScale() const {
	return glm::mix(1.0f, cfg.huntLightScale, blend);
}

float HuntCycle::warningPulse() const {
	if(current != HuntPhase::Warning) {
		return 1.0f;
	}
	// Sine kept strictly at or below 1.0 rather than swinging either side of
	// it: the pulse should read as the fire being repeatedly snuffed and
	// recovering, not as it getting brighter than a torch ever is.
	float s = 0.5f + 0.5f * std::sin(elapsed * cfg.warningPulseHz * 2.0f * (float)M_PI);
	return 1.0f - cfg.warningPulseDepth * (1.0f - s);
}

#endif
