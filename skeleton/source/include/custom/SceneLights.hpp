// ***** CUSTOM *****

// Owns the scene's light sources.
//
// How it fits in:
//   at startup   main.cpp calls init(), which reads assets/scenes/lights.json
//                and turns each entry into a LightData
//   every frame  main.cpp calls update(), gets the list back, copies it into
//                the global uniform buffer, and the GPU sends it to the shader
//
// The shader never reads a file, and never sees this class. This is the only
// path from lights.json to the screen.
//
// The three light types, all from L09:
//   direct  infinitely far away, so one direction for the whole scene, and no
//           fading with distance. The sun.
//   point   sits at a position and shines in all directions, fading with
//           distance. A lamp or a candle.
//   spot    a point light restricted to a cone. Has an aim and two angles.
//
// The formulas are in notes.md.
//
// A light's position is either explicit world coordinates or "instance" plus
// "offset", which hangs it off a scene.json instance so it survives moving that
// instance. Same idea as the authored collision boxes.
//
// Header-only module like the rest of custom/, implementation gated behind
// SCENELIGHTS_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// modules/Starter.hpp and modules/Scene.hpp are already included.

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// MAX_LIGHTS and LIGHT_*, the same file CookTorrance.frag includes.
#include "custom/LightConstants.glsl"

// Matches the GLSL struct field for field. Each float after a vec3 fills the 4
// bytes std140 would otherwise pad, so the struct is 64 bytes in both languages.
struct LightData {
	alignas(16) glm::vec3 pos;		// point/spot only
	float g;						// decay reference distance
	alignas(16) glm::vec3 dir;		// direct: travel direction. spot: aim
	float beta;						// decay exponent
	alignas(16) glm::vec3 color;	// l, the emitted color
	float cosIn;					// spot: cosine of the half inner angle
	float cosOut;					// spot: cosine of the half outer angle
	int type;						// LIGHT_DIRECT / LIGHT_POINT / LIGHT_SPOT
	// -1: doesn't cast a shadow. Else an index into the shadow array that
	// matches this light's TYPE, assigned in declaration order by init()
	// below from lights.json's "castsShadow" flag:
	//   LIGHT_DIRECT / LIGHT_SPOT  -> index into the 2D depth maps
	//                                 (NUM_SHADOW_MAPS_2D, shadowLightSpace2D
	//                                 in main.cpp)
	//   LIGHT_POINT                -> index into the cube shadow maps
	//                                 (NUM_SHADOW_CUBES, one CubeShadowMap
	//                                 per torch in main.cpp)
	// One slot each, unlike the old two-perspective-map workaround: a real
	// cube map answers in every direction on its own (see shadowFactor() in
	// CookTorrance.frag).
	//
	// Fits in the same 16-byte slot as cosOut+type without changing that
	// slot's size: std140 pads a struct used in an array (this one, via
	// Light lights[MAX_LIGHTS] in the shader) up to a multiple of 16 bytes
	// regardless, so cosOut(4)+type(4) was already sharing its 16-byte slot
	// with 8 bytes of otherwise-wasted padding. shadowIndex spends 4 of it.
	int shadowIndex;
};

// Hemispheric ambient, E07 s.47-54: the scene's indirect lighting, two colors
// blended by which way a surface faces.
//
// Really the scene's stand-in ENVIRONMENT, which is the more useful way to read
// it: besides being the diffuse indirect term, these two colors are what
// metalAmbient() reflects in place of the prefiltered cube map a real split-sum
// IBL would sample. A metal has no diffuse lobe, so they are most of what one
// looks like away from a direct highlight.
//
// Both default to BLACK, i.e. every level authors its own: the right answer is
// "what is around this room", and there is no default that is right for both a
// sealed dungeon and an open courtyard. Getting it wrong by omission is how a
// corridor ends up lit by a sky. lights.json authors the interior values this
// scene uses, and documents what an exterior one would put here instead.
struct AmbientLight {
	glm::vec3 upper = glm::vec3(0.0f);				// sky color
	glm::vec3 lower = glm::vec3(0.0f);				// ground color
	glm::vec3 dir = glm::vec3(0.0f, 1.0f, 0.0f);	// which way "up" blends

	// Share of indirect light, 0..1. CookTorrance.frag blends rather than sums:
	// ambient takes `weight`, direct light keeps (1 - weight). So this is the
	// brightness and upper/lower above are only the two COLORS -- which is why
	// an outdoor level authors them near 1, not near 0.1 as when they summed.
	//
	// Still meaningful with the hemisphere black: `bounce` below spends the
	// same share, so this sizes the torches' indirect contribution.
	//
	// A material can override it per model (Material::ambientWeight), and that
	// is the whole point: an enclosed room and an open courtyard need
	// different values in the same frame.
	//
	// The default is the INDOOR value, matching lights.json: the game is played
	// inside the dungeon, so a model that names no weight is far more likely to
	// be in a sealed room than under the sky, and should not be lit as if a
	// ceiling above it did nothing.
	float weight = 0.05f;

	// How much of each POINT/SPOT light's radiance comes back as indirect
	// light, 0..1. The second half of the indirect term, and the half that
	// actually belongs to this scene.
	//
	// The hemispheric model above answers "what colour is the light arriving
	// from the sky and from the ground", which is the right question outdoors
	// and a meaningless one in a sealed corridor: there is no sky down there,
	// and what little indirect light exists is torchlight that has bounced off
	// a wall. Authoring that as a constant is what left every dungeon surface
	// under a fixed cool wash no torch could warm and no distance could dim --
	// upper/lower blended at 0.5 by Material::interiorAmbient come out
	// (0.425, 0.4625, 0.60), a cold grey-blue laid over warm firelight.
	//
	// So the point lights now feed the indirect term too. Same decay and same
	// colour as their direct contribution -- including the flicker -- but with
	// no shadow test and a soft wrap instead of a clamped cosine, because that
	// is what one bounce off a rough stone wall does to light: it reaches
	// round corners, it has no terminator, and it dies with distance from the
	// flame that ultimately produced it.
	//
	// Not physically derived. It is one number standing in for the albedo of
	// an average wall times the solid angle it subtends, in the same spirit as
	// the hemisphere above being two authored colours: what the assets allow
	// short of the baked AO (or a real GI pass) neither of them can have.
	//
	// This is a SHARE of the same indirect bucket `weight` sizes, not an
	// addition on top of it, so turning the Ambient Light cheat off still
	// removes all of it -- see CookTorrance.frag's blend.
	float bounce = 0.35f;
};

class SceneLights {
	public:
	// Must run after Scene::init: an "instance" reference needs its world matrix.
	void init(Scene *SC, const std::string &file);

	// Advances the animated lights and returns the list to upload. Only the
	// lights whose type is enabled below come back, so lights.json stays the
	// single source of truth and the switches never edit it.
	const std::vector<LightData> &update(float deltaT);

	// How many lights the last update() handed over, i.e. after the type
	// switches dropped what they drop. The authored total is what init()
	// prints at startup.
	int count() const { return (int)activeLights.size(); }

	// The full authored list, unfiltered by the enable switches below and
	// without waiting for an update() tick. main.cpp uses this once at
	// startup to find the shadow-casting lights (shadowIndex >= 0) and their
	// positions/directions, to build the light-space matrices shadow
	// rendering needs -- those matrices don't change frame to frame (the sun
	// and the torches are static), so there is no need to go through the
	// per-frame activeLights path just to read them.
	const std::vector<LightData> &all() const { return lights; }

	// Not animated, so it skips update(). Returned as authored: the
	// ambientEnabled switch below is applied in the shader, not here.
	AmbientLight ambient() const;

	// Debug switches, wired to the cheat menu in main.cpp, which flips these
	// bools in place. Public because that is the whole interface: the HUD holds
	// a bool* and there is nothing to recompute when one changes.
	//
	// One per light TYPE rather than per light: the point is answering "is this
	// the sun or a lantern doing that?", and the scene has one sun, two matched
	// lanterns and one spot, so per-light switches would only add rows.
	bool directEnabled = true;	// the sun
	bool pointEnabled = true;	// the torches and candles
	bool spotEnabled = true;	// none authored today, kept for completeness
	// The whole indirect term, hemisphere AND per-light bounce, since the two
	// share one bucket. Off, the only light is what the sources above put there
	// directly -- which is how you tell an unlit surface from a merely dim one.
	// main.cpp turns it into the shader's LIGHT_DEBUG_NO_AMBIENT.
	bool ambientEnabled = true;

	// Forces an orbit onto the directional lights that were authored static
	// (orbitSpeed 0, which is every one of them right now, see lights.json).
	// Sweeping the sun around is the fastest way to see how the whole scene
	// reacts to an incidence angle, so it is worth a switch even though the
	// authored scene deliberately parks the sun. Lights WITH an authored
	// orbitSpeed ignore this and keep running at their own speed either way.
	bool orbitOverride = false;

	private:
	std::vector<LightData> lights;
	// Filtered copy of `lights` handed to the caller, rebuilt every update().
	// A member rather than a local so update() can keep returning a reference.
	std::vector<LightData> activeLights;
	AmbientLight ambientLight;

	// Animation state, index-matched with `lights`. Zero for anything static.
	std::vector<float> orbitSpeed;	// degrees per second around world Y
	std::vector<glm::vec3> baseDir;	// direction before any rotation

	// Flame flicker (lights.json "flicker"), index-matched with `lights` too.
	std::vector<glm::vec3> baseColor;	// color before flicker scales it
	std::vector<bool> flickerEnabled;
	std::vector<float> flickerStrength;	// how far the scale swings from 1.0
	// Per-light stagger so authored-identical torches don't flicker in sync.
	// The golden-angle-ish step (index * 2.399963, radians) needs no RNG and
	// still lands the handful of lights this scene has on visibly different
	// points of the sine sum.
	std::vector<float> flickerPhase;
	float orbitAngle = 0.0f;

	// Degrees per second used by orbitOverride. Far faster than a plausible
	// day/night cycle (a full turn every 24s): this is meant for looking at the
	// scene from every sun angle in a few seconds, not for looking natural.
	static constexpr float DEBUG_ORBIT_SPEED = -15.0f;
	// Angle for the override only, so switching it on starts the sweep from the
	// authored direction instead of jumping to wherever a shared clock had got
	// to, and switching it off puts the sun back where lights.json wants it.
	float debugOrbitAngle = 0.0f;

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);

	// Next shadowIndex to hand out, incremented once per "castsShadow": true
	// entry in declaration order -- one counter per shadow ARRAY (2D depth
	// maps vs. point-light cube maps, see LightData::shadowIndex), since a
	// direct/spot light and a point light no longer share the same array.
	// Not reset after init() -- there is only ever one pass over lights.json.
	int nextShadowIndex2D = 0;
	int nextShadowIndexCube = 0;
};

#ifdef SCENELIGHTS_IMPLEMENTATION

glm::vec3 SceneLights::readVec3(const nlohmann::json &js, const glm::vec3 &fallback) {
	if(js.size() != 3) {
		std::cout << "SceneLights: expected 3 values, got " << js.size()
				  << ", using the default\n";
		return fallback;
	}
	// Explicit get<float>(), as in SceneMaterials: glm::vec3 has several 3-arg
	// constructors and the implicit json conversion can pick the wrong one.
	return glm::vec3(js[0].get<float>(), js[1].get<float>(), js[2].get<float>());
}

void SceneLights::init(Scene *SC, const std::string &file) {
	std::ifstream ifs(file);
	if(!ifs.is_open()) {
		std::cout << "SceneLights: '" << file << "' not found, scene has no lights\n";
		return;
	}

	// ignore_comments, like materials.json: these numbers need their notes.
	nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);
	if(js.contains("ambient")) {
		const nlohmann::json &a = js["ambient"];
		if(a.contains("upper"))     ambientLight.upper = readVec3(a["upper"], ambientLight.upper);
		if(a.contains("lower"))     ambientLight.lower = readVec3(a["lower"], ambientLight.lower);
		if(a.contains("direction")) ambientLight.dir = glm::normalize(readVec3(a["direction"], ambientLight.dir));
		// Clamped, not just read: above 1 the blend would take more than the
		// whole frame and leave the direct term negative.
		if(a.contains("weight")) ambientLight.weight = glm::clamp(a["weight"].get<float>(), 0.0f, 1.0f);
		// Clamped for a different reason than "weight" above: nothing breaks
		// past 1, it just claims a wall returns more light than hit it, which
		// is the one thing a bounce factor must not do.
		if(a.contains("bounce")) ambientLight.bounce = glm::clamp(a["bounce"].get<float>(), 0.0f, 1.0f);
	}

	if(!js.contains("lights")) {
		std::cout << "SceneLights: '" << file << "' has no \"lights\" array\n";
		return;
	}

	for(const auto &l : js["lights"]) {
		if((int)lights.size() >= MAX_LIGHTS) {
			std::cout << "SceneLights: more than " << MAX_LIGHTS
					  << " lights, the rest are ignored\n";
			break;
		}

		LightData L{};
		L.pos = glm::vec3(0.0f);
		L.dir = glm::vec3(0.0f, -1.0f, 0.0f);
		L.color = glm::vec3(1.0f);
		L.g = 1.0f;
		L.beta = 1.0f;
		L.cosIn = 1.0f;
		L.cosOut = 0.0f;
		L.shadowIndex = -1;

		const std::string type = l.value("type", std::string("point"));
		if(type == "direct")     L.type = LIGHT_DIRECT;
		else if(type == "point") L.type = LIGHT_POINT;
		else if(type == "spot")  L.type = LIGHT_SPOT;
		else {
			std::cout << "SceneLights: unknown type '" << type << "', skipped\n";
			continue;
		}

		if(l.contains("color"))     L.color = readVec3(l["color"], L.color);
		if(l.contains("direction")) L.dir   = glm::normalize(readVec3(l["direction"], L.dir));
		if(l.contains("g"))         L.g     = l["g"].get<float>();
		if(l.contains("beta"))      L.beta  = l["beta"].get<float>();

		// Explicit world coordinates, or an instance to hang off.
		if(l.contains("position")) {
			L.pos = readVec3(l["position"], L.pos);
		} else if(l.contains("instance")) {
			const std::string instanceId = l["instance"];
			auto it = SC->InstanceIds.find(instanceId);
			if(it == SC->InstanceIds.end()) {
				std::cout << "SceneLights: scene has no instance '" << instanceId
						  << "', light skipped\n";
				continue;
			}
			// Offset is in WORLD units, not model-local: these models are Z-up
			// and rotated by the scene, and world units are what the Show
			// Coordinates overlay reads out, which is how these get tuned.
			glm::vec3 origin = glm::vec3(SC->I[it->second]->Wm[3]);
			glm::vec3 offset = l.contains("offset") ? readVec3(l["offset"], glm::vec3(0.0f))
													: glm::vec3(0.0f);
			L.pos = origin + offset;
		} else if(L.type != LIGHT_DIRECT) {
			std::cout << "SceneLights: a " << type
					  << " light needs \"position\" or \"instance\", skipped\n";
			continue;
		}

		// Authored as FULL angles in degrees, the way a beam width is normally
		// described. The shader wants the cosine of the half angle (L09 s.29),
		// converted here once instead of per fragment.
		if(L.type == LIGHT_SPOT) {
			float innerDeg = l.value("innerAngle", 30.0f);
			float outerDeg = l.value("outerAngle", 45.0f);
			if(outerDeg < innerDeg) {
				std::cout << "SceneLights: outerAngle is smaller than innerAngle, swapped\n";
				std::swap(innerDeg, outerDeg);
			}
			L.cosIn  = std::cos(glm::radians(innerDeg * 0.5f));
			L.cosOut = std::cos(glm::radians(outerDeg * 0.5f));
		}

		if(l.value("castsShadow", false)) {
			// A point light draws from the CUBE array (one real 6-face cube
			// map per torch), everything else from the 2D array: a direct
			// light (the sun) shadows through a single orthographic box that
			// already covers the whole scene, and a spot only emits inside its
			// cone, so for both of those a single 2D depth map is the whole
			// story.
			if(L.type == LIGHT_POINT) {
				if(nextShadowIndexCube >= NUM_SHADOW_CUBES) {
					std::cout << "SceneLights: out of cube shadow map slots ("
							  << NUM_SHADOW_CUBES << "), '" << l.value("id", std::string("?"))
							  << "' renders unshadowed\n";
					L.shadowIndex = -1;
				} else {
					L.shadowIndex = nextShadowIndexCube++;
				}
			} else {
				if(nextShadowIndex2D >= NUM_SHADOW_MAPS_2D) {
					std::cout << "SceneLights: out of 2D shadow map slots ("
							  << NUM_SHADOW_MAPS_2D << "), '" << l.value("id", std::string("?"))
							  << "' renders unshadowed\n";
					L.shadowIndex = -1;
				} else {
					L.shadowIndex = nextShadowIndex2D++;
				}
			}
		}

		lights.push_back(L);
		baseDir.push_back(L.dir);
		orbitSpeed.push_back(l.value("orbitSpeed", 0.0f));

		baseColor.push_back(L.color);
		flickerEnabled.push_back(l.value("flicker", false));
		flickerStrength.push_back(l.value("flickerStrength", 0.35f));
		flickerPhase.push_back((float)(lights.size() - 1) * 2.399963f);
	}

	std::cout << "SceneLights: " << lights.size() << " lights loaded\n";
}

AmbientLight SceneLights::ambient() const {
	// As authored, cheat or no cheat. Blacking the colors here used to be the
	// switch, and was only half the job: it missed the per-light bounce and any
	// model overriding the share in materials.json. It now lives downstream of
	// both, in the shader's ambientShare() (LIGHT_DEBUG_NO_AMBIENT).
	return ambientLight;
}

const std::vector<LightData> &SceneLights::update(float deltaT) {
	orbitAngle += deltaT;
	// Only ticks while the override is on, and rewinds when it goes off.
	debugOrbitAngle = orbitOverride ? debugOrbitAngle + deltaT : 0.0f;

	for(size_t i = 0; i < lights.size(); i++) {
		float speed = orbitSpeed[i];
		float angle = orbitAngle;
		if(speed == 0.0f) {
			// Static as authored, unless the override claims it. Restricted to
			// directional lights: a point light ignores dir entirely, and
			// swinging the spot's aim around world up is a different effect
			// from the one this switch advertises.
			if(!orbitOverride || lights[i].type != LIGHT_DIRECT) {
				// Not a no-op after the override goes off: puts the light back
				// on its authored direction.
				lights[i].dir = baseDir[i];
				continue;
			}
			speed = DEBUG_ORBIT_SPEED;
			angle = debugOrbitAngle;
		}
		// The tilt is baked into the authored direction, this only sweeps it
		// around world up. That's the sun crossing the sky.
		glm::mat4 R = glm::rotate(glm::mat4(1.0f),
								  glm::radians(speed * angle),
								  glm::vec3(0.0f, 1.0f, 0.0f));
		lights[i].dir = glm::vec3(R * glm::vec4(baseDir[i], 0.0f));
	}

	// Flicker: a sum of three sine waves at incommensurate frequencies, so the
	// result doesn't visibly repeat over the timescale anyone watches a torch.
	// orbitAngle is elapsed seconds (accumulated above, unconditionally), so
	// this runs whether or not orbitOverride is on. flickerPhase staggers each
	// light's copy of the same sum so identical torches don't pulse in sync.
	for(size_t i = 0; i < lights.size(); i++) {
		if(!flickerEnabled[i]) continue;
		float t = orbitAngle + flickerPhase[i];
		float n = 0.5f * std::sin(2.1f * t)
				+ 0.3f * std::sin(4.7f * t + 1.3f)
				+ 0.2f * std::sin(9.3f * t + 2.6f);
		lights[i].color = baseColor[i] * (1.0f + flickerStrength[i] * n);
	}

	// Rebuilt from scratch every frame: a switch can flip between two of them,
	// and at 4 lights the copy costs nothing worth tracking dirty state for.
	activeLights.clear();
	for(const LightData &L : lights) {
		bool enabled = (L.type == LIGHT_DIRECT) ? directEnabled
					 : (L.type == LIGHT_POINT)  ? pointEnabled
												: spotEnabled;
		if(enabled) {
			activeLights.push_back(L);
		}
	}

	return activeLights;
}

#endif
