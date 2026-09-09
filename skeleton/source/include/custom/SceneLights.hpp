// ***** CUSTOM *****

// Owns the scene's lights: init() reads lights.json into LightData; update()
// runs animation and returns the list main.cpp uploads to the GPU each frame.
// Types: direct (sun), point (lamp/candle, fades with distance), spot (cone).
// Header-only, gated behind SCENELIGHTS_IMPLEMENTATION (Libs.cpp).

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// MAX_LIGHTS and LIGHT_*, the same file CookTorrance.frag includes.
#include "custom/LightConstants.glsl"

// Matches the GLSL struct field for field; each float after a vec3 fills std140's pad byte.
struct LightData {
	alignas(16) glm::vec3 pos;		// point/spot only
	float g;						// decay reference distance
	alignas(16) glm::vec3 dir;		// direct: travel direction. spot: aim
	float beta;						// decay exponent
	alignas(16) glm::vec3 color;	// l, the emitted color
	float cosIn;					// spot: cosine of the half inner angle
	float cosOut;					// spot: cosine of the half outer angle
	int type;						// LIGHT_DIRECT / LIGHT_POINT / LIGHT_SPOT
	// -1: no shadow. Else index into the cube shadow maps (NUM_SHADOW_CUBES),
	// assigned in declaration order from lights.json's "castsShadow". Only
	// LIGHT_POINT can take one (real cube map, see shadowFactor() in CookTorrance.frag).
	int shadowIndex;
};

// Hemispheric ambient (E07 s.47-54): scene's indirect light, two colors blended by
// surface normal. Also stands in for the environment metalAmbient() reflects.
// Both default BLACK: every level must author its own, there's no right default
// for both a sealed dungeon and an open courtyard.
struct AmbientLight {
	glm::vec3 upper = glm::vec3(0.0f);				// sky color
	glm::vec3 lower = glm::vec3(0.0f);				// ground color
	glm::vec3 dir = glm::vec3(0.0f, 1.0f, 0.0f);	// which way "up" blends

	// Share of indirect light, 0..1: CookTorrance.frag blends ambient*weight with
	// direct*(1-weight). Default is the INDOOR value (game is played in the dungeon).
	// Overridable per model via Material::ambientWeight.
	float weight = 0.05f;

	// Share of each POINT/SPOT light's radiance fed back as indirect light, 0..1.
	// Same decay/color as the light's direct term (incl. flicker), but no shadow
	// test and a soft wrap instead of a clamped cosine (one bounce off rough stone
	// reaches round corners, has no terminator). Not physically derived -- stands
	// in for an average wall's albedo * solid angle. Shares the `weight` bucket,
	// not additive on top. Toggled independently by the Torch Bounce cheat.
	float bounce = 0.35f;
};

class SceneLights {
	public:
	// Must run after Scene::init: an "instance" reference needs its world matrix.
	void init(Scene *SC, const std::string &file);

	// Advances animated lights, returns the list to upload (only enabled types).
	const std::vector<LightData> &update(float deltaT);

	// Lights handed over by the last update(), after the type switches filtered them.
	int count() const { return (int)activeLights.size(); }

	// Full authored list, unfiltered, without waiting for update(). Used once at
	// startup to find shadow-casting lights (shadowIndex >= 0) and build their
	// light-space matrices (static, so no need for the per-frame path).
	const std::vector<LightData> &all() const { return lights; }

	// Not animated. bounceEnabled below is applied in the shader, not here.
	AmbientLight ambient() const;

	// Debug switches wired to the cheat menu (flips these bools in place).
	// One per light TYPE, not per light: scene has one sun, two matched lanterns.
	bool directEnabled = true;	// the sun
	bool pointEnabled = true;	// the torches and candles
	// Torch/candle radiance bounced off nearby surfaces (LIGHT_DEBUG_NO_BOUNCE in shader).
	bool bounceEnabled = true;

	// Forces an orbit onto directional lights authored static (orbitSpeed 0).
	// Lights WITH an authored orbitSpeed ignore this, keep their own speed.
	bool orbitOverride = false;

	private:
	std::vector<LightData> lights;
	// Filtered copy handed to the caller, rebuilt every update(); member so update() can return a reference.
	std::vector<LightData> activeLights;
	AmbientLight ambientLight;

	// Animation state, index-matched with `lights`. Zero for anything static.
	std::vector<float> orbitSpeed;	// degrees per second around world Y
	std::vector<glm::vec3> baseDir;	// direction before any rotation

	// Flame flicker (lights.json "flicker"), index-matched with `lights` too.
	std::vector<glm::vec3> baseColor;	// color before flicker scales it
	std::vector<bool> flickerEnabled;
	std::vector<float> flickerStrength;	// how far the scale swings from 1.0
	// Per-light stagger (golden-angle-ish step, index * 2.399963 rad) so
	// identical torches don't flicker in sync; needs no RNG.
	std::vector<float> flickerPhase;
	float orbitAngle = 0.0f;

	// Degrees/sec for orbitOverride: fast (full turn ~24s) for scanning sun angles quickly.
	static constexpr float DEBUG_ORBIT_SPEED = -15.0f;
	// Override's own angle: starts the sweep from the authored direction, and
	// resets the sun to it when switched off.
	float debugOrbitAngle = 0.0f;

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);

	// Next shadowIndex to hand out, incremented per "castsShadow": true point light
	// in declaration order. Not reset after init() -- only one pass over lights.json.
	int nextShadowIndexCube = 0;
};

#ifdef SCENELIGHTS_IMPLEMENTATION

glm::vec3 SceneLights::readVec3(const nlohmann::json &js, const glm::vec3 &fallback) {
	if(js.size() != 3) {
		std::cout << "SceneLights: expected 3 values, got " << js.size()
				  << ", using the default\n";
		return fallback;
	}
	// Explicit get<float>(): glm::vec3's several 3-arg ctors let implicit conversion pick the wrong one.
	return glm::vec3(js[0].get<float>(), js[1].get<float>(), js[2].get<float>());
}

void SceneLights::init(Scene *SC, const std::string &file) {
	std::ifstream ifs(file);
	if(!ifs.is_open()) {
		std::cout << "SceneLights: '" << file << "' not found, scene has no lights\n";
		return;
	}

	// ignore_comments: these numbers need their notes.
	nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);
	if(js.contains("ambient")) {
		const nlohmann::json &a = js["ambient"];
		if(a.contains("upper"))     ambientLight.upper = readVec3(a["upper"], ambientLight.upper);
		if(a.contains("lower"))     ambientLight.lower = readVec3(a["lower"], ambientLight.lower);
		if(a.contains("direction")) ambientLight.dir = glm::normalize(readVec3(a["direction"], ambientLight.dir));
		// Clamped: above 1 the blend would leave the direct term negative.
		if(a.contains("weight")) ambientLight.weight = glm::clamp(a["weight"].get<float>(), 0.0f, 1.0f);
		// Clamped: past 1 a wall would return more light than hit it.
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
			// Offset is in WORLD units, not model-local (models are Z-up and
			// rotated by the scene; world units match the Show Coordinates overlay).
			glm::vec3 origin = glm::vec3(SC->I[it->second]->Wm[3]);
			glm::vec3 offset = l.contains("offset") ? readVec3(l["offset"], glm::vec3(0.0f))
													: glm::vec3(0.0f);
			L.pos = origin + offset;
		} else if(L.type != LIGHT_DIRECT) {
			std::cout << "SceneLights: a " << type
					  << " light needs \"position\" or \"instance\", skipped\n";
			continue;
		}

		// Authored as FULL angles in degrees; shader wants cosine of the half
		// angle (L09 s.29), converted here once instead of per fragment.
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
			// Only a point light can cast a shadow (CUBE array, one 6-face map per torch).
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
				std::cout << "SceneLights: '" << l.value("id", std::string("?"))
						  << "' requested castsShadow but only a point light can cast one, "
						  << "renders unshadowed\n";
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
	// As authored; bounceEnabled is applied downstream in the shader (LIGHT_DEBUG_NO_BOUNCE).
	return ambientLight;
}

const std::vector<LightData> &SceneLights::update(float deltaT) {
	orbitAngle += deltaT;
	// Ticks only while override is on, rewinds when it goes off.
	debugOrbitAngle = orbitOverride ? debugOrbitAngle + deltaT : 0.0f;

	for(size_t i = 0; i < lights.size(); i++) {
		float speed = orbitSpeed[i];
		float angle = orbitAngle;
		if(speed == 0.0f) {
			// Static unless the override claims it; restricted to directional
			// lights (a point light ignores dir, a spot's aim is a different effect).
			if(!orbitOverride || lights[i].type != LIGHT_DIRECT) {
				lights[i].dir = baseDir[i]; // restores authored direction once override drops
				continue;
			}
			speed = DEBUG_ORBIT_SPEED;
			angle = debugOrbitAngle;
		}
		// Tilt is baked into the authored direction; this only sweeps it around world up.
		glm::mat4 R = glm::rotate(glm::mat4(1.0f),
								  glm::radians(speed * angle),
								  glm::vec3(0.0f, 1.0f, 0.0f));
		lights[i].dir = glm::vec3(R * glm::vec4(baseDir[i], 0.0f));
	}

	// Sum of three sine waves at incommensurate frequencies so flicker doesn't
	// visibly repeat. orbitAngle is elapsed seconds, runs regardless of orbitOverride.
	// flickerPhase staggers each light's copy so identical torches don't sync.
	for(size_t i = 0; i < lights.size(); i++) {
		if(!flickerEnabled[i]) continue;
		float t = orbitAngle + flickerPhase[i];
		float n = 0.5f * std::sin(2.1f * t)
				+ 0.3f * std::sin(4.7f * t + 1.3f)
				+ 0.2f * std::sin(9.3f * t + 2.6f);
		lights[i].color = baseColor[i] * (1.0f + flickerStrength[i] * n);
	}

	// Rebuilt every frame: at a handful of lights the copy is cheaper than dirty tracking.
	activeLights.clear();
	for(const LightData &L : lights) {
		bool enabled = (L.type == LIGHT_DIRECT) ? directEnabled
					 : (L.type == LIGHT_POINT)  ? pointEnabled
												: true;	// LIGHT_SPOT (exit spill): no cheat gate
		if(enabled) {
			activeLights.push_back(L);
		}
	}

	return activeLights;
}

#endif
