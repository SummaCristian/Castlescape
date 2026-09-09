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
	// -1: no shadow. Else index into the cube shadow maps. Point lights only.
	int shadowIndex;
};

// Hemispheric ambient: scene's indirect light, two colors blended by surface
// normal. Both default BLACK: no right default for both a dungeon and a courtyard.
struct AmbientLight {
	glm::vec3 upper = glm::vec3(0.0f);				// sky color
	glm::vec3 lower = glm::vec3(0.0f);				// ground color
	glm::vec3 dir = glm::vec3(0.0f, 1.0f, 0.0f);	// which way "up" blends

	// Share of indirect vs direct light, 0..1. Overridable per model (Material::ambientWeight).
	float weight = 0.05f;

	// Share of point/spot radiance fed back as indirect light. Not physical --
	// stands in for an average wall bouncing light around. Part of `weight`,
	// not additive. Torch Bounce cheat toggle.
	float bounce = 0.35f;
};

class SceneLights {
	public:
	// Must run after Scene::init: an "instance" reference needs its world matrix.
	void init(Scene *SC, const std::string &file);

	// Advances animated lights, returns the list to upload (only enabled types).
	const std::vector<LightData> &update(float deltaT);

	// Lights from the last update(), after cheat-switch filtering.
	int count() const { return (int)activeLights.size(); }

	// Full unfiltered list. Used once at startup to find shadow casters
	// (shadowIndex >= 0) and build their light-space matrices.
	const std::vector<LightData> &all() const { return lights; }

	// Not animated. bounceEnabled below is applied in the shader, not here.
	AmbientLight ambient() const;

	// Cheat menu switches, one per light TYPE (not per light).
	bool directEnabled = true;	// the sun
	bool pointEnabled = true;	// torches and candles
	bool bounceEnabled = true;	// torch/candle bounce (LIGHT_DEBUG_NO_BOUNCE in shader)

	// Forces an orbit onto static directional lights. Authored orbitSpeed wins over this.
	bool orbitOverride = false;

	private:
	std::vector<LightData> lights;
	std::vector<LightData> activeLights;	// filtered copy, rebuilt every update()
	AmbientLight ambientLight;

	// Animation state, index-matched with `lights`. Zero for anything static.
	std::vector<float> orbitSpeed;	// degrees per second around world Y
	std::vector<glm::vec3> baseDir;	// direction before any rotation

	// Flicker state, index-matched with `lights`.
	std::vector<glm::vec3> baseColor;	// color before flicker scales it
	std::vector<bool> flickerEnabled;
	std::vector<float> flickerStrength;	// swing from 1.0
	std::vector<float> flickerPhase;		// per-light stagger so torches don't sync
	float orbitAngle = 0.0f;

	// orbitOverride speed/angle: fast sweep for scanning sun angles, resets on toggle-off.
	static constexpr float DEBUG_ORBIT_SPEED = -15.0f;
	float debugOrbitAngle = 0.0f;

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);

	// Next shadowIndex to hand out, one per shadow-casting point light in declaration order.
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
			// Offset in world units, not model-local.
			glm::vec3 origin = glm::vec3(SC->I[it->second]->Wm[3]);
			glm::vec3 offset = l.contains("offset") ? readVec3(l["offset"], glm::vec3(0.0f))
													: glm::vec3(0.0f);
			L.pos = origin + offset;
		} else if(L.type != LIGHT_DIRECT) {
			std::cout << "SceneLights: a " << type
					  << " light needs \"position\" or \"instance\", skipped\n";
			continue;
		}

		// Authored as full angles in degrees; converted to half-angle cosine once here.
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
			// Static unless override claims it, and only for directional lights.
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

	// Three sine waves at unrelated frequencies so the flicker doesn't visibly repeat.
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
