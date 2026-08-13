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
};

// Hemispheric ambient, E07 s.47-54: the scene's indirect lighting, two colors
// blended by which way a surface faces.
struct AmbientLight {
	glm::vec3 upper = glm::vec3(0.1f);				// sky color
	glm::vec3 lower = glm::vec3(0.05f);				// ground color
	glm::vec3 dir = glm::vec3(0.0f, 1.0f, 0.0f);	// which way "up" blends
};

class SceneLights {
	public:
	// Must run after Scene::init: an "instance" reference needs its world matrix.
	void init(Scene *SC, const std::string &file);

	// Advances the animated lights and returns the list to upload.
	const std::vector<LightData> &update(float deltaT);

	int count() const { return (int)lights.size(); }

	// Not animated, so it skips update().
	const AmbientLight &ambient() const { return ambientLight; }

	private:
	std::vector<LightData> lights;
	AmbientLight ambientLight;

	// Animation state, index-matched with `lights`. Zero for anything static.
	std::vector<float> orbitSpeed;	// degrees per second around world Y
	std::vector<glm::vec3> baseDir;	// direction before any rotation
	float orbitAngle = 0.0f;

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);
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

		lights.push_back(L);
		baseDir.push_back(L.dir);
		orbitSpeed.push_back(l.value("orbitSpeed", 0.0f));
	}

	std::cout << "SceneLights: " << lights.size() << " lights loaded\n";
}

const std::vector<LightData> &SceneLights::update(float deltaT) {
	orbitAngle += deltaT;

	for(size_t i = 0; i < lights.size(); i++) {
		if(orbitSpeed[i] == 0.0f) continue;
		// The tilt is baked into the authored direction, this only sweeps it
		// around world up. That's the sun crossing the sky.
		glm::mat4 R = glm::rotate(glm::mat4(1.0f),
								  glm::radians(orbitSpeed[i] * orbitAngle),
								  glm::vec3(0.0f, 1.0f, 0.0f));
		lights[i].dir = glm::vec3(R * glm::vec4(baseDir[i], 0.0f));
	}

	return lights;
}

#endif
