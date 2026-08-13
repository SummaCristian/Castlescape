// ***** CUSTOM *****

// The scene's light sources, loaded from assets/scenes/lights.json.
//
// L09 presents three types, and this file carries all three:
//
//   direct  A source infinitely far away: one direction, one color, the same
//           everywhere in the scene. The sun or the moon. No position, no decay.
//   point   Emits from a position in all directions: lamps, bulbs, candles.
//           Its direction varies from point to point (it always aims at the
//           lamp), and its color falls off with distance.
//   spot    A point light confined to a cone. Same position, same decay, plus
//           an aim direction and two angles.
//
// Decay, for point and spot (L09 slides 19-23). Physically the intensity falls
// with the inverse square of the distance, which in a rendered image usually
// comes out too dark, so the model exposes it as two authored numbers instead:
//   g     the distance at which the light is exactly its stated color. Closer
//         than g it is brighter, further it dims.
//   beta  the falloff exponent: 0 constant, 1 inverse-linear, 2 the physically
//         correct inverse-square.
// The shader computes (g / |p - x|)^beta.
//
// A light's position can be given two ways: "position" for explicit world
// coordinates, or "instance" plus "offset" to hang it off a scene.json instance,
// which is what the two gate lanterns use. The second form is the one that
// survives moving the lantern in scene.json: same idea as the authored collision
// boxes, which follow their instance's world matrix rather than repeating its
// coordinates.
//
// Same header-only "module" pattern as the rest of custom/, implementation
// gated behind SCENELIGHTS_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// "modules/Starter.hpp" and "modules/Scene.hpp" are already included.

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// MAX_LIGHTS and the LIGHT_* type tags. The very same file is included by
// CookTorrance.frag, so there is one definition rather than two that have to be kept
// in agreement by hand.
#include "custom/LightConstants.glsl"

// One light, laid out to match the GLSL struct field for field.
//
// The vec3-then-float pairing is deliberate and is the same std140 idiom
// already used for the material: a vec3 has 16-byte alignment and 12-byte size,
// so a float placed right after it lands in the 4 bytes that would otherwise be
// padding. Three pairs plus two trailing scalars come to 56 bytes, rounded up
// to 64 by the struct's own 16-byte alignment, in GLSL and in C++ alike.
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

// Hemispheric ambient light: the scene's indirect lighting (E07 slides 47-54).
//
// A constant ambient term says "some light arrives from everywhere, equally".
// That is never true outdoors: a surface facing up sees the sky, one facing down
// sees the ground, and those are different colors. This model is the cheapest
// thing that captures it, two colors blended by the surface's orientation.
//
// It is what replaces the 0.015 constant the shader used to add. The project
// rules require indirect lighting and say a constant term barely qualifies.
struct AmbientLight {
	glm::vec3 upper = glm::vec3(0.1f);				// sky color
	glm::vec3 lower = glm::vec3(0.05f);				// ground color
	glm::vec3 dir = glm::vec3(0.0f, 1.0f, 0.0f);	// which way "up" blends
};

class SceneLights {
	public:
	// Reads `file` and resolves every light against the scene (an "instance"
	// reference needs the instance's world matrix, so this must run after
	// Scene::init).
	void init(Scene *SC, const std::string &file);

	// Advances the animated lights by `deltaT` seconds and returns the list to
	// upload. Only the orbit of a direct light is animated for now; everything
	// else passes straight through.
	const std::vector<LightData> &update(float deltaT);

	int count() const { return (int)lights.size(); }

	// The scene's indirect lighting. Static, so it is read once rather than
	// going through update().
	const AmbientLight &ambient() const { return ambientLight; }

	private:
	std::vector<LightData> lights;
	AmbientLight ambientLight;

	// Per-light animation state, index-matched with `lights`.
	// Zero for everything that doesn't move, which is nearly everything.
	std::vector<float> orbitSpeed;	// degrees per second around world Y
	std::vector<glm::vec3> baseDir;	// direction before any rotation
	float orbitAngle = 0.0f;		// accumulated, shared by all orbiting lights

	static glm::vec3 readVec3(const nlohmann::json &js, const glm::vec3 &fallback);
};

#ifdef SCENELIGHTS_IMPLEMENTATION

glm::vec3 SceneLights::readVec3(const nlohmann::json &js, const glm::vec3 &fallback) {
	if(js.size() != 3) {
		std::cout << "SceneLights: expected 3 values, got " << js.size()
				  << ", using the default\n";
		return fallback;
	}
	// get<float>() rather than an implicit conversion, for the same reason as
	// SceneColliders and SceneMaterials: glm::vec3 has several 3-argument
	// constructors and the conversion can pick the wrong one.
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

		// Position: either explicit world coordinates, or an instance to hang
		// off. The instance form keeps the light attached to the model it comes
		// out of, so moving the lantern in scene.json moves its flame with it.
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
			// The instance's world matrix gives its origin; the offset is added
			// in world units, not model-local ones. These models are Z-up and
			// rotated 90 degrees on X by the scene, so a local offset would need
			// the reader to keep that in mind for every number. World units are
			// what the Show Coordinates HUD reads out, which is how these get
			// tuned in the first place.
			glm::vec3 origin = glm::vec3(SC->I[it->second]->Wm[3]);
			glm::vec3 offset = l.contains("offset") ? readVec3(l["offset"], glm::vec3(0.0f))
													: glm::vec3(0.0f);
			L.pos = origin + offset;
		} else if(L.type != LIGHT_DIRECT) {
			std::cout << "SceneLights: a " << type
					  << " light needs \"position\" or \"instance\", skipped\n";
			continue;
		}

		// Cone angles are authored in degrees, as full angles, because that is
		// how a lamp's beam width is normally described. The shader wants the
		// cosine of the HALF angle (L09 slide 29), so the conversion happens
		// here, once at load, rather than per fragment.
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
		// Turns the authored direction around the world up axis. This is what
		// makes the sun cross the sky: the tilt is baked into the authored
		// direction, the rotation only sweeps it around.
		glm::mat4 R = glm::rotate(glm::mat4(1.0f),
								  glm::radians(orbitSpeed[i] * orbitAngle),
								  glm::vec3(0.0f, 1.0f, 0.0f));
		lights[i].dir = glm::vec3(R * glm::vec4(baseDir[i], 0.0f));
	}

	return lights;
}

#endif
