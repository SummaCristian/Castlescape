// ***** CUSTOM *****

// Owns the surface parameters that make stone look like stone and brass like
// brass under the same light.
//
//   startup      init() reads assets/scenes/materials.json into one Material
//                per model
//   every frame  main.cpp calls forModel() per object and copies the result
//                into that object's uniform buffer
//
// The BRDF parameters (maths in notes.md):
//   specularColor  color of the highlight. White for non-metals, since the
//                  highlight is the color of the lamp, not of the object.
//   roughness      0 = mirror with a tiny sharp highlight, 1 = chalk with
//                  none. The one that actually gets tuned.
//   F0             reflectance when looking straight at the surface. ~0.04 for
//                  every dielectric; metals are much higher.
//   k              balance between plain color and highlight. Ignored by
//                  metals, which have no diffuse term.
//
// The base color is not here: it comes per pixel from the texture.
//
// Keyed by MODEL id, so the 4 towers share one entry instead of repeating it
// per instance. It is a data file and not constants in the code because these
// values are tuned by looking at the result.
//
// Header-only like the rest of custom/: the implementation is compiled only
// where SCENEMATERIALS_IMPLEMENTATION is defined (Libs.cpp). Assumes
// modules/Starter.hpp and modules/Scene.hpp were included first.

#include <fstream>
#include <string>
#include <vector>

// Defaults are a neutral dielectric, so a model missing from the data file
// still renders sensibly instead of turning black.
struct Material {
	glm::vec3 specularColor = glm::vec3(1.0f);	// mS
	float roughness = 0.6f;						// rho, width of the GGX lobe
	float F0 = 0.04f;							// reflectance head-on
	float k = 0.9f;								// diffuse share
	// The MGCG meshes average their normals across hard edges, which smears the
	// shading of anything that should be crisp. With this on the shader derives
	// the face normal itself instead. Leave it off for genuinely curved
	// surfaces, which the averaged normals suit.
	//
	// int and not bool because it is copied straight into a uniform buffer, and
	// std140 has no bool. Same for the other flags below.
	int flatNormals = 0;

	// Whether this model is rendered into the shadow maps. Off for the fixtures
	// that hold a light: a torch bracket sits between the wall and its own
	// flame, and an occluder that close to a point light covers a huge solid
	// angle, throwing a cone of shadow over most of the room. No bias or
	// resolution fixes that; the fix is not to treat it as an occluder.
	//
	// The cost is that the bracket casts no small shadow of its own.
	bool castsShadow = true;

	// Hemispheric ambient blends "faces the sky" and "faces the ground" by the
	// surface normal (hemisphericAmbient() in CookTorrance.frag). That is right
	// outdoors and wrong in a closed room: a ceiling points straight down, so it
	// would collect only the ground color, which lights.json authored as bounce
	// off a dirt courtyard - dark and brown.
	//
	// With this on the shader uses the weight of a vertical surface instead, so
	// a ceiling gets the same ambient as the walls it sits on. Opt-in, because
	// the castle exterior wants the real thing.
	int interiorAmbient = 0;

	// Per-model override of AmbientLight::weight. Negative means "inherit the
	// scene's", which is the default: the scene weight is the indoor one, so it
	// is the exterior models that carry an override.
	//
	// Distinct from interiorAmbient: that changes the DIRECTION the hemisphere
	// is sampled from, this changes HOW MUCH of it is used. A ceiling needs
	// both, the dungeon floor only this one.
	//
	// It exists because hemispheric ambient has no visibility term, so a sealed
	// room would collect as much indirect light as an open courtyard. Until
	// there is an AO map (the MGCG pack ships albedo only), this is the manual
	// stand-in for occlusion: one number per model saying how enclosed it is.
	float ambientWeight = -1.0f;

	// Shade this model as a metal instead of a dielectric. The parameters above
	// cannot express a conductor by tuning alone; this switches two things in
	// CookTorrance.frag:
	//
	//   no diffuse lobe (k forced to 0). The diffuse term is light that entered
	//   the surface and scattered back out, which the free electrons of a metal
	//   absorb instead. Any leftover diffuse paints the object with its albedo
	//   texture, which is what made the brass padlock read as orange plastic.
	//
	//   the indirect term becomes a reflection (metalAmbient()) rather than the
	//   hemisphere times the albedo. This is the one that sells it: what you
	//   see on a lock or a chain is mostly the room around it. It also rescues
	//   dark metal out of direct light, where a near-black albedo multiplied
	//   the diffuse ambient away and the chains went black.
	//
	// With this on, specularColor becomes the material's reflectance color
	// rather than the highlight color, and is no longer optional: mS * F0
	// should equal the metal's measured reflectance (iron ~0.56/0.57/0.58,
	// brass ~0.95/0.64/0.37). Left white, a metal reflects like chrome.
	int metallic = 0;
};

class SceneMaterials {
	public:
	// One Material per model. Anything without an entry falls back to the file's
	// "default", or to Material's own defaults if there's no file.
	void init(Scene *SC, const std::string &file);

	// Indexed by Instance::Mid instead of looked up by name, so the render loop
	// doesn't hash a string per object per frame.
	const Material &forModel(int modelIndex) const {
		if(modelIndex < 0 || modelIndex >= (int)byModel.size()) return fallback;
		return byModel[modelIndex];
	}

	private:
	Material fallback;
	std::vector<Material> byModel;

	// Reads only the fields present, leaving the rest of `m` untouched: that is
	// what makes the default/override layering work.
	static void readInto(const nlohmann::json &js, Material &m);
};

#ifdef SCENEMATERIALS_IMPLEMENTATION

void SceneMaterials::readInto(const nlohmann::json &js, Material &m) {
	if(js.contains("specularColor")) {
		const nlohmann::json &c = js["specularColor"];
		if(c.size() == 3) {
			// Explicit get<float>(): glm::vec3 has several 3-arg constructors
			// and the implicit json conversion can pick the wrong one.
			m.specularColor = glm::vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
		} else {
			std::cout << "SceneMaterials: specularColor needs 3 values, got "
					  << c.size() << ", kept previous\n";
		}
	}
	if(js.contains("roughness")) m.roughness = js["roughness"].get<float>();
	if(js.contains("F0"))        m.F0 = js["F0"].get<float>();
	if(js.contains("k"))         m.k = js["k"].get<float>();
	if(js.contains("flatNormals")) m.flatNormals = js["flatNormals"].get<bool>() ? 1 : 0;
	if(js.contains("castsShadow")) m.castsShadow = js["castsShadow"].get<bool>();
	if(js.contains("interior"))    m.interiorAmbient = js["interior"].get<bool>() ? 1 : 0;
	if(js.contains("metallic"))    m.metallic = js["metallic"].get<bool>() ? 1 : 0;
	// Only the top end is clamped: negative is the "inherit the scene's" flag.
	if(js.contains("ambientWeight")) m.ambientWeight = glm::min(js["ambientWeight"].get<float>(), 1.0f);

	// roughness 0 divides by zero in GGX. Clamped here, once, rather than
	// guarded per fragment.
	m.roughness = glm::clamp(m.roughness, 0.03f, 1.0f);
}

void SceneMaterials::init(Scene *SC, const std::string &file) {
	byModel.assign(SC->ModelCount, fallback);

	std::ifstream ifs(file);
	if(!ifs.is_open()) {
		std::cout << "SceneMaterials: '" << file
				  << "' not found, every model gets the default material\n";
		return;
	}

	// parse(input, callback, allow_exceptions, ignore_comments): the last flag
	// lets the file keep its // lines.
	nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);

	// Two layers: the file's "default" replaces the built-in one, then
	// per-model entries override that.
	if(js.contains("default")) {
		readInto(js["default"], fallback);
		byModel.assign(SC->ModelCount, fallback);
	}

	int named = 0;
	if(js.contains("models")) {
		for(auto it = js["models"].begin(); it != js["models"].end(); ++it) {
			auto mIt = SC->MeshIds.find(it.key());
			if(mIt == SC->MeshIds.end()) {
				std::cout << "SceneMaterials: scene has no model '" << it.key()
						  << "', skipped\n";
				continue;
			}
			readInto(it.value(), byModel[mIt->second]);
			named++;
		}
	}

	std::cout << "SceneMaterials: " << named << " of " << SC->ModelCount
			  << " models have their own material\n";
}

#endif
