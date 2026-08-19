// ***** CUSTOM *****

// Owns the surface parameters that make stone look like stone and brass like
// brass, under the same light.
//
// How it fits in:
//   at startup   main.cpp calls init(), which reads
//                assets/scenes/materials.json into one Material per model
//   every frame  for each object it draws, main.cpp calls forModel() and copies
//                the result into that object's uniform buffer for the shader
//
// The four parameters, one line each (the maths is in notes.md):
//   specularColor  colour of the highlight. White for anything that isn't
//                  metal, because the highlight is the colour of the lamp, not
//                  of the object. Only metals tint it.
//   roughness      how rough the surface is, 0 to 1. 0 is a mirror with a tiny
//                  sharp highlight, 1 is chalk with none. This is the one you
//                  actually tune.
//   F0             how reflective it is when you look straight at it. About
//                  0.04 for every non-metal there is, so it gets copied rather
//                  than chosen. Metals are far higher.
//   k              how much of the surface's response is plain colour versus
//                  highlight. High for ordinary materials, low for metals.
//
// The base colour isn't here: it comes from the texture, per pixel.
//
// A data file rather than constants in main.cpp, for the same reason as
// colliders.json: these get tuned by looking at the result. scene.json can't
// carry them (Scene.hpp parses only id/model/texture/translate/eulerAngles/scale
// and Starter.hpp is off limits).
//
// Keyed by MODEL id, unlike colliders.json: a material belongs to the surface,
// so the 4 towers share one entry instead of repeating it 23 times. A future
// "instances" section could override this one, resolved after it.
//
// Header-only module like the rest of custom/, implementation gated behind
// SCENEMATERIALS_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// modules/Starter.hpp and modules/Scene.hpp are already included.

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
	// shading of anything that should be crisp. Set this for those models and
	// the shader derives the face normal itself. Leave it off for genuinely
	// curved surfaces (the towers), which the averaged normals suit.
	int flatNormals = 0;

	// Whether this model is drawn into the shadow maps at all. On for
	// everything except the fixtures that HOLD a light: the torch bracket sits
	// between the wall and its own flame and the skull hangs 22cm above it, and
	// an occluder that close to a point light subtends a huge solid angle -- the
	// pair of them threw a cone of shadow across most of the room, which read as
	// the torch not lighting anything. There is no bias or resolution that fixes
	// an occluder practically touching the light; the fix is not to treat it as
	// one. Not a BRDF parameter like the rest of this struct, but it is per
	// model and this is the one per-model table the render loop already has in
	// hand (see the forModel() call in populateCommandBuffer).
	//
	// The cost is that the bracket casts no little shadow of its own downwards.
	// Getting that honestly needs the light moved out of the fixture, or the
	// fixture's own shadow faked separately.
	bool castsShadow = true;
};

class SceneMaterials {
	public:
	// One Material per model. Anything without an entry falls back to the file's
	// "default", or to Material's own defaults if there's no file.
	void init(Scene *SC, const std::string &file);

	// Indexed by Instance::Mid rather than looked up by name, so the render loop
	// doesn't hash 23 strings a frame.
	const Material &forModel(int modelIndex) const {
		if(modelIndex < 0 || modelIndex >= (int)byModel.size()) return fallback;
		return byModel[modelIndex];
	}

	private:
	Material fallback;
	std::vector<Material> byModel;

	// Reads only the fields present, leaving the rest of `m` alone. That's what
	// makes the default/override layering work.
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

	// roughness 0 divides by zero in GGX. Caught here rather than guarded per
	// fragment: it's a data error.
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

	// Last arg is ignore_comments, so the file can keep its // lines.
	// parse(input, callback, allow_exceptions, ignore_comments).
	nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);

	// Two layers: the file's "default" replaces the built-in one, per-model
	// entries override that.
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
