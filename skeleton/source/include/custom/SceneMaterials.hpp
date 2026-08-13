// ***** CUSTOM *****

// Per-model surface parameters for the BRDF: the numbers that make stone look
// like stone and the lantern's brass look like metal, under the same light.
//
// The BRDF (see shaders/CookTorrance.frag) is the sum of a diffuse and a specular
// term, and each needs a material parameter:
//   mD, the diffuse color, is the "main color of the surface" (L09 slide 59).
//       It already exists: it's the albedo texture, per fragment, so it isn't
//       here.
//   mS, the specular color, says how the highlight reflects the light's RGB.
//       Most materials have mS white (the highlight is the color of the lamp),
//       metals have mS close to their own diffuse color (L09 slide 69).
//   roughness (rho on the slides) is the width of the microfacet distribution:
//       0 is a mirror, 1 is completely matte. It replaced the old specular
//       exponent, which said the same thing with a number that had no physical
//       scale to anchor it to.
//   F0 is the fraction of light reflected when looking straight at the surface.
//       Around 0.04 for every dielectric there is, much higher for metals. The
//       Fresnel term grows from F0 towards 1 as the view gets grazing.
//   k balances the diffuse and specular halves of the BRDF (E06 slide 38):
//       the model interpolates between them rather than adding both at full
//       strength, which is part of what makes it closer to energy-conserving.
// Those four are per-material constants, which is what this file holds.
//
// Why a data file and not constants in main.cpp: same reason the authored
// collision boxes live in colliders.json. These are numbers an artist tweaks by
// looking at the result, and scene.json can't carry them (Scene.hpp only parses
// id / model / texture / translate / eulerAngles / scale, and Starter.hpp is
// off limits, so extending its parser isn't an option).
//
// Keyed by MODEL id, not instance id, unlike colliders.json. A material is a
// property of the surface, so all 4 towers are the same stone; keying by
// instance would mean copying the same three numbers 23 times. If one instance
// ever needs to differ (a rusted barrel among clean ones) the natural fix is a
// second "instances" section overriding this one, resolved after it.
//
// Same header-only "module" pattern as the rest of custom/: declarations plus
// implementation in one file, implementation gated behind
// SCENEMATERIALS_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// "modules/Starter.hpp" and "modules/Scene.hpp" are already included.

#include <fstream>
#include <string>
#include <vector>

// The half of the BRDF's parameters that isn't the albedo texture.
// Defaults describe a neutral, slightly shiny dielectric, so a model missing
// from the data file still renders sensibly instead of turning black.
struct Material {
	glm::vec3 specularColor = glm::vec3(1.0f);	// mS
	float roughness = 0.6f;						// rho, width of the GGX lobe
	float F0 = 0.04f;							// reflectance seen head-on
	float k = 0.9f;								// diffuse share of the BRDF
};

class SceneMaterials {
	public:
	// Reads `file` and resolves one Material per model in the scene.
	// A model with no entry gets the file's "default", or this class's own
	// defaults if the file has none (or doesn't exist at all).
	void init(Scene *SC, const std::string &file);

	// Material for the model at `modelIndex`, i.e. an Instance's Mid.
	// Indexed rather than looked up by name so the per-frame render loop
	// doesn't hash a string 23 times a frame.
	const Material &forModel(int modelIndex) const {
		if(modelIndex < 0 || modelIndex >= (int)byModel.size()) return fallback;
		return byModel[modelIndex];
	}

	private:
	Material fallback;
	std::vector<Material> byModel;

	// Reads whichever of the two fields are present, leaving the rest of `m`
	// alone. That's what makes the default/override layering work: an entry
	// only has to name what it changes.
	static void readInto(const nlohmann::json &js, Material &m);
};

#ifdef SCENEMATERIALS_IMPLEMENTATION

void SceneMaterials::readInto(const nlohmann::json &js, Material &m) {
	if(js.contains("specularColor")) {
		const nlohmann::json &c = js["specularColor"];
		if(c.size() == 3) {
			// get<float>() rather than letting json convert itself: glm::vec3 has
			// several 3-argument constructors and the implicit conversion can pick
			// the wrong one. Same reasoning as SceneColliders::addBoxes().
			m.specularColor = glm::vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
		} else {
			std::cout << "SceneMaterials: specularColor needs 3 values, got "
					  << c.size() << ", kept previous\n";
		}
	}
	if(js.contains("roughness")) m.roughness = js["roughness"].get<float>();
	if(js.contains("F0"))        m.F0 = js["F0"].get<float>();
	if(js.contains("k"))         m.k = js["k"].get<float>();

	// A roughness of exactly 0 makes the GGX distribution divide by zero (a
	// perfect mirror has all its microfacets in one direction, so the lobe has
	// no width at all). Clamped here rather than in the shader: it is a data
	// error, and catching it once at load beats guarding every fragment.
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

	// parse() rather than `ifs >> js`, for the last argument: comments in JSON.
	// These numbers are tuned by eye and mean nothing without a note saying what
	// they represent, so the file keeps its `//` lines. Signature is
	// parse(input, callback, allow_exceptions, ignore_comments).
	nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);

	// The file's "default" replaces the built-in one for every model, then the
	// per-model entries override that. Two layers, applied in this order.
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
