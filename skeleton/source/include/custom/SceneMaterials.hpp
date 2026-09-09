// ***** CUSTOM *****
// BRDF material params per model, loaded from assets/scenes/materials.json.
// main.cpp calls forModel()/forInstance() per object each frame to fill its UBO.
// Base color not here: comes per pixel from the texture.
// Header-only, implementation gated behind SCENEMATERIALS_IMPLEMENTATION (Libs.cpp).

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// Defaults: neutral dielectric, so a model missing from the data still renders sensibly.
struct Material {
	glm::vec3 specularColor = glm::vec3(1.0f);		// highlight color; white for non-metals
	float roughness = 0.6f;						// GGX lobe width; 0=mirror, 1=chalk
	float F0 = 0.04f;								// reflectance head-on; ~0.04 dielectric, higher for metals
	float diffuseShare = 0.9f;						// diffuse vs highlight balance, JSON key "k"; ignored by metals
	// Derives face normals in-shader instead of using averaged mesh normals (crisp edges vs curved surfaces).
	// int not bool: copied into a UBO, std140 has no bool. Same for the flags below.
	int flatNormals = 0;

	// Rendered into shadow maps or not. Off for light-holding fixtures (e.g. torch bracket):
	// an occluder that close to a point light shadows most of the room.
	bool castsShadow = true;

	// Gates procedural grime on interior metal props (chains, padlock, key); see CookTorrance.frag grime block.
	int interiorAmbient = 0;

	// Per-model override of AmbientLight::weight; negative inherits the scene's.
	// Distinct from interiorAmbient (direction vs amount of hemisphere sampled).
	// Stand-in for occlusion until an AO map exists.
	float ambientWeight = -1.0f;

	// Shades as metal in CookTorrance.frag: no diffuse lobe, indirect term becomes
	// a reflection (metalAmbient()) instead of hemisphere*albedo.
	// specularColor becomes the reflectance color: specularColor * F0 should equal
	// the metal's measured reflectance (iron ~0.56/0.57/0.58, brass ~0.95/0.64/0.37).
	int metallic = 0;
};

class SceneMaterials {
	public:
	// One Material per model; falls back to file's "default" or Material's own defaults.
	void init(Scene *SC, const std::string &file);

	// Indexed by Instance::Mid, not name, so the render loop avoids hashing strings.
	const Material &forModel(int modelIndex) const {
		if(modelIndex < 0 || modelIndex >= (int)byModel.size()) return fallback;
		return byModel[modelIndex];
	}

	// Honors per-instance override from file's "instances" block; falls through to forModel().
	// Used where instances of the same model must shade differently (e.g. the three door keys).
	const Material &forInstance(int instanceIndex, int modelIndex) const {
		auto it = byInstance.find(instanceIndex);
		return it != byInstance.end() ? it->second : forModel(modelIndex);
	}

	private:
	Material fallback;
	std::vector<Material> byModel;
	std::unordered_map<int, Material> byInstance;	// Instance::Iid -> override

	// Reads only present fields, leaving rest of `m` untouched (enables default/override layering).
	static void readInto(const nlohmann::json &js, Material &m);
};

#ifdef SCENEMATERIALS_IMPLEMENTATION

void SceneMaterials::readInto(const nlohmann::json &js, Material &m) {
	if(js.contains("specularColor")) {
		const nlohmann::json &c = js["specularColor"];
		if(c.size() == 3) {
			// Explicit get<float>(): glm::vec3's json conversion can pick the wrong ctor.
			m.specularColor = glm::vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
		} else {
			std::cout << "SceneMaterials: specularColor needs 3 values, got "
					  << c.size() << ", kept previous\n";
		}
	}
	if(js.contains("roughness")) m.roughness = js["roughness"].get<float>();
	if(js.contains("F0"))        m.F0 = js["F0"].get<float>();
	if(js.contains("k"))         m.diffuseShare = js["k"].get<float>();
	if(js.contains("flatNormals")) m.flatNormals = js["flatNormals"].get<bool>() ? 1 : 0;
	if(js.contains("castsShadow")) m.castsShadow = js["castsShadow"].get<bool>();
	if(js.contains("interior"))    m.interiorAmbient = js["interior"].get<bool>() ? 1 : 0;
	if(js.contains("metallic"))    m.metallic = js["metallic"].get<bool>() ? 1 : 0;
	// Only top end clamped: negative is the "inherit scene's" flag.
	if(js.contains("ambientWeight")) m.ambientWeight = glm::min(js["ambientWeight"].get<float>(), 1.0f);

	// roughness 0 divides by zero in GGX; clamp once here instead of per fragment.
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

	// ignore_comments=true: lets the file keep its // lines.
	nlohmann::json js = nlohmann::json::parse(ifs, nullptr, true, true);

	// Layering: default -> models -> instances
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

	int namedInstances = 0;
	if(js.contains("instances")) {
		for(auto it = js["instances"].begin(); it != js["instances"].end(); ++it) {
			auto iIt = SC->InstanceIds.find(it.key());
			if(iIt == SC->InstanceIds.end()) {
				std::cout << "SceneMaterials: scene has no instance '" << it.key()
						  << "', skipped\n";
				continue;
			}
			// Layer on the instance's model material; override only names differing fields.
			Material m = byModel[SC->I[iIt->second]->Mid];
			readInto(it.value(), m);
			byInstance[iIt->second] = m;
			namedInstances++;
		}
	}

	std::cout << "SceneMaterials: " << named << " of " << SC->ModelCount
			  << " models have their own material, " << namedInstances
			  << " instances override theirs\n";
}

#endif
