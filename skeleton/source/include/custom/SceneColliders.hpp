// ***** CUSTOM *****
// Builds the gameplay collider list: scene.json's auto-generated boxes plus
// boxes/ramps authored in assets/scenes/colliders.json (for hollow/profiled models
// a single auto-fit box gets wrong, e.g. gate archways, staircases).
// Boxes authored in MODEL space, multiplied by instance world matrix.
// Header-only, implementation gated behind SCENECOLLIDERS_IMPLEMENTATION (Libs.cpp).

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// Inclined (or flat) walkable surface: staircases collide as a plain slope, not steps.
// Not a Collider: getExtents() returns an axis-aligned box, which can't represent a
// tilted surface. Ramps need a height FUNCTION, kept in a separate list read only
// by the ground pass (never the wall pass -- a ramp is walked onto, not blocked by).
struct GroundVolume {
	glm::mat4 Wm;			// model local -> world
	glm::mat4 invWm;		// world -> model local
	int runAxis;			// along the slope
	int riseAxis;			// "up" (-z for these Z-up models)
	int widthAxis;			// across the slope
	float runFrom, runTo;
	float riseFrom, riseTo;	// surface height at runFrom and at runTo
	float widthFrom, widthTo;

	// Writes surface height at worldPos if inside the volume; returns false otherwise.
	bool groundAt(const glm::vec3 &worldPos, float &worldY) const;
};

class SceneColliders {
	public:
	// Builds the full gameplay collider list from scene.json + `file`.
	void init(Scene *SC, const std::string &file);

	// Flat list of leaf boxes, for the collision loops in GameLogic().
	const std::vector<Collider *> &list() const { return colliders; }

	// Sloped/flat walkable surfaces, for the ground pass only.
	const std::vector<GroundVolume> &ramps() const { return groundVolumes; }

	// Deletes only what this class allocated; scene.json's colliders belong to Scene.
	void cleanup();

	private:
	std::vector<Collider *> colliders;	// scene.json's + ours
	std::vector<Collider *> owned;		// just ours, so cleanup() doesn't double-free
	std::vector<GroundVolume> groundVolumes;

	void addBox(const glm::mat4 &Wm, glm::vec3 lo, glm::vec3 hi);
	void addBoxes(Scene *SC, const std::string &instanceId, const nlohmann::json &boxes);
	void addRamps(Scene *SC, const std::string &instanceId, const nlohmann::json &ramps);
};

#ifdef SCENECOLLIDERS_IMPLEMENTATION

// Maps an axis name from the data file to a glm::vec3 component index.
static int sceneCollidersAxis(const std::string &name) {
	if(name == "x") return 0;
	if(name == "y") return 1;
	if(name == "z") return 2;
	std::cout << "SceneColliders: unknown axis '" << name << "', assuming x\n";
	return 0;
}

bool GroundVolume::groundAt(const glm::vec3 &worldPos, float &worldY) const {
	// Tested in model space, where the footprint is axis-aligned regardless of instance rotation.
	glm::vec3 p = glm::vec3(invWm * glm::vec4(worldPos, 1.0f));

	float w = p[widthAxis];
	if(w < std::min(widthFrom, widthTo) || w > std::max(widthFrom, widthTo)) {
		return false;
	}
	float r = p[runAxis];
	if(r < std::min(runFrom, runTo) || r > std::max(runFrom, runTo)) {
		return false;
	}

	// Flat volume (riseFrom == riseTo) is a level ramp via the same formula.
	float t = (runTo == runFrom) ? 0.0f : (r - runFrom) / (runTo - runFrom);
	float h = riseFrom + (riseTo - riseFrom) * t;

	// Transform back to world rather than assuming local "up" == world +Y.
	glm::vec3 surface = p;
	surface[riseAxis] = h;
	worldY = (Wm * glm::vec4(surface, 1.0f)).y;
	return true;
}

void SceneColliders::addBox(const glm::mat4 &Wm, glm::vec3 lo, glm::vec3 hi) {
	Collider *c = new Collider();
	// Sort corners: don't trust the data file to give min/max order.
	c->initAABB(std::min(lo.x, hi.x), std::min(lo.y, hi.y), std::min(lo.z, hi.z),
				std::max(lo.x, hi.x), std::max(lo.y, hi.y), std::max(lo.z, hi.z));
	c->setWorldMatrix(Wm);

	owned.push_back(c); // Add collider for cleanup
	colliders.push_back(c); // Add collider
}

void SceneColliders::init(Scene *SC, const std::string &file) {
	// scene.json's auto-fit boxes are still correct for solid box-shaped models.
	colliders = SC->GlobalColliders;

	std::ifstream ifs(file);
	if(!ifs.is_open()) {
		std::cout << "SceneColliders: '" << file
				  << "' not found, using scene.json's colliders only\n";
		return;
	}

	nlohmann::json js;
	ifs >> js;

	for(auto it = js.begin(); it != js.end(); ++it) {
		const std::string instanceId = it.key();
		if(SC->InstanceIds.find(instanceId) == SC->InstanceIds.end()) {
			std::cout << "SceneColliders: scene has no instance '" << instanceId
					  << "', skipped\n";
			continue;
		}
		if(it.value().contains("boxes")) {
			addBoxes(SC, instanceId, it.value()["boxes"]);
		}
		if(it.value().contains("ramps")) {
			addRamps(SC, instanceId, it.value()["ramps"]);
		}
	}

	std::cout << "SceneColliders: " << colliders.size() << " colliders ("
			  << owned.size() << " authored) + " << groundVolumes.size() << " ramps\n";
}

// "boxes": list of [x1,y1,z1, x2,y2,z2] AABBs in model space.
void SceneColliders::addBoxes(Scene *SC, const std::string &instanceId,
							  const nlohmann::json &boxes) {
	glm::mat4 Wm = SC->I[SC->InstanceIds[instanceId]]->Wm;

	for(const auto &b : boxes) {
		if(b.size() != 6) {
			std::cout << "SceneColliders: '" << instanceId
					  << "' has a box with " << b.size() << " values, needs 6, skipped\n";
			continue;
		}
		// Explicit get<float>(): glm::vec3's json conversion can pick the wrong ctor.
		addBox(Wm, glm::vec3(b[0].get<float>(), b[1].get<float>(), b[2].get<float>()),
				   glm::vec3(b[3].get<float>(), b[4].get<float>(), b[5].get<float>()));
	}
}

// "ramps": sloped walkable surfaces in model space; a flat ramp (no rise) is a landing.
void SceneColliders::addRamps(Scene *SC, const std::string &instanceId,
							  const nlohmann::json &ramps) {
	glm::mat4 Wm = SC->I[SC->InstanceIds[instanceId]]->Wm;

	for(const auto &r : ramps) {
		GroundVolume G;
		G.Wm        = Wm;
		G.invWm     = glm::inverse(Wm);
		G.runAxis   = sceneCollidersAxis(r["run"]["axis"]);
		G.riseAxis  = sceneCollidersAxis(r["rise"]["axis"]);
		G.widthAxis = sceneCollidersAxis(r["width"]["axis"]);
		G.runFrom   = r["run"]["from"];
		G.runTo     = r["run"]["to"];
		G.riseFrom  = r["rise"]["from"];
		G.riseTo    = r["rise"]["to"];
		G.widthFrom = r["width"]["from"];
		G.widthTo   = r["width"]["to"];
		groundVolumes.push_back(G);
	}
}

void SceneColliders::cleanup() {
	for(Collider *c : owned) {
		delete c;
	}
	owned.clear();
	colliders.clear();
	groundVolumes.clear();
}

#endif
