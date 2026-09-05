// ***** CUSTOM *****

// Builds the collider list the gameplay code collides against: everything
// scene.json auto-generated, plus the boxes and ramps authored in
// assets/scenes/colliders.json.
//
// scene.json's "collider" field can only fit ONE box around a whole model,
// which is right for solid box-shaped meshes (walls, crates) and wrong for
// anything hollow or profiled:
//   - a gate's archway is a hole, and one box fills it in, sealing the passage
//     the player is meant to walk through;
//   - a staircase's box is as tall as its top step along its whole length, so
//     instead of climbing it you walk into a wall.
// Those models leave "collider" out of scene.json and get their collision
// geometry authored here instead.
//
// Boxes are authored in MODEL space and multiplied by the INSTANCE's world
// matrix, so they follow the translate/rotate/scale in scene.json with no
// duplicated numbers. Careful: these models are Z-up with the origin at the
// base, so local "up" is -Z and only becomes +Y after the instance rotation.
//
// The resulting list is flat and holds leaf boxes only. Scene.hpp can parse a
// BVH collider with children, but it also pushes the parent node into
// GlobalColliders, and the parent's extents are the union of its children -
// which brings back the box that seals the archway.
//
// Header-only like the rest of custom/: the implementation is compiled only
// where SCENECOLLIDERS_IMPLEMENTATION is defined (Libs.cpp). Assumes
// modules/Starter.hpp and modules/Scene.hpp were included first.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// An inclined (or flat) walkable surface: the collision answer for a staircase.
// The mesh keeps its steps, the collision is a plain slope, and the player
// walks up smoothly instead of being snapped up one riser at a time. In first
// person you never see your feet, so being half a riser off is invisible.
//
// Not a Collider on purpose: the collision code reads geometry through
// getExtents(), which returns a world-space axis-ALIGNED box, and a tilted
// surface can't survive that (it comes back as its fattened envelope). A slope
// needs a height FUNCTION, not a box, so ramps live in a separate list read
// only by the ground pass. The wall pass never sees them, which is correct: a
// ramp is something you walk onto, never something that blocks you.
struct GroundVolume {
	glm::mat4 Wm;			// model local -> world
	glm::mat4 invWm;		// world -> model local
	int runAxis;			// along the slope
	int riseAxis;			// "up" (-z for these Z-up models)
	int widthAxis;			// across the slope
	float runFrom, runTo;
	float riseFrom, riseTo;	// surface height at runFrom and at runTo
	float widthFrom, widthTo;

	// If worldPos is over this volume, writes the surface height under it and
	// returns true. Height varies continuously along the run.
	bool groundAt(const glm::vec3 &worldPos, float &worldY) const;
};

class SceneColliders {
	public:
	// Reads `file` and builds the full gameplay collider list: everything
	// scene.json already produced, plus everything authored in `file`.
	void init(Scene *SC, const std::string &file);

	// Flat list of leaf boxes, ready for the collision loops in GameLogic().
	const std::vector<Collider *> &list() const { return colliders; }

	// Sloped/flat walkable surfaces, for the ground pass only.
	const std::vector<GroundVolume> &ramps() const { return groundVolumes; }

	// Deletes only what this class allocated. The colliders that came from
	// scene.json belong to Scene, which frees them in its own localCleanup().
	void cleanup();

	private:
	// Everything the gameplay collides against (scene.json's + ours).
	std::vector<Collider *> colliders;
	// Just the subset we allocated, so cleanup() doesn't double-free Scene's.
	std::vector<Collider *> owned;
	// Sloped surfaces, kept apart from the boxes above (see GroundVolume).
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
	// Tested in model space, where the footprint is axis-aligned whatever the
	// instance rotation is. This is what stays exact for ramps not turned by a
	// multiple of 90 degrees, which an AABB never could.
	glm::vec3 p = glm::vec3(invWm * glm::vec4(worldPos, 1.0f));

	float w = p[widthAxis];
	if(w < std::min(widthFrom, widthTo) || w > std::max(widthFrom, widthTo)) {
		return false;
	}
	float r = p[runAxis];
	if(r < std::min(runFrom, runTo) || r > std::max(runFrom, runTo)) {
		return false;
	}

	// Height at that point along the run. A flat volume (riseFrom == riseTo)
	// falls out of the same formula, so a landing is just a level ramp.
	float t = (runTo == runFrom) ? 0.0f : (r - runFrom) / (runTo - runFrom);
	float h = riseFrom + (riseTo - riseFrom) * t;

	// Transform the surface point back to world and read its height there,
	// rather than assuming local "up" maps to world +Y.
	glm::vec3 surface = p;
	surface[riseAxis] = h;
	worldY = (Wm * glm::vec4(surface, 1.0f)).y;
	return true;
}

void SceneColliders::addBox(const glm::mat4 &Wm, glm::vec3 lo, glm::vec3 hi) {
	Collider *c = new Collider();
	// Sort the corners here instead of trusting the data file to give them in
	// min/max order.
	c->initAABB(std::min(lo.x, hi.x), std::min(lo.y, hi.y), std::min(lo.z, hi.z),
				std::max(lo.x, hi.x), std::max(lo.y, hi.y), std::max(lo.z, hi.z));
	c->setWorldMatrix(Wm);

	owned.push_back(c);
	colliders.push_back(c);
}

void SceneColliders::init(Scene *SC, const std::string &file) {
	// Start from scene.json's auto-fit boxes: still the right answer for every
	// solid box-shaped model.
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

// "boxes": a list of [x1,y1,z1, x2,y2,z2] AABBs in model space. For shapes a
// single auto-fit box gets wrong but a few hand-placed ones describe exactly,
// like a gate: two piers and a lintel, archway left open.
void SceneColliders::addBoxes(Scene *SC, const std::string &instanceId,
							  const nlohmann::json &boxes) {
	glm::mat4 Wm = SC->I[SC->InstanceIds[instanceId]]->Wm;

	for(const auto &b : boxes) {
		if(b.size() != 6) {
			std::cout << "SceneColliders: '" << instanceId
					  << "' has a box with " << b.size() << " values, needs 6, skipped\n";
			continue;
		}
		// Explicit get<float>(): glm::vec3 has several 3-argument constructors
		// and the implicit json conversion can pick the wrong one.
		addBox(Wm, glm::vec3(b[0].get<float>(), b[1].get<float>(), b[2].get<float>()),
				   glm::vec3(b[3].get<float>(), b[4].get<float>(), b[5].get<float>()));
	}
}

// "ramps": sloped walkable surfaces in model space. The castle staircase uses
// two: the flight itself and the flat landing at the top (a ramp with no rise).
//
// Preferred over one box per step because the steps are big next to the player
// (0.45 riser against a 1.8 body, a quarter of his height every 0.8 units
// walked): stepped collision is correct but feels like a violent stutter. The
// average slope is an ordinary 29 degrees, so only the quantisation hurt.
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
