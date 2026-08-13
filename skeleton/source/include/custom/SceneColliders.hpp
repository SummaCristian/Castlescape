// ***** CUSTOM *****

// Builds the collider list the gameplay code actually collides against.
//
// scene.json's own "collider" field can only auto-fit ONE box around a whole
// model. That's exactly right for solid, roughly box-shaped meshes (walls,
// towers, crates) and wrong for anything hollow or profiled:
//   - the gate's archway is a hole, and a single box fills it in, so the
//     passage everyone is supposed to walk through becomes solid;
//   - a staircase's box reaches the height of its TOP step everywhere along
//     its length, so instead of climbing it you walk into a 9-unit wall.
// For those models the "collider" field is left out of scene.json entirely and
// their collision geometry is authored here instead, as a handful of boxes in
// a separate data file. Keeping collision geometry separate from render
// geometry (and much coarser than it) is the normal split; what matters is
// that it's *data*, sitting next to the scene it describes, rather than magic
// numbers compiled into main.cpp.
//
// Boxes are authored in the MODEL's local space and get the INSTANCE's world
// matrix applied, so they follow whatever translate/eulerAngles/scale
// scene.json gives that instance with no duplicated numbers. Note these models
// are Z-up with their origin at the base, so "up" in local space is -Z: it
// becomes +Y only after the instance's usual 90-degree X rotation.
//
// The list this produces is deliberately FLAT and contains only leaf boxes.
// Scene.hpp can also parse a "BVH" collider with children, but it pushes the
// BVH's parent node into GlobalColliders alongside its children, and that
// parent's getExtents() returns the union of them all, which would bring back
// the very box-that-fills-the-archway problem we're avoiding. Filtering those
// parents back out isn't possible from here (Collider::type and ::children are
// private, and only ColliderShow is a friend), so authoring the leaves
// ourselves is also what keeps the list usable.
//
// Same header-only "module" pattern as TextMaker/Scene/UiQuad/CheatHud:
// declarations + implementation in this one file, implementation gated behind
// SCENECOLLIDERS_IMPLEMENTATION (defined once in Libs.cpp). Like those modules
// this file isn't self-guarded, and assumes "modules/Starter.hpp" and
// "modules/Scene.hpp" are already included by whoever includes it.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// An inclined (or flat) walkable surface. This is the collision answer for a
// staircase: the visual mesh keeps its steps, the collision is a plain slope,
// and the player walks up smoothly instead of being snapped a riser at a time.
// In first person you never see your own feet, so being up to half a riser off
// the drawn step is invisible.
//
// It deliberately isn't a Collider: the collision code reads geometry through
// getExtents(), which returns a world-space axis-ALIGNED box, so a tilted
// surface can't survive that trip (an OOBB comes back out as its fattened
// envelope). A slope needs a height *function*, not a box, so ramps live in
// their own list and are consulted only by the ground pass. The wall pass never
// sees them, which is also exactly right: a ramp is by definition something you
// walk onto, never something that blocks you.
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
	// Test in the model's local space: the footprint is axis-aligned there no
	// matter how the instance is rotated or scaled, so this stays exact for
	// ramps that aren't turned by a whole multiple of 90 degrees (which is
	// precisely what an AABB-based collider could never manage).
	glm::vec3 p = glm::vec3(invWm * glm::vec4(worldPos, 1.0f));

	float w = p[widthAxis];
	if(w < std::min(widthFrom, widthTo) || w > std::max(widthFrom, widthTo)) {
		return false;
	}
	float r = p[runAxis];
	if(r < std::min(runFrom, runTo) || r > std::max(runFrom, runTo)) {
		return false;
	}

	// Surface height at that point along the run. A flat volume (riseFrom equal
	// to riseTo) drops out of the same formula, so a landing at the top of a
	// flight is just a level ramp rather than a second concept.
	float t = (runTo == runFrom) ? 0.0f : (r - runFrom) / (runTo - runFrom);
	float h = riseFrom + (riseTo - riseFrom) * t;

	// Send the surface point back through the world matrix and read its height
	// there, instead of assuming local "up" ends up as world +Y.
	glm::vec3 surface = p;
	surface[riseAxis] = h;
	worldY = (Wm * glm::vec4(surface, 1.0f)).y;
	return true;
}

void SceneColliders::addBox(const glm::mat4 &Wm, glm::vec3 lo, glm::vec3 hi) {
	Collider *c = new Collider();
	// Order the corners here rather than trusting the data file to have them
	// the right way round.
	c->initAABB(std::min(lo.x, hi.x), std::min(lo.y, hi.y), std::min(lo.z, hi.z),
				std::max(lo.x, hi.x), std::max(lo.y, hi.y), std::max(lo.z, hi.z));
	c->setWorldMatrix(Wm);

	owned.push_back(c);
	colliders.push_back(c);
}

void SceneColliders::init(Scene *SC, const std::string &file) {
	// Start from what scene.json already built: the auto-fit boxes are still
	// the right answer for every solid, box-shaped model in the scene.
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

// "boxes": a plain list of [x1,y1,z1, x2,y2,z2] AABBs in model-local space.
// Used for shapes an auto-fit box gets wrong but that a couple of hand-placed
// boxes describe exactly, like the gate (two piers and a lintel, archway left
// open in between).
void SceneColliders::addBoxes(Scene *SC, const std::string &instanceId,
							  const nlohmann::json &boxes) {
	glm::mat4 Wm = SC->I[SC->InstanceIds[instanceId]]->Wm;

	for(const auto &b : boxes) {
		if(b.size() != 6) {
			std::cout << "SceneColliders: '" << instanceId
					  << "' has a box with " << b.size() << " values, needs 6, skipped\n";
			continue;
		}
		// get<float>() rather than letting the json objects convert themselves:
		// glm::vec3 has several 3-argument constructors, so implicit conversions
		// from json can pick the wrong one (or fail to pick at all).
		addBox(Wm, glm::vec3(b[0].get<float>(), b[1].get<float>(), b[2].get<float>()),
				   glm::vec3(b[3].get<float>(), b[4].get<float>(), b[5].get<float>()));
	}
}

// "ramps": one or more sloped walkable surfaces, in model-local space. This is
// what the castle staircase uses: two entries, the flight itself and the flat
// landing at the top (a ramp whose rise doesn't change).
//
// Preferred over expanding the flight into one box per tread because these
// steps are huge next to the player (a 0.45 riser against a 1.8 body, a quarter
// of his height every 0.8 units of walking): stepped collision is technically
// correct there but reads as a violent stutter. The average slope is a perfectly
// ordinary 29 degrees, so the quantisation was the only thing hurting.
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
