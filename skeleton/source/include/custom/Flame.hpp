// ***** CUSTOM *****

// A small, reusable, low-poly animated flame. Built for the held torch, but
// nothing here is torch-specific: any caller just spawn()s a flame and feeds
// it a world matrix every frame (the same way SceneLights anchors a light to
// an instance + offset), so a lantern can get one later with one more call.
//
// Where the "low-poly" look comes from: the mesh is a short stack of rings
// tapering to a point, each vertex nudged by a small deterministic jitter so
// the silhouette isn't a perfect cone, built once on the CPU (createMesh()).
// All the motion -- sway, flicker -- happens in Flame.vert every frame, driven
// by gubo.time, so the vertex/index buffers themselves never change. The
// fragment shader shades unlit and flat (Flame.frag), a plain height-based
// gradient, one flat color per triangle: the geometry does the faceting, the
// shader just declares its inputs "flat" so nothing gets smoothed across it.
//
// Rendered as an ordinary opaque pass inside the SAME render pass/subpass as
// the main scene (main.cpp calls populateCommandBuffer() right after
// SC.populateCommandBuffer(), no separate RenderPass like UiQuad needs):
// that's what lets it depth-test against the castle/dungeon geometry the
// normal way, for free.
//
// Header-only module like the rest of custom/, implementation gated behind
// FLAME_IMPLEMENTATION (defined once in Libs.cpp). Assumes modules/Starter.hpp
// is already included by whoever includes this one.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

struct FlameVertex {
	glm::vec3 pos;
	float tier;		// 0 at the base, 1 at the tip -- see Flame.vert/.frag
	// A fixed per-vertex random in [0,1), baked in at mesh-build time. Fed
	// into Flame.vert's sway phase so the individual tips whip independently
	// instead of the whole flame swaying as one rigid blob, and into
	// Flame.frag's brightness flicker for the same reason.
	float swaySeed;
};

// Matches FlameUniformBufferObject in Flame.vert field for field. The two
// trailing floats aren't read by the shader; they exist so this struct's C++
// size (80 bytes) matches std140's rounding of the block to a multiple of
// mat4's 16-byte alignment, the same concern notes.md walks through for
// LightData and the main UniformBufferObject.
struct FlameUniformBufferObject {
	glm::mat4 mvpMat;
	float seed;
	float _pad0, _pad1, _pad2;
};

// A single camera-facing quad in local XY (see createGlowMesh): position and
// billboard orientation are baked into mvpMat by the caller every frame (it
// needs the camera's own right/up, which Flame has no reason to know about),
// same struct shape as FlameUniformBufferObject so FlameGlow.vert/.frag can
// share its layout -- both are just "an mvp and a seed".
using GlowUniformBufferObject = FlameUniformBufferObject;

// One ember shard particle is a tiny tetrahedron (see createEmberMesh); every
// one of its 4 vertices carries the SAME particleSeed, so the shape moves as
// one rigid little chunk rather than its own corners drifting apart.
struct EmberVertex {
	glm::vec3 pos;
	float particleSeed;
};

class Flame {
	public:
	// maxInstances is fixed at init time (like MAX_LIGHTS): descriptor sets
	// come out of BaseProject's one shared pool, sized before it's created,
	// so "how many flames will ever exist" has to be known up front rather
	// than grown on demand. _DSglobal is the app's own already-mapped global
	// descriptor set (DSglobal in main.cpp): Flame binds that directly as
	// set 0 instead of keeping (and re-mapping) a duplicate.
	void init(BaseProject *_BP, DescriptorSetLayout *_DSLglobal, DescriptorSet *_DSglobal,
			  int maxInstances = 8);

	// Claims one of the pre-allocated instance slots. Returns its id, or -1
	// if maxInstances is already used up. seed offsets that instance's sway
	// phase so several flames (this torch, a future lantern) don't move in
	// lockstep.
	int spawn(float seed);

	// Call every frame for every spawned id, with that flame's full MVP (the
	// anchor instance's Wm times whatever local offset places the flame at
	// its head, times ViewPrj). Same idea as re-mapping a scene instance's
	// UBO: the command buffer is recorded once, only the buffer contents
	// change per frame.
	void update(int id, const glm::mat4 &mvpMat, int currentImage);

	// The soft glow billboard that fakes bloom around the flame (there's no
	// post-process pass in this renderer to do it for real -- see the notes
	// on FlameGlow.frag). Its MVP is a SEPARATE matrix from update()'s,
	// because a billboard has to face the camera every frame rather than
	// riding the flame's own orientation, and building that basis needs the
	// camera's world-space right/up, which only the caller (main.cpp) has.
	void updateGlow(int id, const glm::mat4 &mvpMat, int currentImage);

	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// Issued inline in the main pass, right after the scene's own draw
	// calls, so it shares the main RenderPass/depth buffer instead of
	// needing one of its own.
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;
	DescriptorSet *DSglobal = nullptr;	// set 0, owned by main.cpp

	VertexDescriptor VD;
	DescriptorSetLayout DSLflame;
	Pipeline P;
	Model *M = nullptr;

	// The glow billboard: its own tiny vertex format/layout/pipeline/mesh,
	// same set-0/set-1 shape as the flame itself, just a different shader
	// pair and geometry (one quad instead of the ring stack).
	VertexDescriptor VDglow;
	DescriptorSetLayout DSLglow;
	Pipeline Pglow;
	Model *Mglow = nullptr;
	std::vector<DescriptorSet> DSglow;	// index-matched with DS

	// Ember shard particles: small tetrahedra drifting up out of the crown.
	// No separate descriptor set of their own -- each shard's rise/drift is
	// entirely procedural (computed in Ember.vert from gubo.time and its own
	// per-particle seed baked into the mesh), so the only per-flame data they
	// need is the SAME mvp+seed the flame body already has in DS[]. Just a
	// different vertex format/pipeline/mesh, reusing DSLflame's layout.
	VertexDescriptor VDember;
	Pipeline Pember;
	Model *Member = nullptr;
	static constexpr int EMBER_COUNT = 7;

	int maxInstances = 0;
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;	// one per spawned instance, set 1
	std::vector<float> seeds;		// index-matched with DS

	void createMesh();
	void createGlowMesh();
	void createEmberMesh();

	// Deterministic hash instead of a real RNG: the mesh is built once and
	// baked into a static buffer, so this only ever needs to be repeatable,
	// not high quality.
	static float jitter(int a, int b);
};

#ifdef FLAME_IMPLEMENTATION

float Flame::jitter(int a, int b) {
	uint32_t h = (uint32_t)(a * 374761393 + b * 668265263 + 74848617);
	h = (h ^ (h >> 13)) * 1274126177u;
	h ^= (h >> 16);
	return (float)(h % 100000) / 100000.0f;	// [0, 1)
}

void Flame::init(BaseProject *_BP, DescriptorSetLayout *_DSLglobal, DescriptorSet *_DSglobal,
				  int _maxInstances) {
	BP = _BP;
	DSglobal = _DSglobal;
	maxInstances = _maxInstances;
	seeds.resize(maxInstances, 0.0f);

	VD.init(BP, {
			  {0, sizeof(FlameVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(FlameVertex, pos),
					 sizeof(glm::vec3), POSITION},
			  {0, 1, VK_FORMAT_R32_SFLOAT, offsetof(FlameVertex, tier),
					 sizeof(float), OTHER},
			  {0, 2, VK_FORMAT_R32_SFLOAT, offsetof(FlameVertex, swaySeed),
					 sizeof(float), OTHER}
			});

	DSLflame.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					sizeof(FlameUniformBufferObject), 1}
			  });

	VDglow.init(BP, {
				  {0, sizeof(glm::vec2), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32_SFLOAT, 0, sizeof(glm::vec2), OTHER}
				});
	// FRAGMENT_BIT too: FlameGlow.frag reads gubo.time itself (for the
	// flicker) AND this same block for its own seed/mvp -- both stages need
	// it, unlike the flame body's vertex-only block.
	DSLglow.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
					sizeof(GlowUniformBufferObject), 1}
			  });

	// One uniform block and one descriptor set per instance slot, for both
	// the flame body and its glow billboard, on top of whatever the rest of
	// the app already asked for. Set 0 (the global uniform) is NOT counted
	// here: Flame reuses main.cpp's existing DSglobal rather than
	// allocating its own copy.
	BP->DPSZs.uniformBlocksInPool += maxInstances * 2;
	BP->DPSZs.setsInPool += maxInstances * 2;

	P.init(BP, &VD, "shaders/Flame.vert.spv", "shaders/Flame.frag.spv",
					{_DSLglobal, &DSLflame});
	// Thin, jittered, faceted geometry: backface culling would punch holes
	// in it from some angles, so both sides render.
	P.setCullMode(VK_CULL_MODE_NONE);

	Pglow.init(BP, &VDglow, "shaders/FlameGlow.vert.spv", "shaders/FlameGlow.frag.spv",
					{_DSLglobal, &DSLglow});
	Pglow.setCullMode(VK_CULL_MODE_NONE);
	Pglow.setTransparency(true);
	// The glow sits at essentially the same depth as the flame body/other
	// glows it overlaps (several billboards centered near the same point):
	// LESS_OR_EQUAL is the same same-depth-overdraw fix UiQuad/TextMaker use
	// for their own stacked same-z quads, otherwise the second one drawn
	// silently fails the depth test against the first.
	Pglow.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	VDember.init(BP, {
				  {0, sizeof(EmberVertex), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(EmberVertex, pos),
						 sizeof(glm::vec3), POSITION},
				  {0, 1, VK_FORMAT_R32_SFLOAT, offsetof(EmberVertex, particleSeed),
						 sizeof(float), OTHER}
				});
	// Reuses DSLflame/DS[]: embers ride the same per-flame mvp+seed as the
	// body (see the EMBER_COUNT comment above), no descriptor pool budget of
	// their own to add.
	Pember.init(BP, &VDember, "shaders/Ember.vert.spv", "shaders/Ember.frag.spv",
					{_DSLglobal, &DSLflame});
	Pember.setCullMode(VK_CULL_MODE_NONE);
	Pember.setTransparency(true);
	Pember.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	createMesh();
	createGlowMesh();
	createEmberMesh();
}

void Flame::createGlowMesh() {
	// A single quad in local XY, corners at (+-1,+-1): FlameGlow.vert maps
	// it straight through to world space via a billboard basis the caller
	// builds (see main.cpp), and FlameGlow.frag reads the same local
	// position back as a UV to fade the glow out radially.
	static const glm::vec2 corners[4] = {
		{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}
	};

	Mglow = new Model();
	Mglow->indices = {0, 1, 2, 0, 2, 3};
	Mglow->vertices.resize(sizeof(corners));
	memcpy(Mglow->vertices.data(), corners, sizeof(corners));
	Mglow->initMesh(BP, &VDglow, false);
}

void Flame::createEmberMesh() {
	// One tiny tetrahedron per particle, all EMBER_COUNT of them baked into
	// one static mesh at the local origin: Ember.vert moves each one out to
	// its own rising/drifting position every frame using its particleSeed,
	// same idea as the flame body's swaySeed but per-shard instead of
	// per-vertex.
	static const glm::vec3 tetra[4] = {
		{0.0f, 0.06f, 0.0f}, {0.05f, -0.03f, 0.03f},
		{-0.05f, -0.03f, 0.03f}, {0.0f, -0.03f, -0.06f}
	};
	static const uint32_t tetraIdx[12] = {
		0, 1, 2,  0, 2, 3,  0, 3, 1,  1, 3, 2
	};

	std::vector<EmberVertex> verts;
	std::vector<uint32_t> idx;
	verts.reserve(EMBER_COUNT * 4);
	idx.reserve(EMBER_COUNT * 12);

	for(int p = 0; p < EMBER_COUNT; p++) {
		float seed = jitter(p, 55321);
		uint32_t base = (uint32_t)verts.size();
		for(int v = 0; v < 4; v++) {
			EmberVertex ev;
			ev.pos = tetra[v];
			ev.particleSeed = seed;
			verts.push_back(ev);
		}
		for(int i = 0; i < 12; i++) {
			idx.push_back(base + tetraIdx[i]);
		}
	}

	Member = new Model();
	Member->indices = idx;
	Member->vertices.resize(verts.size() * sizeof(EmberVertex));
	memcpy(Member->vertices.data(), verts.data(), Member->vertices.size());
	Member->initMesh(BP, &VDember, false);
}

void Flame::createMesh() {
	// A stubby, full-bellied ring stack (the "body") topped with several
	// separate jagged spikes (the "crown") instead of one shared apex point.
	// Real torch flames aren't a single smooth cone: several tongues lick
	// upward at uneven heights. One shared apex read as "simplistic" (a
	// perfect little pyramid); a crown of independently-heighted, slightly
	// inward-curling tips is what actually looks like fire while staying
	// just as low-poly.
	//
	// Heights/radii authored by eye against the torch model's own scale
	// (SM_Torch_Held_01 is about 1.1 units tall overall): tall enough that
	// the flame reads clearly above the head instead of disappearing next
	// to it, not measured against anything -- there's no "correct" size.
	static const float H[]  = {0.00f, 0.10f, 0.22f, 0.36f, 0.50f, 0.62f, 0.70f};	// body rings
	static const float Rr[] = {0.055f, 0.13f, 0.175f, 0.19f, 0.16f, 0.115f, 0.075f};	// belly, then narrows
	const int ringCount = 7;
	const int segCount = 10;			// crown ends up with 10 independent tips
	const float bodyTop = H[ringCount - 1];
	// How far an individual tip pokes up beyond the body, before its own
	// per-tip jitter: varies enough that the crown reads as uneven licks
	// rather than a fluted but still perfectly regular cone.
	const float TIP_EXTRA_MIN = 0.28f;
	const float TIP_EXTRA_MAX = 0.62f;
	const float tipMaxHeight = bodyTop + TIP_EXTRA_MAX;

	// Small per-vertex jitter so the body's silhouette looks hand-placed
	// rather than a perfect lathe shape -- pushed further than the first
	// pass (0.4/0.3): a smooth lathe shape with only mild jitter still reads
	// as tube-like rings from most angles, where the actual "faceted crystal"
	// references look genuinely irregular, no two facets alike. Fixed
	// fractions of the ring's own radius/spacing, not absolute units, so they
	// still make sense if H/Rr above are retuned.
	const float ANGLE_JITTER = 0.55f;	// radians, of a segCount-th of a turn
	const float RADIUS_JITTER = 0.42f;	// fraction of that ring's radius

	std::vector<FlameVertex> verts;
	verts.reserve(ringCount * segCount + segCount);

	auto ringXZ = [&](int r, int s, float &outX, float &outZ) {
		float baseAngle = 2.0f * (float)M_PI * (float)s / (float)segCount;
		float aJit = (jitter(r, s) - 0.5f) * 2.0f * ANGLE_JITTER;
		float rJit = 1.0f + (jitter(r, s + 1000) - 0.5f) * 2.0f * RADIUS_JITTER;
		float angle = baseAngle + aJit;
		float radius = Rr[r] * rJit;
		outX = std::cos(angle) * radius;
		outZ = std::sin(angle) * radius;
	};

	// Body: ringCount stacked, jittered rings. Vertices also get pulled up/
	// down off their ring's own plane (except the base, r=0, kept flat so it
	// sits flush in the torch cup): without that, every vertex on a ring
	// still lands on one flat disc no matter how much its angle/radius
	// jitter, and the mesh reads as a stack of rounded rings rather than the
	// irregular, no-two-facets-alike crystal look this is going for.
	for(int r = 0; r < ringCount; r++) {
		float spacingBelow = (r > 0) ? (H[r] - H[r - 1]) : (H[1] - H[0]);
		float spacingAbove = (r < ringCount - 1) ? (H[r + 1] - H[r]) : spacingBelow;
		float yJitterRange = 0.4f * std::min(spacingBelow, spacingAbove);

		for(int s = 0; s < segCount; s++) {
			float x, z;
			ringXZ(r, s, x, z);
			float yJit = (r == 0) ? 0.0f : (jitter(r, s + 2000) - 0.5f) * 2.0f * yJitterRange;
			float y = H[r] + yJit;

			FlameVertex v;
			v.pos = glm::vec3(x, y, z);
			v.tier = y / tipMaxHeight;
			v.swaySeed = jitter(r, s + 3000);
			verts.push_back(v);
		}
	}

	// Crown: one independent tip per top-ring segment, each with its own
	// height and a pull inward toward the axis (flame tongues thin and curl
	// in as they rise, they don't just spike straight up).
	int tipBase = (int)verts.size();
	for(int s = 0; s < segCount; s++) {
		float x, z;
		ringXZ(ringCount - 1, s, x, z);

		float extraJit = jitter(s, 7000);
		float extra = TIP_EXTRA_MIN + (TIP_EXTRA_MAX - TIP_EXTRA_MIN) * extraJit;
		float curl = 0.35f;	// fraction of the way tips pull toward the axis

		FlameVertex v;
		v.pos = glm::vec3(x * curl, bodyTop + extra, z * curl);
		v.tier = 1.0f;
		v.swaySeed = jitter(s, 9000);
		verts.push_back(v);
	}

	std::vector<uint32_t> idx;
	for(int r = 0; r < ringCount - 1; r++) {
		for(int s = 0; s < segCount; s++) {
			int s2 = (s + 1) % segCount;
			uint32_t i0 = r * segCount + s;
			uint32_t i1 = r * segCount + s2;
			uint32_t i2 = (r + 1) * segCount + s;
			uint32_t i3 = (r + 1) * segCount + s2;

			idx.push_back(i0); idx.push_back(i2); idx.push_back(i1);
			idx.push_back(i1); idx.push_back(i2); idx.push_back(i3);
		}
	}
	// One triangle per segment, from the top ring to that segment's OWN tip
	// (not a shared apex): going all the way around still closes the mesh
	// with no gaps, but now every other vertex on the rim is pulled up to a
	// different height, which is what makes the crown read as jagged.
	for(int s = 0; s < segCount; s++) {
		int s2 = (s + 1) % segCount;
		uint32_t i0 = (ringCount - 1) * segCount + s;
		uint32_t i1 = (ringCount - 1) * segCount + s2;
		uint32_t i2 = (uint32_t)(tipBase + s);
		idx.push_back(i0); idx.push_back(i1); idx.push_back(i2);
	}
	// No bottom cap: the base sits inside the torch head/holder, out of view.

	M = new Model();
	M->indices = idx;
	M->vertices.resize(verts.size() * sizeof(FlameVertex));
	memcpy(M->vertices.data(), verts.data(), M->vertices.size());
	M->initMesh(BP, &VD, false);
}

int Flame::spawn(float seed) {
	if(instanceCount >= maxInstances) {
		std::cout << "Flame: spawn() past maxInstances (" << maxInstances << "), ignored\n";
		return -1;
	}
	int id = instanceCount++;
	// DS itself is allocated in pipelinesAndDescriptorSetsInit(), once the
	// descriptor pool exists; spawn() only reserves the slot and its seed.
	seeds[id] = seed;
	return id;
}

void Flame::update(int id, const glm::mat4 &mvpMat, int currentImage) {
	if(id < 0 || id >= (int)DS.size()) {
		return;
	}
	FlameUniformBufferObject fubo{};
	fubo.mvpMat = mvpMat;
	fubo.seed = seeds[id];
	DS[id].map(currentImage, &fubo, 0);
}

void Flame::updateGlow(int id, const glm::mat4 &mvpMat, int currentImage) {
	if(id < 0 || id >= (int)DSglow.size()) {
		return;
	}
	GlowUniformBufferObject gubo{};
	gubo.mvpMat = mvpMat;
	gubo.seed = seeds[id];
	DSglow[id].map(currentImage, &gubo, 0);
}

void Flame::pipelinesAndDescriptorSetsInit(RenderPass *_RP) {
	RP = _RP;
	P.create(RP);
	Pglow.create(RP);
	Pember.create(RP);

	DS.resize(instanceCount);
	DSglow.resize(instanceCount);
	for(int i = 0; i < instanceCount; i++) {
		DS[i].init(BP, &DSLflame, {});
		DSglow[i].init(BP, &DSLglow, {});
	}
}

void Flame::pipelinesAndDescriptorSetsCleanup() {
	for(auto &d : DS) {
		d.cleanup();
	}
	for(auto &d : DSglow) {
		d.cleanup();
	}
	P.cleanup();
	Pglow.cleanup();
	Pember.cleanup();
}

void Flame::localCleanup() {
	if(M != nullptr) {
		M->cleanup();
	}
	if(Mglow != nullptr) {
		Mglow->cleanup();
	}
	if(Member != nullptr) {
		Member->cleanup();
	}
	DSLflame.cleanup();
	DSLglow.cleanup();
	P.destroy();
	Pglow.destroy();
	Pember.destroy();
}

void Flame::populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
	if(instanceCount == 0) {
		return;
	}

	P.bind(commandBuffer);
	M->bind(commandBuffer);
	// Same DSglobal main.cpp already maps every frame for the main pass;
	// bound once here since it's identical for every flame instance.
	DSglobal->bind(commandBuffer, P, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, P, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)M->indices.size(), 1, 0, 0, 0);
	}

	// Glow drawn after every flame body: it's transparent and reads best
	// layered on top of the (opaque, already depth-written) flames rather
	// than interleaved with them.
	Pglow.bind(commandBuffer);
	Mglow->bind(commandBuffer);
	DSglobal->bind(commandBuffer, Pglow, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DSglow[i].bind(commandBuffer, Pglow, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)Mglow->indices.size(), 1, 0, 0, 0);
	}

	// Embers last: reuse DS[] (the flame body's own mvp+seed), see the
	// EMBER_COUNT comment on why they need no descriptor set of their own.
	Pember.bind(commandBuffer);
	Member->bind(commandBuffer);
	DSglobal->bind(commandBuffer, Pember, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, Pember, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)Member->indices.size(), 1, 0, 0, 0);
	}
}

#endif
