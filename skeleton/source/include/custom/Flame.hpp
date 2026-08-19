// ***** CUSTOM *****

// A reusable animated torch flame. Built for the held torch, but nothing here
// is torch-specific: any caller just spawn()s a flame and feeds it a billboard
// matrix every frame (the same way SceneLights anchors a light to an instance
// + offset), so a lantern can get one later with one more call.
//
// WHAT THIS REPLACED, AND WHY. The first version was a low-poly mesh: a stack
// of jittered rings tapering to a jagged crown, deformed by noise in the vertex
// shader and shaded flat and opaque. It had three problems that no amount of
// extra geometry fixes:
//
//   1. It was opaque, with a hard silhouette. Real flame edges are the MOST
//      transparent part of the flame -- they thin out, they do not stop. A
//      crisp-edged solid reads as a painted object however well it wobbles.
//   2. It was fake-lit: a per-facet normal dotted against a hardcoded light
//      direction. That is directional shading on an EMITTER, which is what
//      made it read as an orange crystal. Fire has no lit side.
//   3. A surface of revolution, however deformed, can bulge and lean but can
//      never pinch off, detach a wisp, or open a hole -- which is most of what
//      fire visibly does.
//
// So the flame is now a VOLUME SHADED IN THE FRAGMENT SHADER rather than a
// deformed surface: three camera-facing quads at slightly different depths
// (createMesh()), through which Flame.frag renders a domain-warped FBM fire
// field with real alpha. That is fewer triangles than the mesh it replaces --
// 6 instead of 130 -- and it gets soft edges, internal structure and detaching
// wisps for free, because they are properties of the field rather than of the
// geometry.
//
// The glow billboard the old version carried is gone entirely. It existed to
// fake bloom, and there is now a real bloom chain downstream (see main.cpp's
// render graph), so the halo comes out of the HDR pipeline instead. That also
// disposes of its depth-write artefacts: Starter.hpp hardcodes
// depthWriteEnable = VK_TRUE on every pipeline, transparent ones included, so
// each glow quad used to write depth across its whole disc -- including the
// outer ring where its alpha was essentially zero -- and punch a circular hole
// in whatever was drawn behind it later. (A faint halo card was briefly tried
// again on top of the bloom chain and removed: even barely visible it read as
// a disc stamped behind the flame, and bloom already does the job.)
//
// Rendered as part of the main scene pass (main.cpp calls
// populateCommandBuffer() right after SC.populateCommandBuffer(), no separate
// RenderPass like UiQuad needs): that's what lets it depth-test against the
// castle/dungeon geometry the normal way, for free. That pass now targets an
// HDR RGBA16F attachment, which is why the shaders write colour values well
// above 1.0 -- see Flame.frag.
//
// Header-only module like the rest of custom/, implementation gated behind
// FLAME_IMPLEMENTATION (defined once in Libs.cpp). Assumes modules/Starter.hpp
// is already included by whoever includes this one.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// One corner of one billboard layer. `corner.x` runs -1..1 across the flame's
// half-width, `corner.y` runs 0..1 from the wick to the natural tip -- NOT
// -1..1, because a flame is anchored at its base and grows upward, and having
// y=0 mean "the wick" is what lets the shaders scale height and apply lean
// without first having to undo a centred quad.
struct FlameVertex {
	glm::vec2 corner;
	// Which of the three depth layers this quad belongs to (0, 1, 2). Read by
	// Flame.vert to offset the quad along the view axis and to give each layer
	// its own noise phase and scroll rate, so the three never sync up and read
	// as one card.
	float layer;
};

// One corner of one spark quad. Every one of a spark's 4 vertices carries the
// SAME seed, so the quad moves as a single particle rather than its corners
// drifting apart.
struct SparkVertex {
	glm::vec2 corner;	// -1..1 in both axes
	float seed;			// per-spark random in [0,1)
};

// Matches FlameUniformBufferObject in Flame.vert/Spark.vert field for field.
// 64 + 4 + 4 + 8 + 4 + 4 = 88 bytes: `lean` lands at offset 72, which is
// 8-aligned as std140 requires for a vec2, and the two trailing floats sit at
// 80/84 -- glm and std140 agree on every offset, so no alignas and no trailing
// pad -- the same concern notes.md walks through for LightData and the main
// UniformBufferObject.
struct FlameUniformBufferObject {
	glm::mat4 mvpMat;
	float seed;
	float intensity;	// BRIGHTNESS envelope only (spring-smoothed CPU-side);
						// no longer scales the card height -- heightScale does
	glm::vec2 lean;
	float heightScale;	// slow height envelope, ~0.78..1.09: a flame shortens
						// over a third of a second, it doesn't teleport
	float glareBoost;	// 1.0 + per-flame stare-at emphasis (see main.cpp's
						// glare block); multiplies the HDR output
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
	// if maxInstances is already used up. seed offsets that instance's noise
	// phase so several flames (this torch, a future lantern) don't animate in
	// lockstep.
	int spawn(float seed);

	// Call every frame for every spawned id.
	//
	//   mvpMat       that flame's billboard basis times ViewPrj. Local space is
	//                x = +/-1 across the half-width, y = 0 at the wick to 1 at
	//                the tip, z = toward the camera. main.cpp builds it.
	//   intensity    the BRIGHTNESS flicker/guttering envelope, ~0.30 to
	//                ~1.40, spring-smoothed. Simulated on the CPU rather than
	//                in the shader because the point light this torch casts
	//                has to flicker off the SAME signal -- see the TorchFlame
	//                struct in main.cpp.
	//   heightScale  the HEIGHT envelope, ~0.78..1.09 -- same underlying
	//                signal, compressed and chased much more slowly, because
	//                a flame's height varies less, and later, than its light
	//                output does. Splitting the two is what killed the old
	//                whole-flame "jumps".
	//   lean         how far the flame is dragged over by the hand carrying
	//                it, in billboard-local units.
	//   glareBoost   1.0 + stare-at emphasis for THIS flame, so a torch being
	//                looked at dead-on overdrives its own HDR output on top
	//                of the global exposure/bloom swell.
	//
	// Same idea as re-mapping a scene instance's UBO: the command buffer is
	// recorded once, only the buffer contents change per frame.
	void update(int id, const glm::mat4 &mvpMat, float intensity, float heightScale,
				const glm::vec2 &lean, float glareBoost, int currentImage);

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

	// Spark particles. No separate descriptor set of their own -- each spark's
	// whole lifecycle is procedural (computed in Spark.vert from gubo.time and
	// its own seed baked into the mesh), so the only per-flame data they need
	// is the SAME mvp/intensity/lean the flame body already has in DS[]. Just
	// a different vertex format/pipeline/mesh, reusing DSLflame's layout.
	VertexDescriptor VDspark;
	Pipeline Pspark;
	Model *Mspark = nullptr;

	// Three layers is the fewest that reads as having depth: one card is
	// obviously flat, two beat against each other, three fills in. More than
	// that stops being distinguishable and starts costing real overdraw, which
	// on a flame-sized quad is the only cost this effect has.
	static constexpr int LAYER_COUNT = 3;

	// Enough sparks that the eye reads a stream rather than counting them, few
	// enough that they stay flecks thrown off a flame rather than a plume of
	// their own. The old version had 7 four-triangle solids, which read as
	// orange confetti; 96 turned out to read as a bonfire. These are 2
	// triangles each and entirely procedural, so this costs no CPU at all and
	// the number is purely an aesthetic choice.
	static constexpr int SPARK_COUNT = 48;

	int maxInstances = 0;
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;	// one per spawned instance, set 1
	std::vector<float> seeds;		// index-matched with DS

	void createMesh();
	void createSparkMesh();

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

	// OTHER for both, not POSITION: the element type only matters when
	// Starter.hpp is filling a vertex buffer from a model file, and these two
	// meshes are built here by hand. (The old glow quad already relied on
	// this.) Nothing in the flame's local space is a world position anyway --
	// `corner` is a quad parameter the vertex shader turns into one.
	VD.init(BP, {
			  {0, sizeof(FlameVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(FlameVertex, corner),
					 sizeof(glm::vec2), OTHER},
			  {0, 1, VK_FORMAT_R32_SFLOAT, offsetof(FlameVertex, layer),
					 sizeof(float), OTHER}
			});

	// ALL_GRAPHICS, not VERTEX_BIT: Flame.frag reads intensity and the seed
	// for itself, not just the vertex shader.
	DSLflame.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
					sizeof(FlameUniformBufferObject), 1}
			  });

	VDspark.init(BP, {
				  {0, sizeof(SparkVertex), VK_VERTEX_INPUT_RATE_VERTEX}
				}, {
				  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(SparkVertex, corner),
						 sizeof(glm::vec2), OTHER},
				  {0, 1, VK_FORMAT_R32_SFLOAT, offsetof(SparkVertex, seed),
						 sizeof(float), OTHER}
				});

	// One uniform block and one descriptor set per instance slot, on top of
	// whatever the rest of the app already asked for. Set 0 (the global
	// uniform) is NOT counted here: Flame reuses main.cpp's existing DSglobal
	// rather than allocating its own copy, and the sparks reuse DS[] rather
	// than allocating one of theirs.
	BP->DPSZs.uniformBlocksInPool += maxInstances;
	BP->DPSZs.setsInPool += maxInstances;

	P.init(BP, &VD, "shaders/Flame.vert.spv", "shaders/Flame.frag.spv",
					{_DSLglobal, &DSLflame});
	// Camera-facing quads whose winding flips depending on which side of the
	// flame the camera has swung round to, so neither face can be culled.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// The three layers sit at slightly different depths and the sparks overlap
	// them, so same-depth fragments are common; LESS_OR_EQUAL is the same
	// stacked-quad fix UiQuad/TextMaker use, otherwise the second one drawn
	// silently fails the depth test against the first.
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	Pspark.init(BP, &VDspark, "shaders/Spark.vert.spv", "shaders/Spark.frag.spv",
					{_DSLglobal, &DSLflame});
	Pspark.setCullMode(VK_CULL_MODE_NONE);
	Pspark.setTransparency(true);
	Pspark.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	createMesh();
	createSparkMesh();
}

void Flame::createMesh() {
	// LAYER_COUNT camera-facing quads, all at the same place in local space --
	// Flame.vert is what pushes each one along the view axis and scales it, so
	// the mesh itself carries nothing but which layer each corner belongs to.
	//
	// Emitted in layer order, which is also BACK TO FRONT (layer 0 is the
	// farthest from the camera). That ordering is load-bearing: these quads are
	// alpha-blended, and Starter.hpp hardcodes depthWriteEnable = VK_TRUE with
	// no way to switch it off, so a nearer layer drawn first would write depth
	// and reject the farther ones behind it. Drawn far-to-near instead, every
	// layer passes the depth test and blends over what is already there.
	// (Flame.frag also discards near-zero alpha, so the invisible fringe never
	// writes depth at all -- between the two, the layers compose correctly.)
	std::vector<FlameVertex> verts;
	std::vector<uint32_t> idx;
	verts.reserve(LAYER_COUNT * 4);
	idx.reserve(LAYER_COUNT * 6);

	static const glm::vec2 corners[4] = {
		{-1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}
	};

	for(int l = 0; l < LAYER_COUNT; l++) {
		uint32_t base = (uint32_t)verts.size();
		for(int c = 0; c < 4; c++) {
			FlameVertex v;
			v.corner = corners[c];
			v.layer = (float)l;
			verts.push_back(v);
		}
		idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
		idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
	}

	M = new Model();
	M->indices = idx;
	M->vertices.resize(verts.size() * sizeof(FlameVertex));
	memcpy(M->vertices.data(), verts.data(), M->vertices.size());
	M->initMesh(BP, &VD, false);
}

void Flame::createSparkMesh() {
	// SPARK_COUNT quads baked into one static mesh, all at the local origin:
	// Spark.vert moves each one out to its own rising, drifting position every
	// frame from its seed, the same way the flame body's own motion is entirely
	// in its vertex shader. One draw call, no CPU particle system, no per-frame
	// buffer writes.
	static const glm::vec2 corners[4] = {
		{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}
	};

	std::vector<SparkVertex> verts;
	std::vector<uint32_t> idx;
	verts.reserve(SPARK_COUNT * 4);
	idx.reserve(SPARK_COUNT * 6);

	for(int p = 0; p < SPARK_COUNT; p++) {
		float seed = jitter(p, 55321);
		uint32_t base = (uint32_t)verts.size();
		for(int c = 0; c < 4; c++) {
			SparkVertex sv;
			sv.corner = corners[c];
			sv.seed = seed;
			verts.push_back(sv);
		}
		idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
		idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
	}

	Mspark = new Model();
	Mspark->indices = idx;
	Mspark->vertices.resize(verts.size() * sizeof(SparkVertex));
	memcpy(Mspark->vertices.data(), verts.data(), Mspark->vertices.size());
	Mspark->initMesh(BP, &VDspark, false);
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

void Flame::update(int id, const glm::mat4 &mvpMat, float intensity, float heightScale,
				   const glm::vec2 &lean, float glareBoost, int currentImage) {
	if(id < 0 || id >= (int)DS.size()) {
		return;
	}
	FlameUniformBufferObject fubo{};
	fubo.mvpMat = mvpMat;
	fubo.seed = seeds[id];
	fubo.intensity = intensity;
	fubo.lean = lean;
	fubo.heightScale = heightScale;
	fubo.glareBoost = glareBoost;
	DS[id].map(currentImage, &fubo, 0);
}

void Flame::pipelinesAndDescriptorSetsInit(RenderPass *_RP) {
	RP = _RP;
	P.create(RP);
	Pspark.create(RP);

	DS.resize(instanceCount);
	for(int i = 0; i < instanceCount; i++) {
		DS[i].init(BP, &DSLflame, {});
	}
}

void Flame::pipelinesAndDescriptorSetsCleanup() {
	for(auto &d : DS) {
		d.cleanup();
	}
	P.cleanup();
	Pspark.cleanup();
}

void Flame::localCleanup() {
	if(M != nullptr) {
		M->cleanup();
	}
	if(Mspark != nullptr) {
		Mspark->cleanup();
	}
	DSLflame.cleanup();
	P.destroy();
	Pspark.destroy();
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

	// Sparks after the bodies: they are thrown clear of the flame, so they
	// mostly sit in front of it, and drawing them second lets them blend over
	// whatever flame pixels they do overlap. They reuse DS[] (the flame body's
	// own uniform block), see the SPARK_COUNT comment on why they need no
	// descriptor set of their own.
	Pspark.bind(commandBuffer);
	Mspark->bind(commandBuffer);
	DSglobal->bind(commandBuffer, Pspark, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, Pspark, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)Mspark->indices.size(), 1, 0, 0, 0);
	}
}

#endif
