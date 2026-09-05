// ***** CUSTOM *****

// A reusable animated flame. Nothing here is torch-specific: a caller spawn()s
// a flame and feeds it a billboard matrix every frame.
//
// The flame is not modelled geometry, it is a volume shaded in the fragment
// shader: three camera-facing quads at slightly different depths, through which
// Flame.frag renders a domain-warped FBM fire field with real alpha. Shape,
// internal structure, soft edges and detaching wisps all come from that field;
// the geometry is six triangles.
//
// There is no glow billboard: the flame writes color values above 1.0 into the
// HDR attachment and the bloom chain downstream turns them into a halo.
//
// Drawn inside the main scene pass (main.cpp calls populateCommandBuffer()
// right after the scene's own), so it depth-tests against the level geometry
// for free, unlike UiQuad which needs its own render pass.
//
// Header-only like the rest of custom/: the implementation is compiled only
// where FLAME_IMPLEMENTATION is defined (Libs.cpp). Assumes modules/Starter.hpp
// was included first.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// One corner of one billboard layer. corner.x is -1..1 across the half-width,
// corner.y is 0..1 from the wick to the tip - not -1..1, because a flame is
// anchored at its base, and having y=0 mean "the wick" lets the shaders scale
// the height and lean it over without undoing a centred quad first.
struct FlameVertex {
	glm::vec2 corner;
	// Which depth layer this quad belongs to (0..2). Flame.vert uses it to
	// offset the quad along the view axis and to give each layer its own noise
	// phase and scroll speed, so they never sync up and read as one card.
	float layer;
};

// One corner of one spark quad. All 4 vertices of a spark carry the SAME seed,
// so the quad moves as one particle instead of its corners drifting apart.
struct SparkVertex {
	glm::vec2 corner;	// -1..1 in both axes
	float seed;			// per-spark random in [0,1)
};

// Matches FlameUniformBufferObject in Flame.vert/Spark.vert field for field.
// The alignas(16) on `color` is std140's rule for a vec3: without it C++ would
// put it at offset 88 and the shader would read it at 96.
struct FlameUniformBufferObject {
	glm::mat4 mvpMat;
	float seed;
	float intensity;	// brightness envelope, smoothed on the CPU
	glm::vec2 lean;		// how far the flame is dragged over, billboard-local
	float heightScale;	// height envelope, ~0.78..1.09, chased more slowly
	float glareBoost;	// 1.0 + emphasis when stared at; scales the HDR output
	alignas(16) glm::vec3 color;	// target hue, see Flame.frag
};

class Flame {
	public:
	// maxInstances is fixed here because descriptor sets come out of
	// BaseProject's single pool, which is sized before it is created: the
	// number of flames that will ever exist has to be known up front.
	// _DSglobal is main.cpp's own global descriptor set, bound directly as
	// set 0 instead of keeping a duplicate.
	void init(BaseProject *_BP, DescriptorSetLayout *_DSLglobal, DescriptorSet *_DSglobal,
			  int maxInstances = 8);

	// Claims one instance slot and returns its id, or -1 if none are left.
	// `seed` offsets that instance's noise so several flames don't animate in
	// lockstep.
	int spawn(float seed);

	// Call every frame for every spawned id.
	//
	//   mvpMat       billboard basis times ViewPrj, built by main.cpp
	//   intensity    brightness envelope, ~0.30..1.40. Simulated on the CPU,
	//                not in the shader, because the point light the torch
	//                casts has to flicker off the same signal
	//   heightScale  height envelope, ~0.78..1.09. The same signal compressed
	//                and chased more slowly, since a flame's height varies
	//                less, and later, than its brightness
	//   lean         how far the flame is dragged over by the hand carrying it
	//   glareBoost   1.0 + emphasis when this flame is stared at
	//   color        target hue, hue-rotated onto the fire gradient in
	//                Flame.frag. The cast light reads the same value, so flame
	//                and light always agree
	//
	// Only the buffer contents change per frame; the command buffer is recorded
	// once.
	void update(int id, const glm::mat4 &mvpMat, float intensity, float heightScale,
				const glm::vec2 &lean, float glareBoost, const glm::vec3 &color,
				int currentImage);

	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// Recorded inline in the main pass, right after the scene's own draw calls,
	// so it shares that pass's render target and depth buffer.
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;
	DescriptorSet *DSglobal = nullptr;	// set 0, owned by main.cpp

	VertexDescriptor VD;
	DescriptorSetLayout DSLflame;
	Pipeline P;
	Model *M = nullptr;

	// Spark particles. They need no descriptor set of their own: each spark's
	// whole lifecycle is computed in Spark.vert from gubo.time and a seed baked
	// into the mesh, so the only per-flame data they use is the same
	// mvp/intensity/lean already in DS[]. Just another vertex format, pipeline
	// and mesh, reusing DSLflame's layout.
	VertexDescriptor VDspark;
	Pipeline Pspark;
	Model *Mspark = nullptr;

	// Fewest layers that reads as having depth: one card is obviously flat, two
	// beat against each other, three fills in. More only costs overdraw, which
	// is the only cost this effect has.
	static constexpr int LAYER_COUNT = 3;

	// Purely aesthetic: enough that the eye reads a stream instead of counting
	// them. Each spark is 2 triangles and fully procedural, so this costs no
	// CPU time.
	static constexpr int SPARK_COUNT = 48;

	int maxInstances = 0;
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;	// one per spawned instance, set 1
	std::vector<float> seeds;		// index-matched with DS

	void createMesh();
	void createSparkMesh();

	// Deterministic hash instead of an RNG: the mesh is built once and baked
	// into a static buffer, so this only has to be repeatable.
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

	// OTHER, not POSITION: the element type only matters when Starter.hpp
	// fills a vertex buffer from a model file, and these meshes are built by
	// hand below. `corner` isn't a position anyway, it's a quad parameter.
	VD.init(BP, {
			  {0, sizeof(FlameVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(FlameVertex, corner),
					 sizeof(glm::vec2), OTHER},
			  {0, 1, VK_FORMAT_R32_SFLOAT, offsetof(FlameVertex, layer),
					 sizeof(float), OTHER}
			});

	// ALL_GRAPHICS, not VERTEX_BIT: Flame.frag reads intensity and seed too.
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

	// Book one uniform block and one descriptor set per instance in the shared
	// pool. Set 0 isn't counted: it is main.cpp's existing DSglobal, and the
	// sparks reuse DS[] rather than allocating their own.
	BP->DPSZs.uniformBlocksInPool += maxInstances;
	BP->DPSZs.setsInPool += maxInstances;

	P.init(BP, &VD, "shaders/Flame.vert.spv", "shaders/Flame.frag.spv",
					{_DSLglobal, &DSLflame});
	// The quads face the camera, so their winding flips as it swings around:
	// neither face may be culled.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// Layers and sparks often land at the same depth; with the default LESS the
	// second one drawn would silently fail the depth test.
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
	// LAYER_COUNT quads, all in the same place in local space: Flame.vert is
	// what pushes each along the view axis and scales it, so the mesh carries
	// only which layer a corner belongs to.
	//
	// Emitted back to front (layer 0 is farthest). This order matters: the
	// quads are alpha-blended, and Starter.hpp always enables depth writes, so
	// a nearer layer drawn first would write depth and reject the ones behind
	// it. Flame.frag also discards near-zero alpha, so the invisible fringe
	// never writes depth either.
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
	// SPARK_COUNT quads in one static mesh, all at the local origin: Spark.vert
	// moves each to its own rising, drifting position from its seed. One draw
	// call, no CPU particle system, no per-frame buffer writes.
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
	// DS is allocated later, in pipelinesAndDescriptorSetsInit(), once the
	// descriptor pool exists. spawn() only reserves the slot and its seed.
	seeds[id] = seed;
	return id;
}

void Flame::update(int id, const glm::mat4 &mvpMat, float intensity, float heightScale,
				   const glm::vec2 &lean, float glareBoost, const glm::vec3 &color,
				   int currentImage) {
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
	fubo.color = color;
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
	// main.cpp already fills this every frame; bound once, it is the same for
	// every instance.
	DSglobal->bind(commandBuffer, P, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, P, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)M->indices.size(), 1, 0, 0, 0);
	}

	// Sparks after the bodies: they are thrown clear of the flame and so end up
	// in front of it, and drawing them second lets them blend over the flame
	// pixels they do overlap. They reuse DS[], the flame body's uniform block.
	Pspark.bind(commandBuffer);
	Mspark->bind(commandBuffer);
	DSglobal->bind(commandBuffer, Pspark, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, Pspark, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)Mspark->indices.size(), 1, 0, 0, 0);
	}
}

#endif
