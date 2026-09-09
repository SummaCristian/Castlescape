// ***** CUSTOM *****

// Reusable animated flame: caller spawn()s one, feeds it a billboard matrix per frame.
// Not modelled geometry: three camera-facing quads, Flame.frag renders an FBM fire
// field with real alpha; values >1.0 bloom via the HDR chain (no glow billboard).
// Drawn in the main scene pass, after scene draws, for free depth-testing.
// Header-only, gated behind FLAME_IMPLEMENTATION (Libs.cpp).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// corner.x: -1..1 across half-width. corner.y: 0..1 wick to tip (not -1..1, so
// shaders can scale/lean the height without undoing a centred quad).
struct FlameVertex {
	glm::vec2 corner;
	// Depth layer (0..2): Flame.vert offsets along view axis, own noise phase per layer.
	float layer;
};

// All 4 vertices of a spark share the SAME seed, so the quad moves as one particle.
struct SparkVertex {
	glm::vec2 corner;	// -1..1 in both axes
	float seed;			// per-spark random in [0,1)
};

// Matches FlameUniformBufferObject in Flame.vert/Spark.vert field for field.
// alignas(16) on `color`: std140's vec3 rule, else C++ offset (88) and shader offset (96) diverge.
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
	// maxInstances fixed here: descriptor sets sized from BaseProject's pool
	// before creation. _DSglobal is main.cpp's global set, bound as set 0.
	void init(BaseProject *_BP, DescriptorSetLayout *_DSLglobal, DescriptorSet *_DSglobal,
			  int maxInstances = 8);

	// Claims an instance slot, returns id or -1. `seed` offsets noise so flames don't sync.
	int spawn(float seed);

	// Per frame, per spawned id.
	//   mvpMat       billboard basis * ViewPrj
	//   intensity    brightness envelope ~0.30..1.40; drives the cast light's flicker too
	//   heightScale  height envelope ~0.78..1.09, chased more slowly than intensity
	//   lean         how far the carrying hand drags the flame over
	//   glareBoost   1.0 + emphasis when stared at
	//   color        target hue; the cast light reads the same value
	void update(int id, const glm::mat4 &mvpMat, float intensity, float heightScale,
				const glm::vec2 &lean, float glareBoost, const glm::vec3 &color,
				int currentImage);

	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// Recorded in the main pass, right after the scene's draws: shares its target/depth buffer.
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;
	DescriptorSet *DSglobal = nullptr;	// set 0, owned by main.cpp

	VertexDescriptor VD;
	DescriptorSetLayout DSLflame;
	Pipeline P;
	Model *M = nullptr;

	// Spark particles: lifecycle computed in Spark.vert from gubo.time + baked
	// seed, so no own descriptor set -- reuses DS[].
	VertexDescriptor VDspark;
	Pipeline Pspark;
	Model *Mspark = nullptr;

	// Fewest layers that read as having depth (more only costs overdraw).
	static constexpr int LAYER_COUNT = 3;

	// Enough to read as a stream, not individual sparks. Spark.vert splits by
	// seed: ~40% dust motes, rest sparks.
	static constexpr int SPARK_COUNT = 80;

	int maxInstances = 0;
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;	// one per spawned instance, set 1
	std::vector<float> seeds;		// index-matched with DS

	void createMesh();
	void createSparkMesh();

	// Deterministic hash, not RNG: mesh baked once into a static buffer, needs only repeatability.
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

	// OTHER, not POSITION: element type only matters for a Starter.hpp-loaded model; this is built by hand.
	VD.init(BP, {
			  {0, sizeof(FlameVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(FlameVertex, corner),
					 sizeof(glm::vec2), OTHER},
			  {0, 1, VK_FORMAT_R32_SFLOAT, offsetof(FlameVertex, layer),
					 sizeof(float), OTHER}
			});

	// ALL_GRAPHICS: Flame.frag reads intensity and seed too.
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

	// One uniform block + set per instance; set 0 (DSglobal) not counted, sparks reuse DS[].
	BP->DPSZs.uniformBlocksInPool += maxInstances;
	BP->DPSZs.setsInPool += maxInstances;

	P.init(BP, &VD, "shaders/fire/Flame.vert.spv", "shaders/fire/Flame.frag.spv",
					{_DSLglobal, &DSLflame});
	// Quads face the camera, winding flips as it swings: neither face may be culled.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// Layers/sparks often share depth; default LESS would fail the 2nd draw at that depth.
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	Pspark.init(BP, &VDspark, "shaders/fire/Spark.vert.spv", "shaders/fire/Spark.frag.spv",
					{_DSLglobal, &DSLflame});
	Pspark.setCullMode(VK_CULL_MODE_NONE);
	Pspark.setTransparency(true);
	Pspark.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	createMesh();
	createSparkMesh();
}

void Flame::createMesh() {
	// LAYER_COUNT quads, co-located; Flame.vert pushes each along the view axis.
	// Emitted back to front (layer 0 farthest): alpha-blend + depth-write means
	// a nearer layer drawn first would reject the ones behind it.
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
	// SPARK_COUNT quads at the local origin; Spark.vert moves each via its seed.
	// One draw call, no CPU particle system, no per-frame buffer writes.
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
	// DS allocated later in pipelinesAndDescriptorSetsInit(); this only reserves the slot/seed.
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
	// Bound once, same for every instance.
	DSglobal->bind(commandBuffer, P, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, P, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)M->indices.size(), 1, 0, 0, 0);
	}

	// Sparks after bodies: lets them blend over the flame pixels they overlap. Reuse DS[].
	Pspark.bind(commandBuffer);
	Mspark->bind(commandBuffer);
	DSglobal->bind(commandBuffer, Pspark, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, Pspark, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)Mspark->indices.size(), 1, 0, 0, 0);
	}
}

#endif
