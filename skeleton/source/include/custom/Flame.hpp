// ***** CUSTOM *****

// A reusable animated flame. Nothing here is torch-specific: a caller spawn()s
// a flame and feeds it a billboard matrix every frame.
//
// It's not modelled geometry but a volume shaded in the fragment shader: three
// camera-facing quads at slightly different depths, through which Flame.frag
// renders a domain-warped FBM fire field with real alpha. Shape, structure,
// soft edges and wisps all come from that field; the geometry is six triangles.
// No glow billboard -- the flame writes values above 1.0 into the HDR
// attachment and the bloom chain makes the halo.
//
// Drawn inside the main scene pass after the scene's own draws, so it
// depth-tests against the level for free.
//
// Header-only, implementation gated behind FLAME_IMPLEMENTATION (Libs.cpp).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

// One corner of one billboard layer. corner.x is -1..1 across the half-width,
// corner.y is 0..1 from wick to tip -- not -1..1, so the shaders can scale the
// height and lean it without undoing a centred quad.
struct FlameVertex {
	glm::vec2 corner;
	// Which depth layer (0..2). Flame.vert offsets the quad along the view
	// axis and gives each layer its own noise phase, so they never sync up.
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
	// maxInstances is fixed here: descriptor sets come from BaseProject's
	// single pool, sized before creation. _DSglobal is main.cpp's own global
	// set, bound directly as set 0.
	void init(BaseProject *_BP, DescriptorSetLayout *_DSLglobal, DescriptorSet *_DSglobal,
			  int maxInstances = 8);

	// Claims one instance slot, returns its id or -1. `seed` offsets its noise
	// so flames don't animate in lockstep.
	int spawn(float seed);

	// Every frame, for every spawned id.
	//   mvpMat       billboard basis * ViewPrj, from main.cpp
	//   intensity    brightness envelope ~0.30..1.40, simulated on the CPU so
	//                the cast point light can flicker off the same signal
	//   heightScale  height envelope ~0.78..1.09; the same signal, chased more
	//                slowly, since height varies less and later than brightness
	//   lean         how far the carrying hand drags the flame over
	//   glareBoost   1.0 + emphasis when stared at
	//   color        target hue, hue-rotated onto the fire gradient; the cast
	//                light reads the same value
	// Only the buffer contents change per frame.
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

	// Spark particles. No descriptor set of their own: each spark's lifecycle
	// is computed in Spark.vert from gubo.time and a baked seed, so they reuse
	// DS[] -- just another vertex format, pipeline and mesh.
	VertexDescriptor VDspark;
	Pipeline Pspark;
	Model *Mspark = nullptr;

	// Fewest layers that reads as having depth: one card is obviously flat, two
	// beat against each other, three fills in. More only costs overdraw, which
	// is the only cost this effect has.
	static constexpr int LAYER_COUNT = 3;

	// Enough that the eye reads a stream, not individual sparks. Each is 2
	// procedural triangles, no CPU cost. Spark.vert splits the set by seed:
	// ~40% dust motes, the rest sparks, so the count covers both.
	static constexpr int SPARK_COUNT = 80;

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

	// OTHER, not POSITION: the element type only matters when Starter.hpp fills
	// a vertex buffer from a model file, and these are built by hand.
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

	// One uniform block and one set per instance. Set 0 isn't counted (it's
	// main.cpp's DSglobal), and the sparks reuse DS[].
	BP->DPSZs.uniformBlocksInPool += maxInstances;
	BP->DPSZs.setsInPool += maxInstances;

	P.init(BP, &VD, "shaders/fire/Flame.vert.spv", "shaders/fire/Flame.frag.spv",
					{_DSLglobal, &DSLflame});
	// The quads face the camera, so their winding flips as it swings around:
	// neither face may be culled.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// Layers and sparks often land at the same depth; with the default LESS the
	// second one drawn would silently fail the depth test.
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
	// LAYER_COUNT quads, all co-located in local space; Flame.vert pushes each
	// along the view axis. Emitted back to front (layer 0 farthest): the quads
	// are alpha-blended with depth writes on, so a nearer layer drawn first
	// would reject the ones behind it.
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

	// Sparks after the bodies: thrown clear of the flame, so drawing them
	// second lets them blend over the flame pixels they overlap. Reuse DS[].
	Pspark.bind(commandBuffer);
	Mspark->bind(commandBuffer);
	DSglobal->bind(commandBuffer, Pspark, 0, currentImage);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, Pspark, 1, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)Mspark->indices.size(), 1, 0, 0, 0);
	}
}

#endif
