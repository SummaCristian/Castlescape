// ***** CUSTOM *****
// Overbright quads outside the exit door; bloom turns them into unlookable glare.
// Geometry not a light, since there's nothing outside to illuminate.
// Several quads (upright wall, ground strip, arch) so the opening frames only white.
// Header-only, implementation gated behind EXITGLOW_IMPLEMENTATION (Libs.cpp).

#include <cstring>
#include <vector>

// Quad corner, -1..1 in both axes; main.cpp's basis matrix places it.
struct ExitGlowVertex {
	glm::vec2 corner;
};

// Matches ExitGlowUniformBufferObject in ExitGlow.vert/frag; alignas(16) for std140 layout.
struct ExitGlowUniformBufferObject {
	glm::mat4 mvpMat;
	alignas(16) glm::vec3 color;
	// Peak radiance in scene units (multiples of bloom threshold); 0 = door shut.
	float intensity;
	// Seconds, for slow breathing effect.
	float time;
	// Rim fade width, 0..1 in quad coords. See ExitGlow.frag.
	float softness;
};

struct ExitGlow {
	// No DSLglobal: reads nothing app-wide, own set 0. count quads allocated up front, no spawn().
	void init(BaseProject *_BP, int count = 3);

	// id: 0..count-1. mvpMat: quad basis * ViewPrj. intensity 0 draws nothing.
	void update(int id, const glm::mat4 &mvpMat, const glm::vec3 &color, float intensity,
				float time, int currentImage);

	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;

	VertexDescriptor VD;
	DescriptorSetLayout DSLglow;
	Pipeline P;
	Model *M = nullptr;

	// One set per quad; only the basis matrix differs.
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;

	static constexpr float EDGE_SOFTNESS = 0.22f;

	void createMesh();
};

#ifdef EXITGLOW_IMPLEMENTATION

void ExitGlow::init(BaseProject *_BP, int count) {
	BP = _BP;
	instanceCount = count > 0 ? count : 0;

	// OTHER, not POSITION: mesh is built here by hand, not from a model file.
	VD.init(BP, {
			  {0, sizeof(ExitGlowVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ExitGlowVertex, corner),
					 sizeof(glm::vec2), OTHER}
			});

	// ALL_GRAPHICS: vertex shader needs mvpMat, fragment shader needs the rest.
	DSLglow.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
					sizeof(ExitGlowUniformBufferObject), 1}
			  });

	BP->DPSZs.uniformBlocksInPool += instanceCount;
	BP->DPSZs.setsInPool += instanceCount;

	P.init(BP, &VD, "shaders/exit/ExitGlow.vert.spv", "shaders/exit/ExitGlow.frag.spv",
					{&DSLglow});
	// No culling: player sees the quad's back face through the opening.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// LESS_OR_EQUAL: ground plane can land at a similar depth (same fix as Flame/UiQuad).
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	createMesh();
}

void ExitGlow::createMesh() {
	// Single quad, centred on origin; no subdivision, falloff is per-fragment.
	static const ExitGlowVertex verts[4] = {
		{{-1.0f, -1.0f}}, {{1.0f, -1.0f}}, {{1.0f, 1.0f}}, {{-1.0f, 1.0f}}
	};

	M = new Model();
	M->indices = {0, 1, 2, 0, 2, 3};
	M->vertices.resize(sizeof(verts));
	memcpy(M->vertices.data(), verts, sizeof(verts));
	M->initMesh(BP, &VD, false);
}

void ExitGlow::update(int id, const glm::mat4 &mvpMat, const glm::vec3 &color, float intensity,
					  float time, int currentImage) {
	if(id < 0 || id >= (int)DS.size()) {
		return;
	}
	ExitGlowUniformBufferObject gubo{};
	gubo.mvpMat = mvpMat;
	gubo.color = color;
	gubo.intensity = intensity;
	gubo.time = time;
	gubo.softness = EDGE_SOFTNESS;
	DS[id].map(currentImage, &gubo, 0);
}

void ExitGlow::pipelinesAndDescriptorSetsInit(RenderPass *_RP) {
	RP = _RP;
	P.create(RP);

	DS.resize(instanceCount);
	for(int i = 0; i < instanceCount; i++) {
		DS[i].init(BP, &DSLglow, {});
	}
}

void ExitGlow::pipelinesAndDescriptorSetsCleanup() {
	for(auto &d : DS) {
		d.cleanup();
	}
	P.cleanup();
}

void ExitGlow::localCleanup() {
	if(M != nullptr) {
		M->cleanup();
	}
	DSLglow.cleanup();
	P.destroy();
}

void ExitGlow::populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
	if(instanceCount == 0) {
		return;
	}

	// Recorded unconditionally: intensity 0 turns a quad off via discard in ExitGlow.frag.
	P.bind(commandBuffer);
	M->bind(commandBuffer);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, P, 0, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)M->indices.size(), 1, 0, 0, 0);
	}
}

#endif
