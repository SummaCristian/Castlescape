// ***** CUSTOM *****

// The daylight behind the exit door: quads parked outside the doorway that
// write heavily overbright white into the HDR scene target, so the bloom chain
// blows them into a glare the player can't look straight at.
//
// Why geometry, not just a bright light: a light illuminates SURFACES, and
// there's nothing outside that door -- past the east wall it's bare ground and
// empty sky. What sells "outside" is something bright VISIBLE through the
// opening. (main.cpp still adds a spill light for what the glow throws back
// into the room; the two are complementary.)
//
// The effect itself is one alpha-blended quad and a falloff. The rest happens
// downstream for free: the RGBA16F target keeps `intensity` past 1, the bloom
// threshold (1.55) is far below what this writes so every pixel blooms and
// bleeds over the door frame, and the depth test does the masking -- the quad
// sits BEHIND the wall, so the arched hole shapes the light. Drawn inline in
// the main pass, sharing the depth buffer.
//
// Several quads, not one: the door swings OUTWARD, so the light must stand
// clear of the leaf, and at that distance a strip of ground reappears under
// the arch. main.cpp uses three -- an upright wall past the leaf, one flat on
// the ground covering the strip, and one over the arch (a player at the
// threshold looking up sees over a finite wall into the skybox). Between them
// the opening frames nothing but white from any angle.
//
// Hence the instance pool: the same DS-per-instance arrangement Flame has,
// minus spawn() -- the count is fixed at init().
//
// Header-only, implementation gated behind EXITGLOW_IMPLEMENTATION (Libs.cpp).

#include <cstring>
#include <vector>

// One corner of the quad. `corner` runs -1..1 in both axes -- a quad
// parameter, not a position; main.cpp's basis matrix places it.
struct ExitGlowVertex {
	glm::vec2 corner;
};

// Matches ExitGlowUniformBufferObject in ExitGlow.vert/frag. std140 puts
// `color` at offset 64 and packs the trailing floats after it; alignas(16) on
// the vec3 is what makes the C++ side agree (notes.md, same as LightData).
struct ExitGlowUniformBufferObject {
	glm::mat4 mvpMat;
	alignas(16) glm::vec3 color;
	// Peak centre radiance in scene units, i.e. "how many times over the bloom
	// threshold". main.cpp drives it from the door's swing; a shut door is 0.
	float intensity;
	// gubo.time equivalent, passed in so this pipeline needs no global set.
	// Only a very slow breathing, so the glare isn't a dead flat card.
	float time;
	// Rim fade width, 0..1 in quad coords. Only keeps the quad's edge from
	// showing; everything inside it is flat and blown out. See ExitGlow.frag.
	float softness;
};

struct ExitGlow {
	// No DSLglobal, unlike Flame::init: this shader reads nothing the app-wide
	// uniform carries, so it binds a set of its own as set 0. `count` quads are
	// allocated up front and all drawn every frame; there's no spawn(), so an
	// unused quad is given intensity 0 (ExitGlow.frag discards on it).
	void init(BaseProject *_BP, int count = 3);

	// Every frame, for every quad.
	//   id         0 .. count-1
	//   mvpMat     that quad's basis * ViewPrj; local space is x/y = +/-1,
	//              z = normal. main.cpp builds it, so the mesh serves both an
	//              upright quad and one flat on the ground.
	//   color      daylight hue; near-white with a touch of warmth
	//   intensity  see the field comment; 0 draws nothing
	//   time       seconds, for the breathing
	// Same as re-mapping a scene instance's UBO: recorded once, contents change.
	void update(int id, const glm::mat4 &mvpMat, const glm::vec3 &color, float intensity,
				float time, int currentImage);

	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// Issued inline in the main pass after the scene's draws.
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;

	VertexDescriptor VD;
	DescriptorSetLayout DSLglow;
	Pipeline P;
	Model *M = nullptr;

	// One set per quad, all sharing the mesh and pipeline; only the basis
	// matrix in the uniform block differs.
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;

	// Rim fade width. Baked in, not exposed: it trades against `intensity`,
	// and moving both independently would only be harder to tune.
	static constexpr float EDGE_SOFTNESS = 0.22f;

	void createMesh();
};

#ifdef EXITGLOW_IMPLEMENTATION

void ExitGlow::init(BaseProject *_BP, int count) {
	BP = _BP;
	instanceCount = count > 0 ? count : 0;

	// OTHER, not POSITION: the element type only matters when Starter.hpp fills
	// a vertex buffer from a model file, and this mesh is built here by hand.
	VD.init(BP, {
			  {0, sizeof(ExitGlowVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ExitGlowVertex, corner),
					 sizeof(glm::vec2), OTHER}
			});

	// ALL_GRAPHICS: the vertex shader needs mvpMat and the fragment shader
	// needs everything else out of the same block.
	DSLglow.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS,
					sizeof(ExitGlowUniformBufferObject), 1}
			  });

	// One block and one set per quad, on top of whatever the rest of the app
	// asked for.
	BP->DPSZs.uniformBlocksInPool += instanceCount;
	BP->DPSZs.setsInPool += instanceCount;

	P.init(BP, &VD, "shaders/exit/ExitGlow.vert.spv", "shaders/exit/ExitGlow.frag.spv",
					{&DSLglow});
	// The door swings out past this quad, so the player can see its back face
	// through the opening. Culling nothing avoids the light vanishing.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// depthWriteEnable is hardcoded on and the ground plane can land at a
	// similar depth; LESS_OR_EQUAL is the same fix Flame/UiQuad use.
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

	createMesh();
}

void ExitGlow::createMesh() {
	// A single quad, centred on the local origin. No subdivision: the falloff
	// is computed per fragment, so extra vertices would buy nothing.
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

	// Recorded unconditionally, even with the door shut: the command buffer is
	// built once, so intensity 0 turns a quad off and ExitGlow.frag discards.
	P.bind(commandBuffer);
	M->bind(commandBuffer);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, P, 0, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)M->indices.size(), 1, 0, 0, 0);
	}
}

#endif
