// ***** CUSTOM *****

// The daylight behind the exit door: one quad, parked outside the castle in
// front of the doorway, that writes heavily overbright white into the HDR
// scene target so the bloom chain blows it into a glare the player cannot
// look straight at.
//
// Why this needs to exist at all, rather than just a very bright light:
// a light illuminates SURFACES, and there is nothing outside that door to
// illuminate -- the courtyard is gone, and past the east wall the level is
// the bare ground plane and empty sky. A light alone would have opened the
// door onto a dark hole with a lit floor in front of it. What sells "outside"
// is something bright actually VISIBLE through the opening, and that is a
// piece of geometry. (main.cpp still adds a spill light beside this, for the
// light this glow throws back into the room; the two are complementary, and
// neither reads right without the other.)
//
// The whole effect is one alpha-blended quad and a radial falloff. Everything
// that makes it look like light rather than a white rectangle happens
// downstream, for free, in machinery that already exists:
//   - the scene pass renders to RGBA16F, so `intensity` well above 1 survives
//     instead of clipping (same reason Flame.frag writes overbright);
//   - BloomBright.frag's threshold (1.55) is far below what this writes, so
//     every pixel of it blooms, and the blur bleeds the glare out over the
//     door frame and the wall around it -- which is exactly the "you cannot
//     see the edges of the doorway any more" part of the effect;
//   - the depth test does the masking: the quad sits BEHIND the wall, so the
//     arched hole in dvDoor is what shapes the light. Nothing here knows the
//     doorway's silhouette, and nothing here has to.
//
// It is drawn inline in the main scene pass (main.cpp calls
// populateCommandBuffer() next to Flame's), so it shares the depth buffer
// with the castle and needs no render pass of its own.
//
// Several quads, not one, because a single plane cannot cover what has to be
// covered. The exit door swings OUTWARD, so the light has to stand far enough
// out that the leaf never sweeps through it -- and at that distance a strip of
// open ground reappears under the bottom of the arch, between the threshold
// and the light, which is precisely the "something out there" the effect
// exists to deny. So main.cpp uses three: an upright wall of light past the
// leaf's reach, one lying flat on the ground in front of it covering the
// strip, and a third over the top of the arch, because the upright wall is
// finite in height too and a player at the threshold looking up sees over it
// into the skybox -- a gap no half-height can close, since the sightline's
// rise diverges as the player nears the wall. Between the three the opening
// frames nothing but white from every angle the player can stand at.
//
// Hence the instance pool. It is the same DS-per-instance arrangement Flame
// has, minus the spawn() bookkeeping: the count is fixed at init() because the
// level knows how many it wants.
//
// Header-only module like the rest of custom/, implementation gated behind
// EXITGLOW_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// modules/Starter.hpp is already included by whoever includes this one.

#include <cstring>
#include <vector>

// One corner of the quad. `corner` runs -1..1 in both axes and is a pure quad
// parameter, not a position: main.cpp's basis matrix is what turns it into
// one, so the same mesh serves whatever size and orientation the caller wants.
struct ExitGlowVertex {
	glm::vec2 corner;
};

// Matches ExitGlowUniformBufferObject in ExitGlow.vert/frag field for field.
// std140 puts `color` at offset 64 (a vec3 needs 16-byte alignment, and the
// mat4 ends exactly there), then packs the three trailing floats into the
// scalars at 76/80/84 -- the same layout arithmetic notes.md walks through for
// LightData. alignas(16) on the vec3 is what makes the C++ side agree.
struct ExitGlowUniformBufferObject {
	glm::mat4 mvpMat;
	alignas(16) glm::vec3 color;
	// Peak radiance at the centre of the field, in the same units the scene
	// is lit in. Everything above BLOOM_THRESHOLD blooms, so this is really
	// "how many times over the bloom threshold", and main.cpp drives it from
	// how far the door has swung: a closed door emits nothing.
	float intensity;
	// gubo.time equivalent, passed in rather than read from set 0 so this
	// pipeline needs no global descriptor set at all. Only used for a very
	// slow breathing, so the glare isn't a dead flat card.
	float time;
	// Width of the rim fade, 0..1 in quad coordinates. The field is flat and
	// fully blown out everywhere inside it; this is only what keeps the quad's
	// own edge from ever showing. See ExitGlow.frag.
	float softness;
};

struct ExitGlow {
	// No DSLglobal/DSglobal parameters, unlike Flame::init: this shader reads
	// nothing the app-wide uniform carries (no camera position, no lights, no
	// debug flags -- it is not shaded at all), so it binds a set of its own
	// as set 0 rather than taking a dependency on main.cpp's.
	//
	// `count` quads are allocated up front and every one of them is drawn
	// every frame; there is no spawn() and no way to skip one, so a quad the
	// caller has nothing to say about should simply be given intensity 0
	// (ExitGlow.frag discards on it).
	void init(BaseProject *_BP, int count = 3);

	// Call every frame, for every quad.
	//
	//   id         0 .. count-1, in the order the caller decided.
	//   mvpMat     that quad's basis times ViewPrj. Local space is x = +/-1
	//              across the half-width, y = +/-1 across the half-height,
	//              z = the plane normal. main.cpp builds it, which is what
	//              lets the same mesh serve both an upright quad and one
	//              lying flat on the ground.
	//   color      the daylight's hue. Near-white with a touch of warmth
	//              reads as sun rather than as a blank screen.
	//   intensity  see the field comment above; 0 draws nothing.
	//   time       seconds, for the breathing.
	//
	// Same idea as re-mapping a scene instance's UBO: the command buffer is
	// recorded once, only the buffer contents change per frame.
	void update(int id, const glm::mat4 &mvpMat, const glm::vec3 &color, float intensity,
				float time, int currentImage);

	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// Issued inline in the main pass, after the scene's own draw calls, so it
	// shares the RenderPass and depth buffer instead of needing its own.
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;

	VertexDescriptor VD;
	DescriptorSetLayout DSLglow;
	Pipeline P;
	Model *M = nullptr;

	// One set per quad, all sharing the single quad mesh and pipeline: the
	// only thing that differs between them is the basis matrix in their
	// uniform block. Allocated in pipelinesAndDescriptorSetsInit(), once the
	// descriptor pool exists.
	int instanceCount = 0;
	std::vector<DescriptorSet> DS;

	// Width of the rim fade. Small: the quad is a flat plateau of blown-out
	// white and this is only the sliver at its edge, which lives behind the
	// wall anyway. Baked in rather than exposed -- it trades against
	// `intensity`, and letting main.cpp move both independently would only
	// make the pair harder to tune.
	static constexpr float EDGE_SOFTNESS = 0.22f;

	void createMesh();
};

#ifdef EXITGLOW_IMPLEMENTATION

void ExitGlow::init(BaseProject *_BP, int count) {
	BP = _BP;
	instanceCount = count > 0 ? count : 0;

	// OTHER, not POSITION: the element type only matters when Starter.hpp
	// fills a vertex buffer from a model file, and this mesh is built here by
	// hand out of quad parameters rather than world positions.
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

	P.init(BP, &VD, "shaders/ExitGlow.vert.spv", "shaders/ExitGlow.frag.spv",
					{&DSLglow});
	// The door swings out past the plane of this quad, so the player can end
	// up looking at its back face through the open doorway. Culling nothing
	// costs one quad and removes the case where the light simply vanishes as
	// you step round the leaf.
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// Starter.hpp hardcodes depthWriteEnable, and the quad sits out on the
	// open ground where the sun-lit floor plane can land at a very similar
	// depth; LESS_OR_EQUAL is the same stacked-surface fix Flame/UiQuad use.
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

	// Recorded unconditionally, even while the door is shut. There is no
	// "skip this draw" available here -- the command buffer is built once and
	// replayed -- so an intensity of 0 is what turns a quad off, and
	// ExitGlow.frag discards on it rather than blending a black quad over the
	// ground outside.
	P.bind(commandBuffer);
	M->bind(commandBuffer);

	for(int i = 0; i < instanceCount; i++) {
		DS[i].bind(commandBuffer, P, 0, currentImage);
		vkCmdDrawIndexed(commandBuffer, (uint32_t)M->indices.size(), 1, 0, 0, 0);
	}
}

#endif
