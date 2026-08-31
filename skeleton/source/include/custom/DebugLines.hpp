// ***** CUSTOM *****
//
// The app's line renderer for debug overlays. It knows nothing about what the
// lines MEAN: callers build a list of world-space segments with the Push*
// helpers below and hand it over, so every overlay drawn as lines shares this
// one pipeline instead of cloning it. Today that's the light gizmos, the
// shadow-cube frustums and the collider wireframes, each cheat-menu gated
// (CheatFlags::showLightGizmos / showShadowFrustums / showColliders in
// main.cpp), which is also where the geometry and the colors are decided.
//
// The framework does ship its own collider visualizer (ColliderShow, in
// modules/Colliders.hpp), deliberately not used here: it wants a RenderPass of
// its own, caps out at MAX_COLLIDERS 20 (this scene has ~54), and re-records
// its command buffer whenever a collider moves. This class re-pushes every
// line from scratch each frame, so nothing has to be told that the world
// changed.
//
// VERTEX PULLING instead of a vertex buffer -- see DebugLines.vert's header
// for why: BaseProject::createBuffer() (Starter.hpp) is only reachable by
// its fixed friend list (Model, DescriptorSet, ...), which a new class can't
// join without editing that immutable file. So every line endpoint lives in
// a uniform buffer array instead, indexed by gl_VertexIndex, filled fresh
// every frame through the ordinary DescriptorSet::map() path -- the same
// idiom DSshadowCube[]/Flame already use for per-frame data.
//
// Fixed draw count: the main command buffer is recorded once per swapchain
// image and reused (main.cpp's populateCommandBuffer()), so vkCmdDraw's
// vertex count is baked in at record time and can't vary frame to frame.
// update() always fills exactly MAX_VERTS entries, padding unused slots with
// a zero-length "line" (both endpoints equal) that rasterizes to nothing --
// only the mapped CONTENTS change per frame, never the count.
//
// Rendered inline in the main scene pass, right after Flame's draw calls
// (main.cpp), so the lines depth-test against castle/dungeon geometry like
// everything else there -- no separate RenderPass needed.
//
// Header-only like the rest of custom/, implementation gated behind
// DEBUGLINES_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// modules/Starter.hpp is already included by whoever includes this one.

#include <algorithm>
#include <array>
#include <vector>

struct DebugLinesVPUBO {
	glm::mat4 vpMat;
};

class DebugLines {
	public:
	// 2048 rather than a few hundred because the collider overlay
	// (CheatFlags::showColliders) draws every gameplay collider at once: ~54
	// boxes today at 24 vertices each, and that list grows with every model
	// scene.json/colliders.json adds. Still well inside the guaranteed
	// maxUniformBufferRange of 64KB -- each array below is MAX_VERTS vec4s,
	// i.e. 32KB, so both fit with room to spare.
	static constexpr int MAX_VERTS = 2048;

	void init(BaseProject *_BP);
	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// pos/color must be the same length, one entry per vertex, in pairs (each
	// consecutive pair of vertices is one line segment) -- built with
	// PushLine/PushCross/PushBox below. Longer than MAX_VERTS is truncated:
	// the rest of that frame's lines are silently dropped rather than
	// overrunning the uniform arrays.
	void update(int currentImage, const glm::mat4 &vpMat,
				const std::vector<glm::vec4> &pos, const std::vector<glm::vec4> &color);

	// Issued inline in the main pass, right after Flame's draw calls -- see
	// this file's header for why it shares RP/depth instead of needing its
	// own RenderPass.
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	// Appends one line segment (2 vertices) in world space.
	static void PushLine(const glm::vec3 &a, const glm::vec3 &b, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// Appends a 3-axis cross (3 segments, 6 vertices) centered on `center`,
	// each arm `halfSize` long -- the light-position gizmo.
	static void PushCross(const glm::vec3 &center, float halfSize, const glm::vec4 &color,
						   std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// Appends a 12-edge axis-aligned wireframe cube (24 vertices) centered on
	// `center` with the given half-extent. EXACT, not approximate: a torch's
	// cube shadow is six 90-degree-FOV, 1:1-aspect perspective frustums
	// (computeShadowMatrices()/updateHandTorchShadow(), main.cpp), and at 90
	// degrees tan(45deg) = 1, so each face's visible extent at distance d is
	// exactly +-d in the other two axes -- the six faces' clip boundary really
	// is a literal cube of this half-extent, not a stand-in for one.
	static void PushBox(const glm::vec3 &center, float halfExtent, const glm::vec4 &color,
						 std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// Same 12 edges, but from an arbitrary min/max corner pair instead of a
	// cube's center+half-extent -- what Collider::getExtents() hands back, and
	// so what the collider overlay draws.
	static void PushAABB(const glm::vec3 &lo, const glm::vec3 &hi, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// A closed 4-point loop (4 segments, 8 vertices), in the given order. Used
	// for the ramps' inclined quads, which are the one piece of collision
	// geometry an axis-aligned box genuinely cannot stand in for.
	static void PushQuad(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c,
						  const glm::vec3 &d, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;

	VertexDescriptor VD;
	DescriptorSetLayout DSL;
	Pipeline P;
	DescriptorSet DS;
};

#ifdef DEBUGLINES_IMPLEMENTATION

void DebugLines::init(BaseProject *_BP) {
	BP = _BP;

	// No vertex attributes at all -- see this file's header. gl_VertexIndex
	// alone drives DebugLines.vert's lookup, so there is nothing per-vertex
	// to bind.
	VD.init(BP, {}, {});

	DSL.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					sizeof(DebugLinesVPUBO), 1},
				{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					(int)(sizeof(glm::vec4) * MAX_VERTS), 1},
				{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					(int)(sizeof(glm::vec4) * MAX_VERTS), 1}
			  });

	// Three uniform blocks, one descriptor set, on top of whatever the rest
	// of the app already asked for -- same accounting Flame/DSshadowCube[]
	// do at their own init().
	BP->DPSZs.uniformBlocksInPool += 3;
	BP->DPSZs.setsInPool += 1;

	P.init(BP, &VD, "shaders/DebugLines.vert.spv", "shaders/DebugLines.frag.spv", {&DSL});
	P.setTopology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
	P.setCullMode(VK_CULL_MODE_NONE);	// lines have no facing to cull
	// LESS_OR_EQUAL, not the default LESS: a gizmo/box edge that lands exactly
	// on a surface (e.g. a torch's own mesh) should still win the depth test,
	// the same reasoning Flame's stacked quads use.
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
}

void DebugLines::pipelinesAndDescriptorSetsInit(RenderPass *_RP) {
	RP = _RP;
	P.create(RP);
	DS.init(BP, &DSL, {});
}

void DebugLines::pipelinesAndDescriptorSetsCleanup() {
	DS.cleanup();
	P.cleanup();
}

void DebugLines::localCleanup() {
	DSL.cleanup();
	P.destroy();
}

void DebugLines::update(int currentImage, const glm::mat4 &vpMat,
						 const std::vector<glm::vec4> &pos, const std::vector<glm::vec4> &color) {
	DebugLinesVPUBO vpUbo{};
	vpUbo.vpMat = vpMat;
	DS.map(currentImage, &vpUbo, 0);

	// Padded to MAX_VERTS every frame: real vertices first, then every unused
	// slot repeats the LAST real vertex (or the origin, if there was none),
	// so every "line" past the real content has both endpoints equal and
	// rasterizes to nothing -- see this file's header for why the draw
	// call's vertex COUNT can never just shrink to match.
	std::array<glm::vec4, MAX_VERTS> posPad{};
	std::array<glm::vec4, MAX_VERTS> colorPad{};
	int n = std::min((int)pos.size(), MAX_VERTS);
	glm::vec4 lastPos = n > 0 ? pos[n - 1] : glm::vec4(0.0f);
	for(int i = 0; i < MAX_VERTS; i++) {
		posPad[i] = (i < n) ? pos[i] : lastPos;
		colorPad[i] = (i < n) ? color[i] : glm::vec4(0.0f);
	}

	DS.map(currentImage, posPad.data(), 1);
	DS.map(currentImage, colorPad.data(), 2);
}

void DebugLines::populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
	P.bind(commandBuffer);
	DS.bind(commandBuffer, P, 0, currentImage);
	vkCmdDraw(commandBuffer, MAX_VERTS, 1, 0, 0);
}

void DebugLines::PushLine(const glm::vec3 &a, const glm::vec3 &b, const glm::vec4 &color,
						   std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	pos.push_back(glm::vec4(a, 1.0f));
	pos.push_back(glm::vec4(b, 1.0f));
	colorOut.push_back(color);
	colorOut.push_back(color);
}

void DebugLines::PushCross(const glm::vec3 &center, float halfSize, const glm::vec4 &color,
							std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	PushLine(center - glm::vec3(halfSize, 0, 0), center + glm::vec3(halfSize, 0, 0), color, pos, colorOut);
	PushLine(center - glm::vec3(0, halfSize, 0), center + glm::vec3(0, halfSize, 0), color, pos, colorOut);
	PushLine(center - glm::vec3(0, 0, halfSize), center + glm::vec3(0, 0, halfSize), color, pos, colorOut);
}

void DebugLines::PushBox(const glm::vec3 &center, float halfExtent, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	PushAABB(center - glm::vec3(halfExtent), center + glm::vec3(halfExtent), color, pos, colorOut);
}

void DebugLines::PushAABB(const glm::vec3 &lo, const glm::vec3 &hi, const glm::vec4 &color,
						   std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	// Corner index bits: 4*xBit + 2*yBit + zBit, each bit 0 (lo) or 1 (hi) --
	// matches the nested loop's emission order below.
	glm::vec3 c[8];
	int idx = 0;
	for(int sx = 0; sx <= 1; sx++)
		for(int sy = 0; sy <= 1; sy++)
			for(int sz = 0; sz <= 1; sz++)
				c[idx++] = glm::vec3(sx ? hi.x : lo.x, sy ? hi.y : lo.y, sz ? hi.z : lo.z);

	// The 12 edges of a box: two corners connected by an edge differ in
	// exactly one bit. Grouped by which axis differs.
	static const int edges[12][2] = {
		{0,1}, {2,3}, {4,5}, {6,7},	// z-edges (bit 0 differs)
		{0,2}, {1,3}, {4,6}, {5,7},	// y-edges (bit 1 differs)
		{0,4}, {1,5}, {2,6}, {3,7}	// x-edges (bit 2 differs)
	};
	for(auto &e : edges) {
		PushLine(c[e[0]], c[e[1]], color, pos, colorOut);
	}
}

void DebugLines::PushQuad(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c,
						   const glm::vec3 &d, const glm::vec4 &color,
						   std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	PushLine(a, b, color, pos, colorOut);
	PushLine(b, c, color, pos, colorOut);
	PushLine(c, d, color, pos, colorOut);
	PushLine(d, a, color, pos, colorOut);
}

#endif
