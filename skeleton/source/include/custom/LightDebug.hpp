// ***** CUSTOM *****
//
// Debug line overlay for lights/shadows: colored crosses at each light's
// position (or an arrow for the sun, which has none), and wireframe boxes
// at each torch's shadow-cube near/far clip distance. Cheat-menu gated
// (CheatFlags::showLightGizmos / showShadowFrustums in main.cpp).
//
// VERTEX PULLING instead of a vertex buffer -- see LightDebug.vert's header
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
// (main.cpp), so gizmos/frustums depth-test against castle/dungeon geometry
// like everything else there -- no separate RenderPass needed.
//
// Header-only like the rest of custom/, implementation gated behind
// LIGHTDEBUG_IMPLEMENTATION (defined once in Libs.cpp). Assumes
// modules/Starter.hpp is already included by whoever includes this one.

#include <algorithm>
#include <array>
#include <vector>

struct LightDebugVPUBO {
	glm::mat4 vpMat;
};

class LightDebug {
	public:
	static constexpr int MAX_VERTS = 512;

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

	private:
	BaseProject *BP = nullptr;
	RenderPass *RP = nullptr;

	VertexDescriptor VD;
	DescriptorSetLayout DSL;
	Pipeline P;
	DescriptorSet DS;
};

#ifdef LIGHTDEBUG_IMPLEMENTATION

void LightDebug::init(BaseProject *_BP) {
	BP = _BP;

	// No vertex attributes at all -- see this file's header. gl_VertexIndex
	// alone drives LightDebug.vert's lookup, so there is nothing per-vertex
	// to bind.
	VD.init(BP, {}, {});

	DSL.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					sizeof(LightDebugVPUBO), 1},
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

	P.init(BP, &VD, "shaders/LightDebug.vert.spv", "shaders/LightDebug.frag.spv", {&DSL});
	P.setTopology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
	P.setCullMode(VK_CULL_MODE_NONE);	// lines have no facing to cull
	// LESS_OR_EQUAL, not the default LESS: a gizmo/box edge that lands exactly
	// on a surface (e.g. a torch's own mesh) should still win the depth test,
	// the same reasoning Flame's stacked quads use.
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
}

void LightDebug::pipelinesAndDescriptorSetsInit(RenderPass *_RP) {
	RP = _RP;
	P.create(RP);
	DS.init(BP, &DSL, {});
}

void LightDebug::pipelinesAndDescriptorSetsCleanup() {
	DS.cleanup();
	P.cleanup();
}

void LightDebug::localCleanup() {
	DSL.cleanup();
	P.destroy();
}

void LightDebug::update(int currentImage, const glm::mat4 &vpMat,
						 const std::vector<glm::vec4> &pos, const std::vector<glm::vec4> &color) {
	LightDebugVPUBO vpUbo{};
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

void LightDebug::populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
	P.bind(commandBuffer);
	DS.bind(commandBuffer, P, 0, currentImage);
	vkCmdDraw(commandBuffer, MAX_VERTS, 1, 0, 0);
}

void LightDebug::PushLine(const glm::vec3 &a, const glm::vec3 &b, const glm::vec4 &color,
						   std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	pos.push_back(glm::vec4(a, 1.0f));
	pos.push_back(glm::vec4(b, 1.0f));
	colorOut.push_back(color);
	colorOut.push_back(color);
}

void LightDebug::PushCross(const glm::vec3 &center, float halfSize, const glm::vec4 &color,
							std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	PushLine(center - glm::vec3(halfSize, 0, 0), center + glm::vec3(halfSize, 0, 0), color, pos, colorOut);
	PushLine(center - glm::vec3(0, halfSize, 0), center + glm::vec3(0, halfSize, 0), color, pos, colorOut);
	PushLine(center - glm::vec3(0, 0, halfSize), center + glm::vec3(0, 0, halfSize), color, pos, colorOut);
}

void LightDebug::PushBox(const glm::vec3 &center, float halfExtent, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut) {
	// Corner index bits: 4*xBit + 2*yBit + zBit, xBit/yBit/zBit each 0 (-1) or
	// 1 (+1) -- matches the nested loop's emission order below.
	glm::vec3 c[8];
	int idx = 0;
	for(int sx = -1; sx <= 1; sx += 2)
		for(int sy = -1; sy <= 1; sy += 2)
			for(int sz = -1; sz <= 1; sz += 2)
				c[idx++] = center + glm::vec3((float)sx, (float)sy, (float)sz) * halfExtent;

	// The 12 edges of a cube: two corners connected by an edge differ in
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

#endif
