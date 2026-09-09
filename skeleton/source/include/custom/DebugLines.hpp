// ***** CUSTOM *****
// Line renderer for debug overlays: callers build world-space segments with Push*
// helpers, one shared pipeline for all of them (gizmos, frustums, colliders).
// Vertex pulling (UBO array indexed by gl_VertexIndex), fixed draw count, padded to MAX_VERTS.
// Header-only, implementation gated behind DEBUGLINES_IMPLEMENTATION (Libs.cpp).

#include <algorithm>
#include <array>
#include <vector>

struct DebugLinesVPUBO {
	glm::mat4 vpMat;
};

class DebugLines {
	public:
	// MAX_VERTS vec4s = 64KB = guaranteed minimum maxUniformBufferRange.
	static constexpr int MAX_VERTS = 4096;

	void init(BaseProject *_BP);
	void pipelinesAndDescriptorSetsInit(RenderPass *_RP);
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();

	// pos/color: one entry per vertex, in pairs (each pair a segment). Beyond MAX_VERTS truncated.
	void update(int currentImage, const glm::mat4 &vpMat,
				const std::vector<glm::vec4> &pos, const std::vector<glm::vec4> &color);

	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);

	// One line segment (2 verts).
	static void PushLine(const glm::vec3 &a, const glm::vec3 &b, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// 3-axis cross (6 verts) -- light-position gizmo.
	static void PushCross(const glm::vec3 &center, float halfSize, const glm::vec4 &color,
						   std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// 12-edge wireframe cube (24 verts): exact bound of a 90-degree-FOV cube shadow face.
	static void PushBox(const glm::vec3 &center, float halfExtent, const glm::vec4 &color,
						 std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// 12 edges from a min/max corner pair, matching Collider::getExtents().
	static void PushAABB(const glm::vec3 &lo, const glm::vec3 &hi, const glm::vec4 &color,
						  std::vector<glm::vec4> &pos, std::vector<glm::vec4> &colorOut);

	// Closed 4-point loop (8 verts) -- ramps' inclined quads.
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

	// No vertex attributes: gl_VertexIndex alone drives the lookup.
	VD.init(BP, {}, {});

	DSL.init(BP, {
				{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					sizeof(DebugLinesVPUBO), 1},
				{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					(int)(sizeof(glm::vec4) * MAX_VERTS), 1},
				{2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT,
					(int)(sizeof(glm::vec4) * MAX_VERTS), 1}
			  });

	BP->DPSZs.uniformBlocksInPool += 3;
	BP->DPSZs.setsInPool += 1;

	P.init(BP, &VD, "shaders/debug/DebugLines.vert.spv", "shaders/debug/DebugLines.frag.spv", {&DSL});
	P.setTopology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
	P.setCullMode(VK_CULL_MODE_NONE);	// lines have no facing to cull
	// LESS_OR_EQUAL: edge exactly on a surface still wins depth test.
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

	// Padding: unused slots repeat the last real vertex -> zero-length line, draws nothing.
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
	// Corner index bits: 4*xBit + 2*yBit + zBit, 0=lo/1=hi.
	glm::vec3 c[8];
	int idx = 0;
	for(int sx = 0; sx <= 1; sx++)
		for(int sy = 0; sy <= 1; sy++)
			for(int sz = 0; sz <= 1; sz++)
				c[idx++] = glm::vec3(sx ? hi.x : lo.x, sy ? hi.y : lo.y, sz ? hi.z : lo.z);

	// 12 edges: connected corners differ in exactly one bit.
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
