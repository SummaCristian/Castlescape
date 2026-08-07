// ***** CUSTOM *****

// A tiny generic flat-colored-quad renderer, used as the background/highlight
// layer under CheatHud's text. TextMaker can only draw glyphs from its font
// atlas, and there's no filled rectangle in its printable ASCII range to fake
// a backdrop with. So a real panel background needs its own (minimal)
// pipeline: no textures/descriptor sets, just a push-constant color per quad.
//
// Same header-only "module" pattern as TextMaker/Scene/Animations/CheatHud:
// declarations + implementation in this one file, implementation gated
// behind UIQUAD_IMPLEMENTATION (defined once in Libs.cpp). Like those
// modules, this file assumes "modules/Starter.hpp" is already included by
// whoever includes this one.

#include <vector>

struct UiQuadVertex {
	glm::vec2 pos;
};

// One flat-colored rectangle, in pixel space (top-left origin, matching
// GLFW's cursor coordinates and CheatHud's own layout math).
struct UiRect {
	float x, y, w, h;
	glm::vec4 color;
};

struct UiQuad;

struct UiQuadAndModel {
	UiQuad *quads;
	Model *M;
};

struct UiQuadColorPushConstant {
	alignas(16) glm::vec4 color;
};

struct UiQuad {
	VertexDescriptor VD;

	BaseProject *BP;
	int screenW, screenH;
	int submitOrder;

	RenderPass RP;
	Pipeline P;
	Model *M = nullptr;

	std::vector<UiRect> rects = {};
	bool commandBufferMustUpdate = false;

	// Draws before TextMaker's default order (10000), so text composites on
	// top of these quads rather than the other way around.
	void init(BaseProject *_BP, int sW, int sH, int so = 9000);
	void resizeScreen(int sW, int sH);
	// Replaces the whole quad list and marks the mesh for a rebuild.
	// Unconditional, same as TextMaker::print: it's up to the caller (e.g.
	// CheatHud's own dirty flag) to only call this when something changed.
	void setQuads(std::vector<UiRect> newRects);
	void createPipeline();
	void createMesh();
	void pipelinesAndDescriptorSetsInit();
	void pipelinesAndDescriptorSetsCleanup();
	void localCleanup();
	static void populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params);
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage);
	static void freeCommandBuffer(void *Params);
	void updateCommandBuffer();

	private:
	void pixelToScr(float x, float y, float &sx, float &sy);
};

#ifdef UIQUAD_IMPLEMENTATION

void UiQuad::pixelToScr(float x, float y, float &sx, float &sy) {
	sx = (x / (float)screenW) * 2.0f - 1.0f;
	sy = (y / (float)screenH) * 2.0f - 1.0f;
}

void UiQuad::init(BaseProject *_BP, int sW, int sH, int so) {
	BP = _BP;
	screenW = sW;
	screenH = sH;
	submitOrder = so;

	VD.init(BP, {
			  {0, sizeof(UiQuadVertex), VK_VERTEX_INPUT_RATE_VERTEX}
			}, {
			  {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(UiQuadVertex, pos),
					 sizeof(glm::vec2), OTHER}
			});

	createPipeline();

	RP.init(BP, sW, sH, -1,
				RenderPass::getStandardAttchmentsProperties(AT_SURFACE_NOAA_DEPTH, BP));
	RP.properties[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	RP.properties[0].initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	RP.properties[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	RP.properties[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
}

void UiQuad::resizeScreen(int sW, int sH) {
	screenW = sW;
	screenH = sH;
	RP.width = sW;
	RP.height = sH;
	commandBufferMustUpdate = true;
}

void UiQuad::createPipeline() {
	// No descriptor set layouts: this pipeline uses no textures/uniforms,
	// only a push-constant color per quad. Pipeline::create builds
	// pSetLayouts from D.size(), so an empty vector here is safe (0 layouts).
	P.init(BP, &VD, "shaders/framework/UiQuad.vert.spv", "shaders/framework/UiQuad.frag.spv", {},
		{{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(UiQuadColorPushConstant)}});
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// The panel background and selection highlight both sit at z=0 and
	// overlap on screen. Pipeline::create() hardcodes depth test+write on
	// for every pipeline, so with the default LESS the second (highlight)
	// quad would fail against the depth the first (background) quad just
	// wrote at the same z. LESS_OR_EQUAL is the same fix TextMaker uses for
	// stacked/overlapping same-z glyph quads: it lets same-depth overdraw
	// through.
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
}

void UiQuad::setQuads(std::vector<UiRect> newRects) {
	rects = std::move(newRects);
	commandBufferMustUpdate = true;
}

void UiQuad::createMesh() {
	M = new Model();

	// Model::initMesh always creates a real, non-zero-size GPU buffer, and
	// this pass's command buffer has to stay valid for the rest of the app's
	// lifetime once first submitted. Starter.hpp's submitCommandBuffer and
	// updateCommandBuffers dereference the "current" named command buffer
	// unconditionally, and there's no supported way to detach one and safely
	// resubmit under the same name later. So when there's nothing to draw
	// (HUD hidden), build one inert placeholder quad instead of a zero-sized
	// mesh: populateCommandBuffer only issues draw calls for entries in
	// "rects" (which stays empty), so the placeholder's geometry exists to
	// keep the buffers valid but is never actually drawn.
	const std::vector<UiRect> &meshRects = rects.empty()
		? std::vector<UiRect>{{0.0f, 0.0f, 0.0f, 0.0f, glm::vec4(0.0f)}}
		: rects;

	int mainStride = VD.Bindings[0].stride;
	M->indices.resize(6 * meshRects.size());
	M->vertices.resize(4 * meshRects.size() * mainStride);

	UiQuadVertex *V = (UiQuadVertex *)(&M->vertices[0]);
	for(int i = 0; i < (int)meshRects.size(); i++) {
		const UiRect &R = meshRects[i];
		float sx, sy;

		pixelToScr(R.x, R.y, sx, sy);
		V[4 * i + 0].pos = {sx, sy};
		pixelToScr(R.x + R.w, R.y, sx, sy);
		V[4 * i + 1].pos = {sx, sy};
		pixelToScr(R.x, R.y + R.h, sx, sy);
		V[4 * i + 2].pos = {sx, sy};
		pixelToScr(R.x + R.w, R.y + R.h, sx, sy);
		V[4 * i + 3].pos = {sx, sy};

		M->indices[6 * i + 0] = 4 * i + 0;
		M->indices[6 * i + 1] = 4 * i + 1;
		M->indices[6 * i + 2] = 4 * i + 2;
		M->indices[6 * i + 3] = 4 * i + 1;
		M->indices[6 * i + 4] = 4 * i + 2;
		M->indices[6 * i + 5] = 4 * i + 3;
	}
	M->initMesh(BP, &VD, false);
}

void UiQuad::pipelinesAndDescriptorSetsInit() {
	RP.create();
	P.create(&RP);
}

void UiQuad::pipelinesAndDescriptorSetsCleanup() {
	P.cleanup();
	RP.cleanup();
}

void UiQuad::localCleanup() {
	if(M != nullptr) {
		M->cleanup();
	}
	P.destroy();
	RP.destroy();
}

void UiQuad::populateCommandBufferAccess(VkCommandBuffer commandBuffer, int currentImage, void *Params) {
	UiQuad *Q = ((UiQuadAndModel *)Params)->quads;
	Q->populateCommandBuffer(commandBuffer, currentImage);
}

void UiQuad::populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
	RP.begin(commandBuffer, currentImage);

	if(!rects.empty()) {
		P.bind(commandBuffer);
		M->bind(commandBuffer);

		for(int i = 0; i < (int)rects.size(); i++) {
			UiQuadColorPushConstant PKv;
			PKv.color = rects[i].color;
			vkCmdPushConstants(
				commandBuffer,
				P.pipelineLayout,
				VK_SHADER_STAGE_FRAGMENT_BIT,
				0,
				sizeof(PKv),
				&PKv);

			vkCmdDrawIndexed(commandBuffer, 6, 1, 6 * i, 0, 0);
		}
	}

	RP.end(commandBuffer);
}

void UiQuad::freeCommandBuffer(void *Params) {
	Model *M = ((UiQuadAndModel *)Params)->M;
	M->cleanup();

	free(Params);
}

void UiQuad::updateCommandBuffer() {
	if(commandBufferMustUpdate) {
		createMesh();

		UiQuadAndModel *qm = (UiQuadAndModel *)malloc(sizeof(UiQuadAndModel));
		qm->quads = this;
		qm->M = M;
		BP->submitCommandBuffer("ui_quad", submitOrder,
							UiQuad::populateCommandBufferAccess, qm,
							UiQuad::freeCommandBuffer);
		commandBufferMustUpdate = false;
	}
}

#endif
