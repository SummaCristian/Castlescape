// ***** CUSTOM *****
// Flat-colored-quad renderer: background/highlight layer under DebugHud's text.
// No textures/descriptor sets, just a push-constant color per quad.
// Header-only, implementation gated behind UIQUAD_IMPLEMENTATION (Libs.cpp).

#include <string>
#include <vector>

struct UiQuadVertex {
	glm::vec2 pos;
};

// One flat-colored rectangle, in pixel space (top-left origin, matching
// GLFW's cursor coordinates and DebugHud's own layout math).
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
	// Named command-buffer slot; instances beyond the HUD's need a distinct name.
	std::string bufferName = "ui_quad";

	// Draws before TextMaker's default order (10000), so text composites on top.
	void init(BaseProject *_BP, int sW, int sH, int so = 9000,
			  std::string bufName = "ui_quad");
	void resizeScreen(int sW, int sH);
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

void UiQuad::init(BaseProject *_BP, int sW, int sH, int so, std::string bufName) {
	BP = _BP;
	screenW = sW;
	screenH = sH;
	submitOrder = so;
	bufferName = bufName;

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
	// No descriptor set layouts: empty vector is safe (Pipeline::create builds pSetLayouts from D.size()).
	P.init(BP, &VD, "shaders/ui/UiQuad.vert.spv", "shaders/ui/UiQuad.frag.spv", {},
		{{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(UiQuadColorPushConstant)}});
	P.setCullMode(VK_CULL_MODE_NONE);
	P.setTransparency(true);
	// LESS_OR_EQUAL: background+highlight overlap at same z, lets same-depth overdraw through
	// (same fix TextMaker uses for stacked glyph quads).
	P.setCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);
}

void UiQuad::setQuads(std::vector<UiRect> newRects) {
	rects = std::move(newRects);
	commandBufferMustUpdate = true;
}

void UiQuad::createMesh() {
	M = new Model();

	// initMesh needs a non-zero GPU buffer; with nothing to draw, use an inert
	// placeholder quad instead (populateCommandBuffer skips it since rects is empty).
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
		BP->submitCommandBuffer(bufferName.c_str(), submitOrder,
							UiQuad::populateCommandBufferAccess, qm,
							UiQuad::freeCommandBuffer);
		commandBufferMustUpdate = false;
	}
}

#endif
