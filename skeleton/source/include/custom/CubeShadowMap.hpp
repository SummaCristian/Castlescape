// ***** CUSTOM *****
//
// Data + math for one point light's real 6-face cube shadow map. The Vulkan
// object creation itself (createImage/createImageView/vkCreateFramebuffer)
// lives in main.cpp, as private methods of the game's BaseProject subclass:
// those calls need BaseProject's PROTECTED helpers (createImage and
// friends), which -- like everywhere else in this codebase -- only that
// subclass's own member functions may reach. Starter.hpp is off limits (the
// professor grades against his own copy), so this struct is deliberately
// just a bag of handles, not a self-contained Texture-like class the way
// Starter.hpp's own Texture/FrameBufferAttachment are.
//
// Storage choice: each face is a single-channel VK_FORMAT_R32_SFLOAT color
// attachment holding the LINEAR distance from the light to the fragment,
// not a depth-only map. A depth-only cube (six independent perspective
// projections) needs a different bias per face/angle to stay artifact-free,
// exactly the problem that forced the old two-map-per-torch workaround this
// replaces to hand-tune two field of views. Linear distance is the same
// unit in every direction, so shadowFactor() in CookTorrance.frag can use
// one flat world-space bias for every torch, every face, every angle.
//
// See CubeShadowMap.hpp's sibling in main.cpp (createCubeShadowMaps()) for
// how these fields get filled in, and computeShadowMatrices() for how
// CUBE_FACE_DIR/CUBE_FACE_UP turn into the six view-projection matrices.

#ifndef CUBE_SHADOW_MAP_HPP
#define CUBE_SHADOW_MAP_HPP

// Deliberately no #include "modules/Starter.hpp" here: that header has no
// include guard of its own (the project's convention is that main.cpp
// includes it exactly once and every custom/*.hpp after it just assumes its
// types already exist -- see SceneLights.hpp's header comment). Including it
// again here would double-define everything in it for any translation unit
// that pulls this header in after Starter.hpp is already visible.

// One real cube shadow map: a 6-layer VkImage (color, R32_SFLOAT,
// VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) plus the views and framebuffers
// needed to render into it face by face and sample it as a whole afterwards.
struct CubeShadowMap {
	// The 6-layer image itself, and the ONE view over all of it
	// (VK_IMAGE_VIEW_TYPE_CUBE) that CookTorrance.frag samples with
	// samplerCube.
	VkImage colorImage = VK_NULL_HANDLE;
	VkDeviceMemory colorMemory = VK_NULL_HANDLE;
	VkImageView cubeView = VK_NULL_HANDLE;

	// One VK_IMAGE_VIEW_TYPE_2D view per layer -- these are what a
	// VkFramebuffer can attach to, a full VK_IMAGE_VIEW_TYPE_CUBE view
	// cannot be a render target.
	VkImageView faceViews[6] = {};

	// Depth, for the rasterizer's z-test only: never sampled afterwards
	// (storeOp DONT_CARE), so ONE 2D image is reused across all 6 faces of
	// THIS torch -- safe because the 6 passes share one VkRenderPass and
	// dependency chain and run in program order within one command buffer,
	// each one clearing it before it's read again.
	VkImage depthImage = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;
	VkImageView depthView = VK_NULL_HANDLE;

	// One framebuffer per face (colorFaceView + depthView), all created
	// against the same shared VkRenderPass (RPShadowCubeCompat in main.cpp).
	VkFramebuffer faceFramebuffers[6] = {};
};

// The six cube-map face directions in Vulkan's layer order (+X,-X,+Y,-Y,+Z,-Z)
// and the up vector each glm::lookAt needs to line up with that convention --
// the same table used by Sascha Willems' shadowmappingomni Vulkan sample.
// Pure math, no BaseProject access needed, hence living here rather than in
// main.cpp.
static const glm::vec3 CUBE_FACE_DIR[6] = {
	glm::vec3( 1.0f,  0.0f,  0.0f),
	glm::vec3(-1.0f,  0.0f,  0.0f),
	glm::vec3( 0.0f,  1.0f,  0.0f),
	glm::vec3( 0.0f, -1.0f,  0.0f),
	glm::vec3( 0.0f,  0.0f,  1.0f),
	glm::vec3( 0.0f,  0.0f, -1.0f),
};

static const glm::vec3 CUBE_FACE_UP[6] = {
	glm::vec3(0.0f, -1.0f,  0.0f),
	glm::vec3(0.0f, -1.0f,  0.0f),
	glm::vec3(0.0f,  0.0f,  1.0f),
	glm::vec3(0.0f,  0.0f, -1.0f),
	glm::vec3(0.0f, -1.0f,  0.0f),
	glm::vec3(0.0f, -1.0f,  0.0f),
};

#endif
