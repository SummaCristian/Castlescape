// ***** CUSTOM *****
//
// Cube shadow map for one point light: the handles for a 6-face map, plus the
// face direction tables. The Vulkan objects behind these handles are created in
// main.cpp (createCubeShadowMaps()), because that needs BaseProject's protected
// helpers; this header is just the data layout.
//
// Each face stores the LINEAR distance from the light to the fragment in an
// R32_SFLOAT color attachment, not a depth buffer. Depth values are non-linear
// and differ per face, so they would need a different bias per face; a distance
// in world units is the same everywhere, so CookTorrance.frag can shadow-test
// every torch with one constant bias.

#ifndef CUBE_SHADOW_MAP_HPP
#define CUBE_SHADOW_MAP_HPP

// No #include of modules/Starter.hpp here on purpose: it has no include guard,
// so including it twice redefines everything. Convention in this project is
// that main.cpp includes it first and every custom/*.hpp assumes its types.

struct CubeShadowMap {
	// The 6-layer image, and the single cube view CookTorrance.frag samples
	// with samplerCube.
	VkImage colorImage = VK_NULL_HANDLE;
	VkDeviceMemory colorMemory = VK_NULL_HANDLE;
	VkImageView cubeView = VK_NULL_HANDLE;

	// One 2D view per layer: a framebuffer can only attach to these, not to a
	// cube view.
	VkImageView faceViews[6] = {};

	// Depth for the z-test only, never sampled (storeOp DONT_CARE). One image
	// is reused by all 6 faces: they run in order in the same command buffer
	// and each pass clears it first.
	VkImage depthImage = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;
	VkImageView depthView = VK_NULL_HANDLE;

	// One framebuffer per face, all built on the same render pass
	// (RPShadowCubeCompat in main.cpp).
	VkFramebuffer faceFramebuffers[6] = {};
};

// Face directions in Vulkan's cube layer order (+X,-X,+Y,-Y,+Z,-Z) and the up
// vector glm::lookAt needs for each to match that convention. Used by
// computeShadowMatrices() in main.cpp to build the six view-projections.
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
