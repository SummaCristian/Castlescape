// ***** CUSTOM *****
// Cube shadow map handles for one point light + face direction tables.
// Vulkan objects created in main.cpp (createCubeShadowMaps()).
// Faces store LINEAR distance (R32_SFLOAT), not depth: same bias works for every face.

#ifndef CUBE_SHADOW_MAP_HPP
#define CUBE_SHADOW_MAP_HPP

// No #include of modules/Starter.hpp: no include guard there, main.cpp includes it first.

struct CubeShadowMap {
	// 6-layer image + single cube view sampled with samplerCube.
	VkImage colorImage = VK_NULL_HANDLE;
	VkDeviceMemory colorMemory = VK_NULL_HANDLE;
	VkImageView cubeView = VK_NULL_HANDLE;

	// One 2D view per layer: framebuffers can't attach to a cube view.
	VkImageView faceViews[6] = {};

	// Z-test only, never sampled (storeOp DONT_CARE). Shared by all 6 faces, cleared each pass.
	VkImage depthImage = VK_NULL_HANDLE;
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;
	VkImageView depthView = VK_NULL_HANDLE;

	// One framebuffer per face (RPShadowCubeCompat in main.cpp).
	VkFramebuffer faceFramebuffers[6] = {};
};

// Vulkan cube layer order (+X,-X,+Y,-Y,+Z,-Z) + up vector per face for glm::lookAt.
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
