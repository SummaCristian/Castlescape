// ***** CUSTOM *****
// Cube shadow map handles for one point light + face direction tables.
// Vulkan objects created in main.cpp (createCubeShadowMaps()).
// Faces store LINEAR distance (R32_SFLOAT), not depth: same bias works for every face.

#ifndef CUBE_SHADOW_MAP_HPP
#define CUBE_SHADOW_MAP_HPP

// No #include of modules/Starter.hpp: no include guard there, main.cpp includes it first.

struct CubeShadowMap {
	// 6-layer image + single cube view sampled with samplerCube.
	VkImage colorImage = VK_NULL_HANDLE;         // the 6-layer distance image itself
	VkDeviceMemory colorMemory = VK_NULL_HANDLE; // GPU memory backing colorImage
	VkImageView cubeView = VK_NULL_HANDLE;       // whole-cube view (single element), bound as samplerCube in CookTorrance.frag

	// One 2D view per layer: framebuffers can't attach to a cube view.
	VkImageView faceViews[6] = {}; // one flat 2D view per face, what ShadowCube.frag draws into

	// Z-test only, never sampled (storeOp DONT_CARE). Shared by all 6 faces, cleared each pass.
	VkImage depthImage = VK_NULL_HANDLE;          // scratch depth buffer, one shared by all 6 faces
	VkDeviceMemory depthMemory = VK_NULL_HANDLE;  // GPU memory backing depthImage
	VkImageView depthView = VK_NULL_HANDLE;       // view used only for the depth attachment

	// One framebuffer per face (RPShadowCubeCompat in main.cpp).
	VkFramebuffer faceFramebuffers[6] = {}; // draw target for each of the 6 ShadowCube passes
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
