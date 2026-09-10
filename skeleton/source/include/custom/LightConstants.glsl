// ***** CUSTOM *****

// Light constants shared by SceneLights.hpp and CookTorrance.frag.
// #define only: the one syntax both C++ and GLSL accept.

#ifndef LIGHT_CONSTANTS_GLSL
#define LIGHT_CONSTANTS_GLSL

// Fixed array size for uniform blocks; lightCount says how much is real.
#define MAX_LIGHTS 32

// Cube shadow map slots (CubeShadowMap.hpp). Last slot = held torch
// (HAND_TORCH_SHADOW_INDEX), fixed. Rest assigned at runtime by
// updateDynamicShadowSlots(): nearest-to-player wins, with hysteresis.
// Growing the pool needs: this number, a samplerCube binding + case in
// sampleShadowCube4() (CookTorrance.frag), one more CubeShadowMap in main.cpp.
// 32+1 samplers stays under real GPU limits though above Vulkan's guaranteed
// minimum (16); degrades gracefully via SHADOW_SWAP_MARGIN if not.
#define NUM_SHADOW_CUBES 32

// Cube shadow face resolution, texels/side. Shared with main.cpp's
// SHADOW_MAP_RES: CookTorrance.frag derives the cube depth bias from
// world-space texel size at shading distance (see shadowFromCube()).
#define SHADOW_CUBE_RES 1024

#define LIGHT_DIRECT 0
#define LIGHT_POINT  1
#define LIGHT_SPOT   2

// Debug view bitmask, packed into gubo.debugFlags (debug menu, main.cpp).
// Bitmask since flags are independent and combine.
#define LIGHT_DEBUG_UNLIT       1	// albedo only, no lighting
#define LIGHT_DEBUG_NORMALS     2	// shading normal as color
#define LIGHT_DEBUG_NO_SPECULAR 4	// diffuse only, BRDF k forced to 1
#define LIGHT_DEBUG_NO_TONEMAP  8	// skip tone map, overexposure clips
#define LIGHT_DEBUG_NO_SHADOWS  16	// shadowFactor() forced to 1
#define LIGHT_DEBUG_HEATMAP     32	// recolor by light intensity, ignoring albedo

// Zeroes ambientShare() so direct lights get the full frame back.
// Must sit downstream of both global and per-material ambientWeight
// (materials.json can override the global), so it's a flag, not a zeroed uniform.
#define LIGHT_DEBUG_NO_AMBIENT  128

// Torch/candle bounce is the only indirect light in the scene; one flag
// covers diffuse and metal. Independent of LIGHT_DEBUG_NO_AMBIENT.
#define LIGHT_DEBUG_NO_BOUNCE     512	// torch/candle bounce forced to 0

#endif
