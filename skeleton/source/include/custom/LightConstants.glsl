// ***** CUSTOM *****

// Light constants, included by both SceneLights.hpp and CookTorrance.frag so
// there is one definition instead of two kept in step by hand.
//
// Preprocessor directives only: it is the one syntax C++ and GLSL agree on, and
// MAX_LIGHTS has to be a #define anyway since it sizes an array. Works because
// CMake passes glslc the same -I as the C++ compiler, and the shader enables
// GL_GOOGLE_include_directive (core GLSL has no #include).

#ifndef LIGHT_CONSTANTS_GLSL
#define LIGHT_CONSTANTS_GLSL

// Uniform blocks need a compile-time size, hence a fixed array plus a live
// lightCount saying how much of it is real.
#define MAX_LIGHTS 32

// Slots for point-light cube shadow maps (CubeShadowMap.hpp), each a real
// 6-face cube. The LAST slot is fixed: main.cpp reserves it for the held
// torch, which moves with the camera and never goes through lights.json
// (HAND_TORCH_SHADOW_INDEX). The rest are handed out at RUNTIME, not at load
// time, by updateDynamicShadowSlots(): a level can author more shadow-worthy
// torches and candles than there are slots, so the ones nearest the player
// win, re-decided as the player moves (see that function for why it needs
// hysteresis rather than a plain "N nearest" rule). No point light gets a
// permanent slot for being implemented first; lights.json's "castsShadow" is
// not read for point lights at all.
//
// Growing the pool means: this number, one more samplerCube binding plus a
// sampleShadowCube4() case in CookTorrance.frag, and one more CubeShadowMap in
// main.cpp. Nothing else needs touching.
//
// These share the fragment stage's sampled-image budget with
// the albedo map, and 32 + 1 is well past the 16
// that Vulkan GUARANTEES (maxPerStageDescriptorSampledImages). Deliberate:
// real desktop GPUs allow far more, and the headroom means a level with many
// torches doesn't have every extra one instantly lose the shadow contest. On a
// device that offered less, the dynamic pool degrades gracefully anyway --
// SHADOW_SWAP_MARGIN (main.cpp) just has more candidates to arbitrate.
#define NUM_SHADOW_CUBES 32

// Side, in texels, of one face of a cube shadow map. Shared with main.cpp's
// SHADOW_MAP_RES (which is what actually sizes the images) because
// CookTorrance.frag needs it too: the depth bias for the cube path is derived
// from how much WORLD space one texel of that map covers at the distance being
// shaded, and that derivation is meaningless without the resolution. See
// shadowFromCube() there.
#define SHADOW_CUBE_RES 1024

#define LIGHT_DIRECT 0
#define LIGHT_POINT  1
#define LIGHT_SPOT   2

// Debug views of the light model, driven by the cheat menu (CheatFlags in
// main.cpp) and packed into one int of the global uniform, gubo.debugFlags.
// A bitmask rather than one field each: they are independent switches, they
// combine, and the alternative is four ints eating four uniform slots.
//
// Which lights exist at all isn't in here: turning off the sun or the lanterns
// happens on the CPU side (SceneLights), which simply doesn't upload them.
#define LIGHT_DEBUG_UNLIT       1	// albedo straight out, no lighting at all
#define LIGHT_DEBUG_NORMALS     2	// the shading normal as a color
#define LIGHT_DEBUG_NO_SPECULAR 4	// diffuse only, the BRDF's k forced to 1
#define LIGHT_DEBUG_NO_TONEMAP  8	// skip the tone map, so overexposure clips
#define LIGHT_DEBUG_NO_SHADOWS  16	// every shadowFactor() forced to 1 (fully lit)
#define LIGHT_DEBUG_HEATMAP     32	// recolor by incoming light intensity, ignoring albedo

// Drop the whole indirect term: ambientShare() returns 0, so the direct lights
// get the entire frame back. The "Ambient Light" cheat.
//
// A flag rather than just zeroing gubo.ambientWeight, which is how this used to
// work and why the cheat silently did nothing on some models: ambientShare()
// prefers the MATERIAL's ambientWeight whenever materials.json sets one (the
// floor does, at 0.20), so zeroing the global left every overriding model at
// full indirect light. The share can be authored in two places; the switch that
// turns it off has to sit downstream of both, and that is here.
#define LIGHT_DEBUG_NO_AMBIENT  128

// Torch/candle bounce is the only indirect light in the scene (no sun, no
// sky -- see CookTorrance.frag's ambient composition), for both diffuse
// surfaces and metals, so one flag now covers all of it. Independent of
// LIGHT_DEBUG_NO_AMBIENT (which drops the whole share via ambientShare()).
#define LIGHT_DEBUG_NO_BOUNCE     512	// torch/candle indirect bounce forced to 0

#endif
