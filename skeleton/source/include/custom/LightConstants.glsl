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
#define MAX_LIGHTS 16

// How many 2D (depth-only) shadow maps exist: one per DIRECT or SPOT
// shadow-casting light. A point light does NOT take a slot here -- it gets a
// genuine cube shadow map instead (NUM_SHADOW_CUBES below), because a single
// perspective map covers at most a hemisphere and a torch radiates in every
// direction. Today only the sun uses this array; the +1 over that is
// headroom for a future shadow-casting spot.
//
// Sized as an array bound (shadow map samplers, light-space matrices), so it
// has to be a compile-time constant like MAX_LIGHTS. SceneLights::init hands
// out slots in lights.json order and warns if a "castsShadow" light finds
// none left.
#define NUM_SHADOW_MAPS_2D 2

// How many point lights can cast a shadow, each through one real 6-face cube
// shadow map (CubeShadowMap.hpp). Exactly one slot is fixed: the last one,
// reserved by main.cpp for the held torch, which never goes through
// lights.json (it moves with the camera -- see the HAND_TORCH_SHADOW_INDEX
// comment there). Every other slot (main.cpp's
// dynamicShadowSlotBase..HAND_TORCH_SHADOW_INDEX, currently 0..10) is handed
// out at RUNTIME instead of at load time: there are more shadow-worthy point
// lights in the scene (the six wall/dv torches, the four colored decorative
// torches, the two candles -- twelve in all) than there are spare slots, so
// main.cpp's updateDynamicShadowSlots() gives them to whichever are
// currently nearest the player, re-deciding as the player moves (see its own
// header for why that needs hysteresis rather than a plain "N nearest"
// rule). No point light gets a PERMANENT slot just for being the first one
// implemented -- lights.json's own "castsShadow" flag isn't used for point
// lights any more, see its comment there.
//
// Adding a slot means: this number, one more samplerCube binding plus a
// sampleShadowCube() case in CookTorrance.frag, and one more CubeShadowMap
// instance in main.cpp -- the dynamic pool just gets wider, nothing else
// needs touching.
//
// These share the fragment stage's sampled-image budget with
// NUM_SHADOW_MAPS_2D and the albedo map. 20 + 2 + 1 = 23 exceeds the Vulkan
// GUARANTEED minimum (maxPerStageDescriptorSampledImages) of 16, so this
// isn't portable to a spec-minimum GPU any more -- deliberately: flames.json
// (main.cpp) now spawns a flame for every torch/candle a LEVEL authors
// (keyed by model, not hardcoded per instance), so the scene's point-light
// count isn't a fixed 13 any more either, and this leaves headroom for
// levels with more torches than today's one without every extra one
// instantly losing the shadow contest. Real desktop GPUs allow far more
// than 16 here in practice. If this ever needs to grow past what a target
// GPU actually offers, the dynamic pool degrades gracefully either way --
// SHADOW_SWAP_MARGIN (main.cpp) just has more candidates to arbitrate.
#define NUM_SHADOW_CUBES 20

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

#endif
