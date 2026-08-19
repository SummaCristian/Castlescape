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

// How many SHADOW MAPS exist, which is no longer the same as how many lights
// cast a shadow: a point light needs two (see below), so this is the sun (1)
// plus two per torch (6 x 2). Sized as an array bound (shadow map samplers,
// light-space matrices), so it has to be a compile-time constant like
// MAX_LIGHTS.
//
// Adding a slot means four edits, all of them in step: this number, one more
// sampler binding plus a sampleShadowMap() case in CookTorrance.frag, one more
// DSLshadowSample binding plus RPShadow entry in main.cpp, and one more aim
// direction in TORCH_SHADOW_DIR. SceneLights::init hands out slots in
// lights.json order and warns if a "castsShadow" light finds none left.
//
// 13 is also close to a hard ceiling: these are 13 of the fragment stage's
// sampled images and the albedo map is a 14th, against a Vulkan guaranteed
// minimum (maxPerStageDescriptorSampledImages) of 16. Real desktop drivers
// allow far more, but past ~16 the code stops being portable by spec.
#define NUM_SHADOW_LIGHTS 13

// Maps per point light: one aimed along its shadow direction, one aimed the
// exact opposite way. A perspective map covers at most a hemisphere, and a
// torch radiates in every direction, so a single map left everything behind
// it either black (if unprojectable points are called shadowed) or lit
// straight through the wall (if they are called lit). Two back-to-back maps
// put every point in front of exactly one of them.
#define SHADOW_MAPS_PER_LIGHT 2

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

#endif
