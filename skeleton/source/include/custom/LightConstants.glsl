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

#endif
