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

#endif
