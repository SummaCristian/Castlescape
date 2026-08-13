// ***** CUSTOM *****

// The one definition of the light constants, shared by the C++ side and the
// shaders. `SceneLights.hpp` and `Blinn.frag` both #include this exact path.
//
// They used to be written twice, once per language, with a comment asking the
// reader to keep them in sync. A comment is not a mechanism: change MAX_LIGHTS
// on one side only and nothing complains, the C++ struct and the GLSL block
// quietly stop describing the same bytes, and what you get is not an error but
// wrong lighting.
//
// This file therefore contains nothing but preprocessor directives, because
// that is the only syntax the C++ and GLSL preprocessors agree on. No types, no
// declarations, no `const int`: those would compile in one language and not the
// other. Note that `#define` is also what GLSL needs for MAX_LIGHTS anyway,
// since it sizes an array and so has to be a compile-time constant.
//
// Two things make the shared include work:
//   - CMake passes `-I source/include` to glslc, so the include path below is
//     spelled identically in both languages;
//   - the shader enables GL_GOOGLE_include_directive, glslc's #include support,
//     which is not part of core GLSL.

#ifndef LIGHT_CONSTANTS_GLSL
#define LIGHT_CONSTANTS_GLSL

// How many lights the uniform block has room for. A uniform block needs a size
// known when the pipeline is built, so the array is fixed and a separate
// lightCount says how much of it is real.
#define MAX_LIGHTS 16

// Light type tags. The shader switches on these, the loader writes them.
#define LIGHT_DIRECT 0
#define LIGHT_POINT  1
#define LIGHT_SPOT   2

#endif
