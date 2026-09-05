// FRAGMENT SHADER for the shadow pass. Writes nothing: the render pass this
// runs on (RPShadow in main.cpp, AT_DEPTH_ONLY) has no color attachment at
// all, only depth, and depth gets written automatically by the fixed-function
// pipeline stage from gl_Position.z -- there is no per-pixel value left for a
// fragment shader to compute.
//
// An empty shader stage still has to exist: Pipeline::init (Starter.hpp)
// always links a vertex AND a fragment module, so this file's only job is to
// be a valid, minimal one.

#version 450
#extension GL_ARB_separate_shader_objects : enable

void main() {
}
