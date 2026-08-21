// VERTEX SHADER for the light/shadow debug line overlay (gizmos + shadow
// frustum wireframes, cheat-menu gated -- see LightDebug.hpp).
//
// VERTEX PULLING, not a vertex buffer: main.cpp's custom classes can only
// reach BaseProject::createBuffer() through the fixed friend list Starter.hpp
// declares (Model, DescriptorSet, ...), which a new class outside that list
// can't join without editing Starter.hpp -- off-limits (see LightDebug.hpp's
// header). So instead of a vertex buffer, every line's endpoints live in a
// uniform buffer as a plain array, and this shader looks its own vertex up
// by gl_VertexIndex. Positions and colors are re-mapped fresh every frame the
// same way DSshadowCube[]/Flame's per-instance UBO already are; the draw
// call's vertex count itself never changes (see the .hpp), only what's
// inside these arrays.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform LightDebugVP {
	mat4 vpMat;
} vp;

// vec4 rather than vec3: std140 packs an array of vec3 as if each were a
// vec4 anyway (16-byte stride), so declaring it vec4 up front avoids a
// mismatch between this and the C++ side's std140-equivalent layout. w is
// unused padding in both arrays.
layout(binding = 1, set = 0) uniform LightDebugPos {
	vec4 pos[512];
} P;

layout(binding = 2, set = 0) uniform LightDebugColor {
	vec4 color[512];
} C;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_Position = vp.vpMat * vec4(P.pos[gl_VertexIndex].xyz, 1.0);
	fragColor = C.color[gl_VertexIndex].rgb;
}
