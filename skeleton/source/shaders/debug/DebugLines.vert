// VERTEX SHADER for the debug line overlays (light gizmos, shadow frustums,
// collider wireframes -- cheat-menu gated, see DebugLines.hpp).
//
// VERTEX PULLING, not a vertex buffer: a custom class can't reach
// BaseProject::createBuffer() without editing Starter.hpp (off-limits), so
// every line's endpoints live in a uniform-buffer array and this shader looks
// its own vertex up by gl_VertexIndex. The arrays are re-mapped every frame;
// the draw's vertex count never changes.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform DebugLinesVP {
	mat4 vpMat;
} viewProj;

// vec4, not vec3: std140 gives an array of vec3 a 16-byte stride anyway, so
// declaring vec4 matches the C++ side. w is unused padding.
layout(binding = 1, set = 0) uniform DebugLinesPos {
	vec4 pos[2048];
} linePositions;

layout(binding = 2, set = 0) uniform DebugLinesColor {
	vec4 color[2048];
} lineColors;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_Position = viewProj.vpMat * vec4(linePositions.pos[gl_VertexIndex].xyz, 1.0);
	fragColor = lineColors.color[gl_VertexIndex].rgb;
}
