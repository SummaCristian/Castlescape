// Debug line overlays (gizmos, frustums, colliders; cheat-menu gated, DebugLines.hpp).
// Vertex pulling, not a vertex buffer: endpoints live in UBO arrays,
// looked up by gl_VertexIndex. Arrays remap per frame; vertex count is fixed.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform DebugLinesVP {
	mat4 vpMat;
} viewProj;

// vec4 not vec3: std140 pads vec3 arrays to 16 bytes anyway; w is unused.
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
