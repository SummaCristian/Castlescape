// Daylight quad behind the exit door (custom/ExitGlow.hpp).
// Fixed plane, not a billboard: orientation baked into mvpMat by main.cpp.
// Look lives in ExitGlow.frag; no global uniform, nothing here is lit.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform ExitGlowUniformBufferObject {
	mat4  mvpMat;    // world basis * ViewPrj, built CPU-side
	vec3  color;     // daylight hue
	float intensity; // peak radiance, HDR (can exceed 1)
	float time;      // seconds, drives breathing in fragment shader
	float softness;  // radial falloff start, in quad coords
} ubo;

// -1..1 across the quad; mvpMat places it (ExitGlow.hpp's createMesh()).
layout(location = 0) in vec2 corner;

layout(location = 0) out vec2 fragCorner;

void main() {
	gl_Position = ubo.mvpMat * vec4(corner, 0.0, 1.0);
	// Interpolated corner = distance from quad centre, used downstream.
	fragCorner = corner;
}
