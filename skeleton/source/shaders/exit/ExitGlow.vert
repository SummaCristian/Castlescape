// VERTEX SHADER for the daylight behind the exit door (custom/ExitGlow.hpp).
//
// Trivial: this quad is NOT a billboard but a fixed plane outside the doorway,
// so main.cpp bakes its orientation into mvpMat directly. The whole look lives
// in ExitGlow.frag. No global uniform: nothing here is lit, and `time` rides
// in the block below.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform ExitGlowUniformBufferObject {
	mat4  mvpMat;    // the quad's world basis * ViewPrj, built CPU-side
	vec3  color;     // daylight hue
	float intensity; // peak radiance at the centre, well above 1 (HDR target)
	float time;      // seconds, for the slow breathing in the fragment shader
	float softness;  // where the radial falloff starts, in quad coordinates
} ubo;

// x and y run -1..1 across the quad. A parameter, not a position: mvpMat
// places it (ExitGlow.hpp's createMesh()).
layout(location = 0) in vec2 corner;

layout(location = 0) out vec2 fragCorner;

void main() {
	gl_Position = ubo.mvpMat * vec4(corner, 0.0, 1.0);
	// Straight through: interpolating the corner parameter IS the distance
	// from the quad centre the fragment shader wants.
	fragCorner = corner;
}
