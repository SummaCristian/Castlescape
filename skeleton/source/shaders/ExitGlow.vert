// VERTEX SHADER for the daylight behind the exit door (see custom/ExitGlow.hpp).
//
// Deliberately trivial: this quad is NOT a billboard. It is a fixed plane
// standing on the open ground just outside the doorway, so unlike the flame it
// has a real orientation in the world and main.cpp bakes that into mvpMat
// directly. The whole look lives in ExitGlow.frag.
//
// set 0 is this effect's own small block. Unlike Flame.vert there is no global
// uniform bound at all: nothing here is lit, so eyePos and the light array
// would go unread, and `time` (the only thing it would have wanted) rides in
// the block below instead.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform ExitGlowUniformBufferObject {
	mat4  mvpMat;    // the quad's world basis * ViewPrj, built CPU-side
	vec3  color;     // daylight hue
	float intensity; // peak radiance at the centre, well above 1 (HDR target)
	float time;      // seconds, for the slow breathing in the fragment shader
	float softness;  // where the radial falloff starts, in quad coordinates
} ubo;

// x and y both run -1..1 across the quad. A pure parameter, not a position:
// mvpMat is what places it (see ExitGlow.hpp's createMesh()).
layout(location = 0) in vec2 corner;

layout(location = 0) out vec2 fragCorner;

void main() {
	gl_Position = ubo.mvpMat * vec4(corner, 0.0, 1.0);
	// Passed straight through, un-normalized: the fragment shader wants the
	// distance from the centre of the QUAD, and interpolating the corner
	// parameter is exactly that.
	fragCorner = corner;
}
