// VERTEX SHADER for the flame's glow billboard (see custom/Flame.hpp).
// There's no bloom pass in this renderer -- CookTorrance.frag tone-maps and
// the swapchain is an ordinary 8-bit sRGB target, so nothing downstream can
// blur an overbright pixel into its neighbours. This billboard fakes that
// bleed directly: a soft, camera-facing, alpha-blended quad centered on the
// flame, so nearby geometry looks lit by a hazy halo instead of the flame
// just being a hard-edged bright shape.
//
// The quad itself is a fixed [-1,1] square (see FlameGlow.frag for the radial
// falloff that keeps it from ever reading as "a square"); its world position,
// facing and size are entirely in mvpMat, built fresh every frame by the
// caller from the camera's own right/up vectors (a billboard has to face the
// camera, not ride the flame's own orientation the way the flame body does).

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 1) uniform FlameGlowUniformBufferObject {
	mat4 mvpMat;
	float seed;
} fubo;

layout(location = 0) in vec2 inPos;

layout(location = 0) out vec2 uv;

void main() {
	gl_Position = fubo.mvpMat * vec4(inPos, 0.0, 1.0);
	uv = inPos;
}
