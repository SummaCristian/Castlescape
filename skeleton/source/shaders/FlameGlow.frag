// FRAGMENT SHADER for the flame's glow billboard. Radial falloff from the
// quad's center (uv, from FlameGlow.vert) to fully transparent at its edge,
// so the fixed [-1,1] square reads as a soft round haze rather than a card.
// Alpha-blended (SRC_ALPHA/ONE_MINUS_SRC_ALPHA is the only blend mode this
// renderer's Pipeline exposes -- see notes.md on why Starter.hpp isn't
// touched): not true additive bloom, but a bright near-white color faded
// through a wide soft radius reads as one at a glance, especially against
// the dark stone this torch is meant to be carried through.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	vec3 ambientUpper;
	vec3 ambientLower;
	vec3 ambientDir;
	int debugFlags;
	float time;
} gubo;

layout(binding = 0, set = 1) uniform FlameGlowUniformBufferObject {
	mat4 mvpMat;
	float seed;
} fubo;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

void main() {
	float d = length(uv);
	if(d > 1.0) {
		discard;
	}

	// Squared falloff: a wide, gentle haze rather than a visible disc edge.
	float falloff = (1.0 - d) * (1.0 - d);

	// Same phase convention as Flame.frag's own flicker (TAU * seed), so the
	// glow pulses in step with the flame it surrounds instead of drifting.
	float flicker = 0.8 + 0.3 * sin(gubo.time * 9.0 + fubo.seed * 6.2831853);

	vec3 glowColor = vec3(1.0, 0.6, 0.22);
	float alpha = falloff * 0.6 * flicker;
	outColor = vec4(glowColor, clamp(alpha, 0.0, 1.0));
}
