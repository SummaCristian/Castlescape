// FRAGMENT SHADER for the flame's ember shard particles. Unlit, bright,
// alpha-blended (same reasoning as FlameGlow.frag: this renderer's Pipeline
// only exposes SRC_ALPHA/ONE_MINUS_SRC_ALPHA, not true additive blending).
// Fades in quickly after spawn and out before its loop resets, so the reset
// itself is never visible -- a shard is already invisible on both ends of
// tphase.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) flat in float tphase;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 hot  = vec3(1.0, 0.95, 0.6);	// bright yellow, freshly out of the flame
	vec3 cool = vec3(1.0, 0.4, 0.05);	// orange, cooling as it rises
	vec3 color = mix(hot, cool, tphase);

	float alpha = smoothstep(0.0, 0.12, tphase) * (1.0 - smoothstep(0.6, 1.0, tphase));
	// Same "clip on purpose" overexposure idea as Flame.frag: a spark should
	// read as a bright pinprick, not a flat-colored dot.
	outColor = vec4(color * 1.6, alpha);
}
