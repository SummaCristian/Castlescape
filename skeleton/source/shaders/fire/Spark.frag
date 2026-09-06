// FRAGMENT SHADER for the flame's spark particles (Spark.vert). Unlit, HDR,
// alpha-blended -- same as Flame.frag: a spark emits. Writes real HDR values
// and lets the bloom pass turn them into pinpricks of light.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// rgb2hsv/hsv2rgb/recolorStop, shared with Flame.frag.
#include "custom/FlameColor.glsl"

layout(location = 0) in vec2 quv;         // raw unstretched quad corner (Spark.vert)
layout(location = 1) flat in float life;  // 0 spawn, 1 despawn
layout(location = 2) flat in float gate;  // intensity gate, folds into alpha
layout(location = 3) flat in float glow;  // envelope brightness coupling
layout(location = 4) flat in vec3 color;  // this flame's target hue
layout(location = 5) flat in float mote;  // 1 dust mote: lit dust, not burning fuel

layout(location = 0) out vec4 outColor;

void main() {
	float d = length(quv);
	if(d > 1.0) {
		discard;
	}

	// Squared falloff: a soft streak with no hard edge.
	float falloff = (1.0 - d) * (1.0 - d);

	// Fade in after spawn and out before the loop resets, so `life` reaching
	// 0 or 1 is never visible.
	float fade = mote > 0.5
		? smoothstep(0.0, 0.15, life) * (1.0 - smoothstep(0.55, 1.0, life))
		: smoothstep(0.0, 0.08, life) * (1.0 - smoothstep(0.75, 1.0, life));
	float alpha = falloff * fade * gate;
	// Motes are faint: they only catch the light, they don't emit.
	if(mote > 0.5) {
		alpha *= 0.45;
	}

	// depthWriteEnable is hardcoded on, so an undiscarded near-zero-alpha
	// fringe would punch a hole through the flame body and other sparks.
	if(alpha < 0.02) {
		discard;
	}

	// Hot yellow-white at birth, cooling to deep orange -- hue-rotated onto
	// this flame's `color` like Flame.frag's stops, so a spark off a coloured
	// flame reads as that colour of fire.
	const vec3 REF_ORANGE = vec3(1.00, 0.42, 0.05);	// same reference as Flame.frag
	vec3 targetHsv = rgb2hsv(color);
	float hueDelta = targetHsv.x - rgb2hsv(REF_ORANGE).x;
	float hueDist = min(abs(hueDelta), 1.0 - abs(hueDelta));

	vec3 hot  = recolorStop(vec3(1.00, 0.95, 0.75), hueDelta, hueDist, targetHsv.y, 0.55);
	vec3 cool = recolorStop(vec3(0.95, 0.35, 0.05), hueDelta, hueDist, targetHsv.y, 0.0);
	vec3 sparkColor = mix(hot, cool, life);

	// Brightest when thrown off, dimming as it cools -- a bloom source only
	// near birth.
	float hdrBoost = mix(12.0, 3.0, life);

	// A mote is dust lit by the fire: a warm grey carrying a little of the
	// flame's colour, near unit brightness so it barely touches the bloom.
	if(mote > 0.5) {
		sparkColor = mix(vec3(0.85, 0.72, 0.52), color, 0.3);
		hdrBoost = mix(1.8, 0.6, life);
	}

	outColor = vec4(sparkColor * hdrBoost * glow, alpha);
}
