// FRAGMENT SHADER: flame spark particles (Spark.vert). Unlit, HDR,
// alpha-blended -- a spark emits; bloom turns them into pinpricks of light.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// rgb2hsv/hsv2rgb/recolorStop, shared with Flame.frag.
#include "custom/FlameColor.glsl"

layout(location = 0) in vec2 quv;         // raw unstretched quad corner
layout(location = 1) flat in float life;  // 0 spawn, 1 despawn
layout(location = 2) flat in float gate;  // intensity gate, folds into alpha
layout(location = 3) flat in float glow;  // envelope brightness coupling
layout(location = 4) flat in vec3 color;  // target hue
layout(location = 5) flat in float mote;  // 1 dust mote, 0 spark

layout(location = 0) out vec4 outColor;

void main() {
	float dist = length(quv);
	if(dist > 1.0) {
		discard;
	}

	// Squared falloff: soft streak, no hard edge.
	float falloff = (1.0 - dist) * (1.0 - dist);

	// Fade in/out around the loop reset so life hitting 0/1 is never visible.
	float fade = mote > 0.5
		? smoothstep(0.0, 0.15, life) * (1.0 - smoothstep(0.55, 1.0, life))
		: smoothstep(0.0, 0.08, life) * (1.0 - smoothstep(0.75, 1.0, life));
	float alpha = falloff * fade * gate;
	// Motes are faint: they catch light, don't emit.
	if(mote > 0.5) {
		alpha *= 0.45;
	}

	// depthWriteEnable is on; an undiscarded near-zero fringe would punch a
	// hole through the flame body and other sparks.
	if(alpha < 0.02) {
		discard;
	}

	// Hot yellow-white at birth, cooling to deep orange, hue-rotated onto
	// `color` like Flame.frag's stops.
	const vec3 REF_ORANGE = vec3(1.00, 0.42, 0.05);
	vec3 targetHsv = rgb2hsv(color);
	float hueDelta = targetHsv.x - rgb2hsv(REF_ORANGE).x;
	float hueDist = min(abs(hueDelta), 1.0 - abs(hueDelta));

	vec3 hot  = recolorStop(vec3(1.00, 0.95, 0.75), hueDelta, hueDist, targetHsv.y, 0.55);
	vec3 cool = recolorStop(vec3(0.95, 0.35, 0.05), hueDelta, hueDist, targetHsv.y, 0.0);
	vec3 sparkColor = mix(hot, cool, life);

	// Brightest at birth, dimming as it cools; bloom source only near birth.
	float hdrBoost = mix(12.0, 3.0, life);

	// A mote is dust lit by the fire: warm grey with a hint of flame color,
	// near unit brightness so it barely touches bloom.
	if(mote > 0.5) {
		sparkColor = mix(vec3(0.85, 0.72, 0.52), color, 0.3);
		hdrBoost = mix(1.8, 0.6, life);
	}

	outColor = vec4(sparkColor * hdrBoost * glow, alpha);
}
