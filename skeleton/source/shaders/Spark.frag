// FRAGMENT SHADER for the flame's spark particles (see Spark.vert). Unlit,
// HDR-bright, alpha-blended -- same reasoning as Flame.frag: a spark is a
// tiny fleck of burning fuel, it emits, there's nothing to light it FROM.
// Sparks write real HDR values and let the bloom pass downstream turn them
// into the pinpricks of light they're supposed to read as.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// rgb2hsv/hsv2rgb/recolorStop, shared with Flame.frag so sparks recolor onto
// `color` the same way the flame body does.
#include "custom/FlameColor.glsl"

// quv: the spark's raw, unstretched quad corner (see Spark.vert for why this
// alone is enough to reconstruct an elliptical falloff without also passing
// the streak's stretch factor).
layout(location = 0) in vec2 quv;
// 0 at spawn, 1 at despawn.
layout(location = 1) flat in float life;
// Intensity gate and envelope glow from Spark.vert: the gate folds into
// alpha so a spark overtaken mid-life by a guttering flame dims away like a
// dying spark; glow scales brightness gently with the flame's envelope.
layout(location = 2) flat in float gate;
layout(location = 3) flat in float glow;
// This flame's target hue, see Flame.frag's recolorStop() for the technique.
layout(location = 4) flat in vec3 color;

layout(location = 0) out vec4 outColor;

void main() {
	float d = length(quv);
	if(d > 1.0) {
		discard;
	}

	// Squared falloff: a soft streak with no visible hard edge, rather than
	// a disc/capsule silhouette.
	float falloff = (1.0 - d) * (1.0 - d);

	// Fade in fast right after spawn and out before the loop resets, so
	// `life` ever actually reaching 0 or 1 is never visible -- the spark is
	// already fully transparent on both ends of its own loop.
	float fade = smoothstep(0.0, 0.08, life) * (1.0 - smoothstep(0.75, 1.0, life));
	float alpha = falloff * fade * gate;

	// Same depth-write reasoning as Flame.frag: depthWriteEnable is hardcoded
	// VK_TRUE on every pipeline including this one, so an undiscarded
	// near-zero-alpha fringe would still punch a hole through the flame body
	// and through other sparks behind it.
	if(alpha < 0.02) {
		discard;
	}

	// Hot yellow-white at birth, cooling to deep orange by death -- hue-rotated
	// onto this flame's `color` exactly like Flame.frag's own stops, so a
	// spark thrown off a colored flame reads as that same color of fire
	// rather than always the default orange regardless of the flame it came
	// from.
	const vec3 REF_ORANGE = vec3(1.00, 0.42, 0.05);	// same reference as Flame.frag
	vec3 targetHsv = rgb2hsv(color);
	float hueDelta = targetHsv.x - rgb2hsv(REF_ORANGE).x;
	float hueDist = min(abs(hueDelta), 1.0 - abs(hueDelta));

	vec3 hot  = recolorStop(vec3(1.00, 0.95, 0.75), hueDelta, hueDist, targetHsv.y, 0.55);
	vec3 cool = recolorStop(vec3(0.95, 0.35, 0.05), hueDelta, hueDist, targetHsv.y, 0.0);
	vec3 sparkColor = mix(hot, cool, life);

	// Brightest the instant it's thrown off, dimming as it cools -- so a
	// spark is a genuine bloom source only near birth, not for its whole
	// short life.
	float hdrBoost = mix(12.0, 3.0, life);

	outColor = vec4(sparkColor * hdrBoost * glow, alpha);
}
