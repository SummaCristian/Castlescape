// FRAGMENT SHADER for the flame's spark particles (see Spark.vert). Unlit,
// HDR-bright, alpha-blended -- same reasoning as Flame.frag: a spark is a
// tiny fleck of burning fuel, it emits, there's nothing to light it FROM.
// Replaces the old Ember.frag, which faked "bright" by clipping color at a
// fixed 1.6x multiplier because there was no HDR target or bloom pass to
// hand real overbright values to. This renderer now has both, so sparks
// write real HDR values and let bloom turn them into the pinpricks of light
// they're supposed to read as.

#version 450
#extension GL_ARB_separate_shader_objects : enable

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

	// Hot yellow-white at birth, cooling to deep orange by death -- the same
	// two-stop idea the old Ember.frag used, just with brighter HDR output
	// at the hot end instead of a clipped flat multiplier.
	vec3 hot = vec3(1.00, 0.95, 0.75);
	vec3 cool = vec3(0.95, 0.35, 0.05);
	vec3 color = mix(hot, cool, life);

	// Brightest the instant it's thrown off, dimming as it cools -- so a
	// spark is a genuine bloom source only near birth, not for its whole
	// short life.
	float hdrBoost = mix(12.0, 3.0, life);

	outColor = vec4(color * hdrBoost * glow, alpha);
}
