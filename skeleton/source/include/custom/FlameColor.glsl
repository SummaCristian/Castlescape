// Recoloring shared by Flame.frag and Spark.frag.
//
// Both shaders have a hand-tuned orange gradient baked in. To draw a flame of a
// different color they don't replace that gradient, they hue-rotate it, so the
// authored shape of the gradient (hot pale core -> saturated tip) survives. The
// per-instance target color comes from FlameUniformBufferObject (Flame.hpp).

// Standard RGB<->HSV pair (Sam Hocevar's branchless form).
vec3 rgb2hsv(vec3 c) {
	vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1.0e-10;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
	vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
	return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Recolors one gradient stop: rotates its hue by `hueDelta` and pulls its
// saturation toward `targetSat` by at most `satBlendMax`.
//
// `hueDist` is how far the target hue is from the default orange, and gates the
// saturation pull: at 0 (default color) the stop comes back unchanged, so the
// orange flame stays exactly as authored.
//
// The saturation pull exists for the near-white stops (the hot core): they are
// almost grey, so hue rotation alone does nothing to them and a blue flame
// would still have a white center. Pulling saturation up tints them too.
vec3 recolorStop(vec3 stop, float hueDelta, float hueDist, float targetSat, float satBlendMax) {
	vec3 hsv = rgb2hsv(stop);
	hsv.x = fract(hsv.x + hueDelta);
	hsv.y = mix(hsv.y, targetSat, satBlendMax * smoothstep(0.0, 0.04, hueDist));
	return hsv2rgb(hsv);
}
