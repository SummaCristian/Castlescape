// Recoloring shared by Flame.frag and Spark.frag.
// Hue-rotates the baked orange gradient instead of replacing it, so the
// authored shape (pale core -> saturated tip) survives.

// RGB<->HSV, branchless form (Sam Hocevar).
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

// Rotates stop hue by hueDelta, pulls saturation toward targetSat (max satBlendMax).
// hueDist = distance from default orange; gates the pull so default color is untouched.
// Saturation pull needed for near-white stops: hue rotation alone can't tint grey.
vec3 recolorStop(vec3 stop, float hueDelta, float hueDist, float targetSat, float satBlendMax) {
	vec3 hsv = rgb2hsv(stop);
	hsv.x = fract(hsv.x + hueDelta);
	hsv.y = mix(hsv.y, targetSat, satBlendMax * smoothstep(0.0, 0.04, hueDist));
	return hsv2rgb(hsv);
}
