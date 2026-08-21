// Shared by Flame.frag and Spark.frag: both recolor an authored,
// realistic-orange palette onto a per-instance target `color` (see
// custom/Flame.hpp's FlameUniformBufferObject) by hue-rotating each of their
// own fixed stops rather than replacing them wholesale -- see recolorStop()
// below for why. Kept in one file so the flame body and its sparks always
// recolor the same way instead of two copies quietly drifting apart.

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

// Rotates one authored stop's hue by `hueDelta`, and pulls its saturation
// toward `targetSat` by up to `satBlendMax` -- gated by `hueDist` (the
// circular distance between the target color's hue and the reference
// orange's) so a flame/spark left at the default color gets hueDelta==0 and
// hueDist==0 and comes back OUT UNCHANGED, bit for bit. Only something
// actually recolored pays for any saturation shift, and pays for it
// gradually, not as a step the moment `color` moves off the default.
//
// The saturation pull matters most for near-white stops (a flame's hot core,
// a spark's hot-yellow birth): authored low-saturation because a real
// flame/spark burns toward white at its hottest regardless of fuel --
// correct for the default orange, but it would leave a colored flame fading
// to a bland white instead of a visibly tinted hot point. Pulling it toward
// the target's own saturation keeps the whole gradient reading as "this
// color of fire," hottest point included.
vec3 recolorStop(vec3 stop, float hueDelta, float hueDist, float targetSat, float satBlendMax) {
	vec3 hsv = rgb2hsv(stop);
	hsv.x = fract(hsv.x + hueDelta);
	hsv.y = mix(hsv.y, targetSat, satBlendMax * smoothstep(0.0, 0.04, hueDist));
	return hsv2rgb(hsv);
}
