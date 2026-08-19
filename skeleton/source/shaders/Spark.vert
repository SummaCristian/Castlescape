// VERTEX SHADER for the flame's spark particles (see custom/Flame.hpp and
// Flame.vert). Entirely procedural, same spirit as the old Ember.vert it
// replaces: no CPU-side particle system and no per-particle uniform data
// beyond what's already baked into the mesh (inSeed) and already bound for
// the flame body (gubo.time, fubo.mvpMat/seed/intensity/lean). Every spark's
// whole lifecycle -- spawn, rise, drift, stretch, shrink -- is a function of
// those, looping forever via fract(), and every spark reuses the flame
// body's OWN descriptor set (set 1) rather than getting one of its own, the
// same reasoning the old Ember.vert used: a spark needs the torch's mvp and
// seed and nothing else, so there's no reason to keep a second copy in step.
// Unlike the old embers, sparks also pick up fubo.lean, so they visibly
// follow the same wind/motion that's bending the flame body itself.

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

layout(binding = 0, set = 1) uniform FlameUniformBufferObject {
	mat4  mvpMat;
	float seed;
	float intensity;
	vec2  lean;
} fubo;

// Quad corner in the spark's own unstretched local units, both axes in
// [-1,1]: x becomes the streak's width, y its length, once main() below
// expands it along the direction the spark is actually travelling.
layout(location = 0) in vec2 inCorner;
// Fixed per-spark random in [0,1), baked in at mesh-build time -- drives
// this spark's spawn point, loop period/phase, drift and streak length so
// SPARK_COUNT sparks don't all move identically.
layout(location = 1) in float inSeed;

// The raw corner, unstretched -- Spark.frag uses length(quv) for a radial
// falloff. Because the expansion below is a linear (affine) map of inCorner,
// interpolating the raw corner across the ALREADY-stretched quad still lines
// up perfectly with the physical shape, so a plain circular falloff on quv
// automatically reads as an ellipse stretched along the streak -- no need to
// also pass the stretch factor to the fragment shader.
layout(location = 0) out vec2 quv;
// 0 at spawn, 1 at despawn; Spark.frag uses it for both the color ramp and
// the fade in/out that hides the loop reset.
layout(location = 1) flat out float life;

// Cheap 1D hash, Dave Hoskins' construction: three fract/multiply rounds are
// enough to decorrelate the handful of quantities derived from inSeed below
// (spawn x/y, drift sign/amount, streak length) so they don't all covary
// with inSeed in the same direction.
float hash11(float x) {
	x = fract(x * 0.1031);
	x *= x + 33.33;
	x *= x + x;
	return fract(x);
}

void main() {
	// Loop period varies per spark (1.6..3.2s), phase-offset by both the
	// spark's own seed and the torch's, so sparks across however many
	// torches are in the scene never pop back to spawn in visible unison.
	float period = 1.6 + inSeed * 1.6;
	float t = fract((gubo.time + fubo.seed * 11.0 + inSeed * 29.0) / period);
	life = t;

	// Spawn scattered across the CROWN (y=0.55..0.8, x=+-0.35), not the
	// wick: sparks are flecks the flame's tip throws off, not fuel rising
	// from the base the way the flame body's own noise does.
	float spawnX = (hash11(inSeed * 17.0) - 0.5) * 0.55;
	float spawnY = mix(0.42, 0.62, hash11(inSeed * 31.0 + 4.1));

	// Sideways drift grows with age (t*t, not t): a spark barely wanders
	// right after being thrown off and curls away increasingly as it cools
	// and slows, rather than drifting at a constant rate its whole life.
	float driftSign = hash11(inSeed * 53.0) < 0.5 ? -1.0 : 1.0;
	float driftAmount = t * t * mix(0.15, 0.35, hash11(inSeed * 61.0)) * driftSign;

	// Rise is close to linear in age -- real sparks decelerate too, but the
	// effect is subtle over their short life and t alone reads cleanly.
	float rise = t * mix(0.30, 0.52, hash11(inSeed * 7.0));

	vec2 center = vec2(spawnX + driftAmount, spawnY + rise);

	// Same lean the flame body itself is bent by, damped: sparks are small
	// and light, so a gust visibly nudges them without swinging them as
	// hard as it swings the flame's own tip (Flame.vert's h*h term).
	center += fubo.lean * 0.6;

	// Direction of travel, for orienting the streak: rise is roughly
	// constant while drift grows with age, so a young spark's direction is
	// nearly straight up and it tilts further sideways as it ages -- the
	// derivative of the same center curve above, not an independent value.
	vec2 travelDir = normalize(vec2(driftAmount, max(rise, 0.05)));
	vec2 perpDir = vec2(-travelDir.y, travelDir.x);

	// Small to begin with and shrinking further with age: a few hundredths
	// of the flame's own half-width (mvpMat's x=1 unit is that half-width),
	// so a spark reads as a fleck against the flame body, never another
	// flame of its own.
	float size = mix(0.05, 0.015, t);
	// Stretched 2.5-4x along its direction of travel so it reads as a
	// streak, not a dot -- varied per spark so they don't all look identical.
	float stretch = mix(2.5, 4.0, hash11(inSeed * 71.0));

	vec2 local = perpDir * (inCorner.x * size) + travelDir * (inCorner.y * size * stretch);

	// A small per-spark, time-independent z jitter: not motion, just enough
	// depth spread that sparks don't all sit exactly on the flame body's own
	// z=0 plane, the same parallax reasoning behind Flame.vert's layer
	// z-offset.
	float z = (hash11(inSeed * 83.0) - 0.5) * 0.15;

	gl_Position = fubo.mvpMat * vec4(center + local, z, 1.0);
	quv = inCorner;
}
