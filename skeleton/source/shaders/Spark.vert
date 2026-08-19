// VERTEX SHADER for the flame's spark particles (see custom/Flame.hpp and
// Flame.vert). Entirely procedural: no CPU-side particle system and no
// per-particle uniform data beyond what's already baked into the mesh
// (inSeed) and already bound for the flame body (gubo.time,
// fubo.mvpMat/seed/intensity/lean). Every spark's whole lifecycle -- spawn,
// rise, drift, stretch, shrink -- is a function of those, looping forever
// via fract(), and every spark reuses the flame body's OWN descriptor set
// (set 1) rather than getting one of its own: a spark needs the torch's mvp
// and seed and nothing else, so there's no reason to keep a second copy in
// step. Sparks also pick up fubo.lean, so they visibly follow the same
// wind/motion that's bending the flame body itself.

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
	float intensity;   // brightness envelope (spring-smoothed), see Flame.vert
	vec2  lean;
	float heightScale; // slow height envelope -- the crown the sparks spawn
	                   // from rides the flame's actual height, see below
	float glareBoost;  // unused by sparks: bloom already turns them into
	                   // pinpricks, boosting them further just makes white dots
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
// How "on" this spark is, 0..1: the per-spark intensity gate below. Passed to
// the fragment shader so a spark caught mid-life by a guttering flame fades
// out like a dying spark instead of popping off.
layout(location = 2) flat out float gate;
// Gentle brightness coupling to the flame envelope, applied in Spark.frag on
// top of the per-spark life ramp.
layout(location = 3) flat out float glow;

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

// 1D value noise on top of hash11: a smooth random walk rather than a jump
// per cell, for the wobble below. The 0.517/0.13 constants just decorrelate
// the lattice from the raw seed multiples hash11 is fed elsewhere.
float noise11(float x) {
	float i = floor(x);
	float f = fract(x);
	float u = f * f * (3.0 - 2.0 * f);
	return mix(hash11(i * 0.517 + 0.13), hash11((i + 1.0) * 0.517 + 0.13), u);
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

	// Wobble: a per-spark noise walk on top of the smooth t*t drift, so the
	// path curls and corrects the way a fleck tossed on turbulent air does,
	// instead of following one clean parabola. Amplitude grows with age (a
	// young spark is still carried by the flame's own updraft, an old one is
	// at the mercy of the eddies) and the noise is sampled along TIME, so
	// the wobble is motion, not a static bend.
	float wob = (noise11(t * 3.0 + inSeed * 47.0) - 0.5) * 0.20 * t;

	// The crown tracks the flame's actual height envelope: when the flame
	// gutters short, sparks spawn (and fly) correspondingly lower instead of
	// popping out of empty air above a shrunken flame.
	vec2 center = vec2(spawnX + driftAmount + wob,
	                   (spawnY + rise) * fubo.heightScale);

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

	// "Spawn rate" without a dynamic mesh: the spark count is baked in, so
	// rate is faked by giving every spark its own intensity THRESHOLD and
	// letting the flame envelope gate it on. Thresholds spread 0.45..1.15
	// against an envelope of 0.30..1.40, so roughly the dimmest 40% of
	// thresholds are always burning and the rest arrive as bursts when the
	// flame flares -- or return in a rush as it recovers from a gutter.
	// Since the envelope is spring-smoothed CPU-side, the smoothstep band
	// here turns into an ~0.1-0.3 s fade rather than a pop.
	float thr = mix(0.45, 1.15, hash11(inSeed * 91.0));
	gate = smoothstep(thr - 0.10, thr + 0.10, fubo.intensity);

	// Small to begin with and shrinking further with age: a few hundredths
	// of the flame's own half-width (mvpMat's x=1 unit is that half-width),
	// so a spark reads as a fleck against the flame body, never another
	// flame of its own. A flaring flame throws slightly bigger flecks; a
	// fully gated-off spark collapses to a degenerate quad, so it costs no
	// fragment work at all while it waits.
	float size = mix(0.05, 0.015, t)
	           * mix(0.85, 1.15, clamp(fubo.intensity, 0.3, 1.4) / 1.4);
	size *= max(gate, 0.001);
	glow = mix(0.75, 1.20, clamp(fubo.intensity, 0.3, 1.4) / 1.4);
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
