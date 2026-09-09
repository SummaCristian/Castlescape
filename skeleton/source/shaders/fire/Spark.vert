// VERTEX SHADER for the flame's spark particles (custom/Flame.hpp, Flame.vert).
// Entirely procedural: no CPU particle system, just the baked per-spark seed
// (inSeed) and the flame body's own uniforms. Every spark's lifecycle --
// spawn, rise, drift, stretch, shrink -- is a function of those, looping
// forever via fract(), and it reuses the flame body's descriptor set (set 1).
// Sparks also pick up fubo.lean, so they follow the same wind as the flame.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	int debugFlags;
	float time;
} gubo;

layout(binding = 0, set = 1) uniform FlameUniformBufferObject {
	mat4  mvpMat;
	float seed;
	float intensity;   // brightness envelope (spring-smoothed), see Flame.vert
	vec2  lean;
	float heightScale; // the spawn crown rides the flame's actual height
	float glareBoost;  // unused by sparks: boosting pinpricks just makes white dots
	vec3  color;       // this flame's target hue, for Spark.frag
} fubo;

// Quad corner, both axes [-1,1]: x is the streak's width, y its length, once
// main() expands it along the travel direction.
layout(location = 0) in vec2 inCorner;
// Fixed per-spark random in [0,1), baked at mesh-build: spawn point, loop
// period/phase, drift, streak length, so the sparks don't move identically.
layout(location = 1) in float inSeed;

// The raw unstretched corner. The expansion below is affine, so interpolating
// it across the stretched quad still lines up, and a circular falloff on quv
// reads as an ellipse along the streak -- no need to pass the stretch factor.
layout(location = 0) out vec2 quv;
layout(location = 1) flat out float life;  // 0 spawn, 1 despawn; colour ramp + loop-hiding fade
layout(location = 2) flat out float gate;  // per-spark intensity gate, so a gated spark dies out, not pops
layout(location = 3) flat out float glow;  // gentle coupling to the flame envelope
layout(location = 4) flat out vec3 color;
layout(location = 5) flat out float mote;  // 1 dust mote, 0 spark -- same quads, slower/dimmer path

// Fraction of the baked particle set that drifts as dust rather than flying
// off as sparks.
const float MOTE_FRACTION = 0.42;

// Cheap 1D hash (Dave Hoskins): decorrelates the quantities derived from
// inSeed so they don't all covary with it.
float hash11(float x) {
	x = fract(x * 0.1031);
	x *= x + 33.33;
	x *= x + x;
	return fract(x);
}

// 1D value noise on hash11: a smooth random walk for the wobble. The
// 0.517/0.13 constants just decorrelate the lattice from hash11's other uses.
float noise11(float x) {
	float cell = floor(x);
	float cellFrac = fract(x);
	float smoothFrac = cellFrac * cellFrac * (3.0 - 2.0 * cellFrac);
	return mix(hash11(cell * 0.517 + 0.13), hash11((cell + 1.0) * 0.517 + 0.13), smoothFrac);
}

void main() {
	bool isMote = hash11(inSeed * 13.0) < MOTE_FRACTION;

	// Loop period varies per spark, phase-offset by both seeds, so sparks
	// never pop back to spawn in unison. Motes live several times longer and
	// hang rather than fly, reading as dust in the light, not thrown fuel.
	float period = isMote ? (7.0 + inSeed * 8.0) : (1.6 + inSeed * 1.6);
	float t = fract((gubo.time + fubo.seed * 11.0 + inSeed * 29.0) / period);
	life = t;

	// Spawn across the CROWN, not the wick: sparks are flecks the tip throws
	// off, not fuel rising from the base.
	float spawnX = (hash11(inSeed * 17.0) - 0.5) * 0.55;
	float spawnY = mix(0.42, 0.62, hash11(inSeed * 31.0 + 4.1));

	// Sideways drift grows with age (t*t): a spark barely wanders when thrown
	// and curls away as it cools and slows.
	float driftSign = hash11(inSeed * 53.0) < 0.5 ? -1.0 : 1.0;
	float driftAmount = t * t * mix(0.15, 0.35, hash11(inSeed * 61.0)) * driftSign;

	// Rise ~linear in age; real deceleration is subtle over a spark's life.
	float rise = t * mix(0.30, 0.52, hash11(inSeed * 7.0));

	// Wobble: a per-spark noise walk on top of the drift, sampled along TIME
	// so it's motion, not a static bend. Amplitude grows with age -- a young
	// spark rides the updraft, an old one is at the mercy of the eddies.
	float wob = (noise11(t * 3.0 + inSeed * 47.0) - 0.5) * 0.20 * t;

	// The crown tracks the flame's height envelope, so a guttering flame's
	// sparks spawn lower instead of popping out of empty air above it.
	vec2 center = vec2(spawnX + driftAmount + wob,
	                   (spawnY + rise) * fubo.heightScale);

	// Same lean the flame is bent by, damped: a gust nudges a light spark
	// without swinging it as hard as the flame's tip.
	center += fubo.lean * 0.6;

	// Motes ignore the crown and sit in a wide box around the flame, wandering
	// on a time-sampled noise walk with a faint net rise. No age-weighting.
	if(isMote) {
		float boxX = (hash11(inSeed * 17.0) - 0.5) * 1.7;
		float boxY = mix(0.10, 0.85, hash11(inSeed * 31.0 + 4.1));
		float wanderX = (noise11(gubo.time * 0.25 + inSeed * 40.0) - 0.5) * 0.5;
		float wanderY = (noise11(gubo.time * 0.20 + inSeed * 70.0) - 0.5) * 0.4;
		center = vec2(boxX + wanderX,
		              (boxY + wanderY + t * 0.12) * fubo.heightScale);
		center += fubo.lean * 0.3;
	}

	// Travel direction, for orienting the streak: the derivative of the center
	// curve above -- straight up when young, tilting sideways as drift grows.
	vec2 travelDir = normalize(vec2(driftAmount, max(rise, 0.05)));
	vec2 perpDir = vec2(-travelDir.y, travelDir.x);

	// "Spawn rate" without a dynamic mesh: the spark count is baked, so rate is
	// faked with a per-spark intensity THRESHOLD the flame envelope gates on.
	// Thresholds 0.45..1.15 vs an envelope of 0.30..1.40, so the dimmest ~40%
	// always burn and the rest arrive as bursts on a flare. The spring-smoothed
	// envelope turns the smoothstep band into an ~0.1-0.3 s fade, not a pop.
	float thr = mix(0.45, 1.15, hash11(inSeed * 91.0));
	gate = smoothstep(thr - 0.10, thr + 0.10, fubo.intensity);
	// Motes never gate fully off -- dust doesn't vanish when the flame gutters.
	if(isMote) {
		gate = mix(0.35, 1.0, clamp((fubo.intensity - 0.3) / 1.1, 0.0, 1.0));
	}

	// Small, shrinking with age: a few hundredths of the flame's half-width,
	// so a spark reads as a fleck, never another flame. A gated-off spark
	// collapses to a degenerate quad and costs no fragment work while it waits.
	float size = mix(0.05, 0.015, t)
	           * mix(0.85, 1.15, clamp(fubo.intensity, 0.3, 1.4) / 1.4);
	size *= max(gate, 0.001);
	glow = mix(0.75, 1.20, clamp(fubo.intensity, 0.3, 1.4) / 1.4);
	// Stretched 2.5-4x along travel so it reads as a streak, not a dot.
	float stretch = mix(2.5, 4.0, hash11(inSeed * 71.0));

	// Motes are small round specks: near-constant size, no streak.
	if(isMote) {
		size = mix(0.018, 0.032, hash11(inSeed * 41.0)) * max(gate, 0.001);
		stretch = 1.0;
	}

	vec2 local = perpDir * (inCorner.x * size) + travelDir * (inCorner.y * size * stretch);

	// A small time-independent z jitter, so sparks don't all sit on the flame
	// body's z=0 plane. Same parallax reasoning as Flame.vert's layer offset.
	float z = (hash11(inSeed * 83.0) - 0.5) * (isMote ? 0.8 : 0.15);

	gl_Position = fubo.mvpMat * vec4(center + local, z, 1.0);
	quv = inCorner;
	color = fubo.color;
	mote = isMote ? 1.0 : 0.0;
}
