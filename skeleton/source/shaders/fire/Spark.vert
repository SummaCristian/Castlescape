// VERTEX SHADER: flame spark particles. Entirely procedural, no CPU particle
// system -- lifecycle (spawn/rise/drift/stretch/shrink) is a function of the
// baked per-spark seed and the flame's own uniforms (set 1), looping via fract().

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
	float intensity;   // brightness envelope, see Flame.vert
	vec2  lean;
	float heightScale; // spawn crown rides the flame's height
	float glareBoost;  // unused by sparks
	vec3  color;       // target hue, for Spark.frag
} fubo;

// x: streak width, y: length, expanded along travel dir in main().
layout(location = 0) in vec2 inCorner;
// Per-spark random in [0,1), baked at mesh-build: spawn/period/drift/length.
layout(location = 1) in float inSeed;

// Raw unstretched corner; expansion is affine so a circular falloff on quv
// reads as an ellipse along the streak.
layout(location = 0) out vec2 quv;
layout(location = 1) flat out float life;  // 0 spawn, 1 despawn
layout(location = 2) flat out float gate;  // per-spark intensity gate
layout(location = 3) flat out float glow;  // coupling to flame envelope
layout(location = 4) flat out vec3 color;
layout(location = 5) flat out float mote;  // 1 dust mote, 0 spark

// Fraction of particles that drift as dust rather than flying off as sparks.
const float MOTE_FRACTION = 0.42;

// Cheap 1D hash (Dave Hoskins), decorrelates quantities derived from inSeed.
float hash11(float x) {
	x = fract(x * 0.1031);
	x *= x + 33.33;
	x *= x + x;
	return fract(x);
}

// 1D value noise on hash11: smooth random walk for the wobble.
float noise11(float x) {
	float cell = floor(x);
	float cellFrac = fract(x);
	float smoothFrac = cellFrac * cellFrac * (3.0 - 2.0 * cellFrac);
	return mix(hash11(cell * 0.517 + 0.13), hash11((cell + 1.0) * 0.517 + 0.13), smoothFrac);
}

void main() {
	bool isMote = hash11(inSeed * 13.0) < MOTE_FRACTION;

	// Loop period per spark, phase-offset so sparks don't pop in unison.
	// Motes live longer and hang, reading as dust not thrown fuel.
	float period = isMote ? (7.0 + inSeed * 8.0) : (1.6 + inSeed * 1.6);
	float t = fract((gubo.time + fubo.seed * 11.0 + inSeed * 29.0) / period);
	life = t;

	// Spawn across the crown (tip), not the wick.
	float spawnX = (hash11(inSeed * 17.0) - 0.5) * 0.55;
	float spawnY = mix(0.42, 0.62, hash11(inSeed * 31.0 + 4.1));

	// Sideways drift grows with age (t*t): barely wanders when thrown, curls
	// away as it cools.
	float driftSign = hash11(inSeed * 53.0) < 0.5 ? -1.0 : 1.0;
	float driftAmount = t * t * mix(0.15, 0.35, hash11(inSeed * 61.0)) * driftSign;

	// Rise ~linear in age.
	float rise = t * mix(0.30, 0.52, hash11(inSeed * 7.0));

	// Wobble: per-spark noise walk sampled along time, amplitude grows with age.
	float wob = (noise11(t * 3.0 + inSeed * 47.0) - 0.5) * 0.20 * t;

	// Crown tracks the flame's height envelope, so sparks spawn lower on a
	// guttering flame instead of popping out of empty air.
	vec2 center = vec2(spawnX + driftAmount + wob,
	                   (spawnY + rise) * fubo.heightScale);

	// Same lean as the flame, damped.
	center += fubo.lean * 0.6;

	// Motes ignore the crown, wander in a wide box around the flame instead.
	if(isMote) {
		float boxX = (hash11(inSeed * 17.0) - 0.5) * 1.7;
		float boxY = mix(0.10, 0.85, hash11(inSeed * 31.0 + 4.1));
		float wanderX = (noise11(gubo.time * 0.25 + inSeed * 40.0) - 0.5) * 0.5;
		float wanderY = (noise11(gubo.time * 0.20 + inSeed * 70.0) - 0.5) * 0.4;
		center = vec2(boxX + wanderX,
		              (boxY + wanderY + t * 0.12) * fubo.heightScale);
		center += fubo.lean * 0.3;
	}

	// Travel direction: derivative of center curve, for orienting the streak.
	vec2 travelDir = normalize(vec2(driftAmount, max(rise, 0.05)));
	vec2 perpDir = vec2(-travelDir.y, travelDir.x);

	// Fake "spawn rate" without a dynamic mesh: per-spark intensity threshold
	// gated by the flame envelope. Dimmest ~40% always burn, rest arrive in
	// bursts on a flare.
	float thr = mix(0.45, 1.15, hash11(inSeed * 91.0));
	gate = smoothstep(thr - 0.10, thr + 0.10, fubo.intensity);
	// Motes never gate fully off.
	if(isMote) {
		gate = mix(0.35, 1.0, clamp((fubo.intensity - 0.3) / 1.1, 0.0, 1.0));
	}

	// Small, shrinking with age, so a gated-off spark collapses to a
	// degenerate quad (no fragment cost while waiting).
	float size = mix(0.05, 0.015, t)
	           * mix(0.85, 1.15, clamp(fubo.intensity, 0.3, 1.4) / 1.4);
	size *= max(gate, 0.001);
	glow = mix(0.75, 1.20, clamp(fubo.intensity, 0.3, 1.4) / 1.4);
	// Stretched along travel so it reads as a streak, not a dot.
	float stretch = mix(2.5, 4.0, hash11(inSeed * 71.0));

	// Motes are small round specks: near-constant size, no streak.
	if(isMote) {
		size = mix(0.018, 0.032, hash11(inSeed * 41.0)) * max(gate, 0.001);
		stretch = 1.0;
	}

	vec2 local = perpDir * (inCorner.x * size) + travelDir * (inCorner.y * size * stretch);

	// Small z jitter so sparks don't all sit on the flame's z=0 plane.
	float z = (hash11(inSeed * 83.0) - 0.5) * (isMote ? 0.8 : 0.15);

	gl_Position = fubo.mvpMat * vec4(center + local, z, 1.0);
	quv = inCorner;
	color = fubo.color;
	mote = isMote ? 1.0 : 0.0;
}
