// ***** CUSTOM *****
// Procedural noise for the flame shaders: Flame.vert displaces mesh,
// Flame.frag colors it. Replaces a Blender Clouds texture (glTF can't
// export a modifier stack).

#ifndef NOISE_GLSL
#define NOISE_GLSL

// Pseudo-random [0,1) from 2D point; constants just scramble neighbours apart.
float hash21(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Value noise: grid corners interpolated; smoothstep on the fraction hides the grid.
float valueNoise(vec2 p) {
	vec2 cell = floor(p);
	vec2 cellFrac = fract(p);

	float cornerBL = hash21(cell);
	float cornerBR = hash21(cell + vec2(1.0, 0.0));
	float cornerTL = hash21(cell + vec2(0.0, 1.0));
	float cornerTR = hash21(cell + vec2(1.0, 1.0));

	vec2 smoothFrac = cellFrac * cellFrac * (3.0 - 2.0 * cellFrac);
	return mix(mix(cornerBL, cornerBR, smoothFrac.x), mix(cornerTL, cornerTR, smoothFrac.x), smoothFrac.y);
}

// fBm: 4 octaves, doubling frequency / halving amplitude each step. Returns ~[0,1].
float fbm(vec2 p) {
	float sum = 0.0;
	float amp = 0.5;
	for(int i = 0; i < 4; i++) {
		sum += amp * valueNoise(p);
		p *= 2.0;
		amp *= 0.5;
	}
	return sum;
}

#endif
