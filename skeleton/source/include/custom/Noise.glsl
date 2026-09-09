// ***** CUSTOM *****
//
// Procedural noise used by the flame shaders: Flame.vert displaces the mesh
// with it, Flame.frag colors it. It replaces the Blender Clouds texture the
// flame was modelled with, since glTF can export animated nodes but not a
// modifier stack.
//
// Lives here and not in shaders/ because include/ is the directory CMake passes
// to glslc with -I (same as for the C++ compiler).

#ifndef NOISE_GLSL
#define NOISE_GLSL

// Deterministic pseudo-random value in [0,1) from a 2D point. The constants are
// arbitrary; they only have to scramble neighbouring integers apart.
float hash21(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Value noise: random values on the integer grid, interpolated in between. The
// smoothstep curve on the fractional part hides the grid, which plain linear
// interpolation would leave visible.
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

// Fractal Brownian motion: 4 octaves of valueNoise, each double the frequency
// and half the amplitude. One octave is too smooth to read as fire; the higher
// ones add the fine wisps. Returns ~[0,1].
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
