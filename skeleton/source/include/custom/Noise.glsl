// ***** CUSTOM *****
//
// Procedural noise shared by the flame's two shader stages (Flame.vert
// displaces the mesh with it, Flame.frag colours the mesh with it). Lives in
// include/custom/ rather than in shaders/ because that is the directory
// CMakeLists.txt hands glslc with -I, the same one the C++ compiler gets --
// see LightConstants.glsl, which is shared between C++ and GLSL the same way.
//
// This is the GPU stand-in for Blender's Clouds texture: the flame was
// originally sculpted with two Displace modifiers reading a Clouds texture
// through a slowly-turning Empty, and glTF cannot carry that (it animates
// nodes, skins and morph targets -- never a modifier stack). Evaluating
// equivalent noise per vertex/per pixel reproduces the idea without needing
// an animation format that can express it.

#ifndef NOISE_GLSL
#define NOISE_GLSL

// Deterministic pseudo-random value in [0,1) from a 2D lattice point. The
// constants are the usual arbitrary large irrationals: any pair works, they
// only have to scramble neighbouring integers apart.
float hash21(vec2 p) {
	return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Value noise: random values at integer lattice points, smoothly interpolated
// between. The smoothstep on the fractional part is what keeps the result from
// showing the underlying square grid the way plain linear interpolation does.
float valueNoise(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);

	float a = hash21(i);
	float b = hash21(i + vec2(1.0, 0.0));
	float c = hash21(i + vec2(0.0, 1.0));
	float d = hash21(i + vec2(1.0, 1.0));

	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Fractal Brownian motion: four octaves of valueNoise at doubling frequency
// and halving amplitude. One octave alone is too smooth to read as fire; the
// higher octaves are what put the fine wisps on the edges. Returns ~[0,1].
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
