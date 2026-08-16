// VERTEX SHADER for the low-poly flame (see custom/Flame.hpp). The mesh is a
// small hand-shaped stack of jittered rings, built once on the CPU; all the
// motion (turbulence, flicker) is added here every frame from gubo.time, so
// the GPU animates a completely static vertex/index buffer.
//
// The motion is driven by real value noise (turbulence() below), not a sum
// of sines: sines, however many are layered, all repeat on a beat a viewer's
// eye locks onto within a couple of seconds, which reads as a rigid shape
// swinging rather than something fluid. Noise doesn't repeat on any beat
// short enough to notice, which is the actual difference between "swaying"
// and "flowing".
//
// set 0 is the SAME global uniform the main pass uses (DSglobal in main.cpp,
// extended with a "time" field); a flame doesn't need eyePos or the light
// array, but binding the very same descriptor set means it doesn't need its
// own copy of eyePos/lightCount/etc. kept in step.
// set 1 is one small per-instance block (its own mvp and a seed so several
// flames don't move in lockstep).

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
	// lights[] follows in the real block; unread here, so left undeclared.
} gubo;

layout(binding = 0, set = 1) uniform FlameUniformBufferObject {
	mat4 mvpMat;
	float seed;
} fubo;

layout(location = 0) in vec3 inPosition;
// 0 at the flame's base (anchored to the torch head) to 1 at its tip.
layout(location = 1) in float inTier;
// Fixed per-vertex random in [0,1), baked in at mesh-build time (Flame.hpp).
// Offsets this vertex's own noise coordinate, so the crown's individual tips
// (and the body's own rings) move independently instead of the whole flame
// as one rigid unit.
layout(location = 2) in float inSwaySeed;

// flat: no interpolation, so each triangle (built from mesh vertices that
// never smoothly blend into each other) reads one tier/seed and shades as
// one hard-edged facet, the low-poly look, without duplicating a vertex.
layout(location = 0) flat out float tier;
layout(location = 1) flat out float swaySeed;
// NOT flat: Flame.frag needs this one to vary smoothly across the triangle
// so dFdx/dFdy give it a real derivative to build a face normal from (the
// same trick CookTorrance.frag uses for flatNormals -- see notes.md). Local
// space rather than world space: this is a fake, stylized "which way does
// this facet catch the light" shade, not real lighting, so there's no need
// to also pass a model matrix just to get world coordinates for it.
layout(location = 2) out vec3 fragLocalPos;

// Cheap 2D value noise: hash the four corners of the cell p falls in, blend
// with a smoothstep so there's no visible grid, no texture lookups needed.
float hash21(vec2 p) {
	p = fract(p * vec2(123.34, 456.21));
	p += dot(p, p + 45.32);
	return fract(p.x * p.y);
}

float noise2(vec2 p) {
	vec2 i = floor(p);
	vec2 f = fract(p);
	float a = hash21(i);
	float b = hash21(i + vec2(1.0, 0.0));
	float c = hash21(i + vec2(0.0, 1.0));
	float d = hash21(i + vec2(1.0, 1.0));
	vec2 u = f * f * (3.0 - 2.0 * f);
	return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Two octaves is enough detail at this triangle count -- a third would just
// add high-frequency wobble finer than the mesh itself can resolve.
float turbulence(vec2 p) {
	return noise2(p) * 0.65 + noise2(p * 2.3 + 11.0) * 0.35;
}

void main() {
	float t = inTier;
	// Per-vertex amplitude variance (0.7..1.3), not just a coordinate
	// offset: without this every vertex still moves through the same range,
	// just decorrelated in time, which alone still reads as mechanical --
	// some tongues need to move further than others, not just differently.
	float ampVar = 0.7 + 0.6 * inSwaySeed;

	// t alone (not t*t) pins only the very base (t=0, sitting in the torch
	// cup) motionless while still giving the BODY real motion: t*t made
	// anything below the crown all but rigid, which read as "only the top
	// spikes move". The extra 0.55*t term keeps the curve short of fully
	// linear, so the tip still moves hardest.
	float sway = t * (0.45 + 0.55 * t) * ampVar;
	vec3 p = inPosition;

	// Each vertex gets its own noise coordinate: height (t) along one axis,
	// a scrolling time coordinate along the other, offset per-vertex by
	// inSwaySeed so the whole flame isn't reading the same noise field at
	// the same point. Scrolling the SAME field for x and z (offset by a
	// large constant so they don't correlate) rather than two unrelated
	// fields is what keeps the motion coherent instead of jittery -- a real
	// flame's whole cross-section drifts together, it doesn't shimmer
	// independently per axis.
	float tScroll = gubo.time * 0.9 + fubo.seed * 4.0 + inSwaySeed * 7.0;
	float nx = turbulence(vec2(t * 3.2, tScroll)) - 0.5;
	float nz = turbulence(vec2(t * 3.2 + 41.0, tScroll)) - 0.5;
	p.x += nx * 0.30 * sway;
	p.z += nz * 0.30 * sway;

	// Flow: the noise coordinate along t runs BACKWARD against time (t*5.0
	// minus, not plus, gubo.time), so the turbulence pattern itself travels
	// up the flame instead of every ring just pulsing in place -- fuel
	// visibly rising through it, which is what actually reads as "liquid"
	// rather than a shape that merely sways and breathes as a whole.
	float flow = turbulence(vec2(t * 5.0 - gubo.time * 3.2, inSwaySeed * 13.0 + fubo.seed)) - 0.5;
	p.xz *= 1.0 + flow * 0.55 * sway;
	p.y += flow * 0.05 * sway;

	// Flicker: a width pulse from the same kind of noise (more felt near
	// the tip) plus a small vertical breathing.
	float flicker = 1.0 + (turbulence(vec2(inSwaySeed * 9.0, gubo.time * 3.5)) - 0.5) * 0.4 * t;
	p.xz *= flicker;

	gl_Position = fubo.mvpMat * vec4(p, 1.0);
	tier = t;
	swaySeed = inSwaySeed;
	fragLocalPos = p;
}
