// FRAGMENT SHADER for the torch flames. Paired with Flame.vert.
//
// A flame is not a lit surface, so there is no BRDF here and nothing from
// materials.json is read: it emits light instead of reflecting it, and its
// colour is a function of how hot that part of the flame is, not of what lamp
// is pointing at it. Nothing in this file samples the albedo texture either --
// the whole appearance is generated from fbm() noise, because the alternative
// (a painted texture on a static mesh) can only ever look like a painted
// static mesh.
//
// The other half of the job is ALPHA. Fire is a translucent volume you see
// through, and an opaque mesh reads as orange plastic no matter how it is
// coloured. The pipeline this shader runs on enables blending (PFlame in
// main.cpp calls setTransparency(true)), so the alpha written here is what
// makes the edges dissolve into the room behind them.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "custom/Noise.glsl"
#include "custom/LightConstants.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in float fragHeight;

layout(location = 0) out vec4 outColor;

// Declared to match the block the scene pipeline uses, field for field: both
// pipelines share DSLlocal and DSLglobal, so the layouts must agree even
// though this shader only reads `time` out of the first and `eyePos` out of
// the second.
layout(binding = 0, set = 1) uniform UniformBufferObject {
	mat4 mvpMat;
	mat4 mMat;
	mat4 nMat;
	vec3 mS;
	float roughness;
	float F0;
	float k;
	int flatNormals;
	float time;
} ubo;

struct Light {
	vec3 pos;
	float g;
	vec3 dir;
	float beta;
	vec3 color;
	float cosIn;
	float cosOut;
	int type;
	// Unused here (the flame has no lighting loop), declared only because
	// this struct sits inside gubo, and both pipelines bind the exact same
	// GlobalUniformBufferObject at set 0 -- see CookTorrance.frag, which
	// actually reads it.
	int shadowIndex;
};

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	vec3 ambientUpper;
	vec3 ambientLower;
	vec3 ambientDir;
	int debugFlags;
	Light lights[MAX_LIGHTS];
} gubo;

// Body-heat ramp: dark red at the cool edges, through orange and yellow, to a
// near-white core. Driven by an intensity value rather than sampled from a
// gradient texture, so it cannot smear when the mesh moves underneath it and
// needs no UV unwrap to be correct.
vec3 fireColor(float t) {
	vec3 c = mix(vec3(0.35, 0.02, 0.0), vec3(0.9, 0.15, 0.0), smoothstep(0.0, 0.35, t));
	c = mix(c, vec3(1.0, 0.5, 0.05), smoothstep(0.35, 0.62, t));
	c = mix(c, vec3(1.0, 0.85, 0.35), smoothstep(0.62, 0.85, t));
	c = mix(c, vec3(1.0, 0.98, 0.85), smoothstep(0.85, 1.0, t));
	return c;
}

void main() {
	float h = clamp(fragHeight, 0.0, 1.0);

	// Noise drifting slowly upward. Kept slow on purpose: the flame's actual
	// movement is the vertex displacement in Flame.vert, and a fast scroll
	// here would fight it, reading as a texture sliding over a still shape.
	float n = fbm(vec2(fragUV.x * 4.0, fragUV.y * 4.0 - ubo.time * 0.45));

	// Heat: hottest at the base where the fuel is, cooling toward the tip,
	// with the noise breaking that gradient up so it is not a clean band.
	// The falloff is gentle (0.45, not 1.0) because the sculpt already tapers
	// to a point: cooling as sharply as a real flame does would land the whole
	// tapered part down at the black end of fireColor and simply erase the tip
	// the mesh went to the trouble of having.
	float heat = (1.0 - h * 0.45) * (0.65 + 0.7 * n);
	// Global pulse, the candle beat. Spatially uniform, so it brightens and
	// dims the whole flame together the way a real one does when it gutters.
	heat *= 0.88 + 0.12 * sin(ubo.time * 7.3 + n * 3.0);
	heat = clamp(heat, 0.0, 1.0);

	// Soft silhouette. A convex volume is thickest where you look straight
	// into it and thins to nothing at the edges, so opacity follows how
	// square-on the surface is to the eye. Without this the mesh keeps a hard
	// outline and still looks solid however it is coloured.
	//
	// Floored at 0.5 rather than reaching 0: the tip is a thin cone, so nearly
	// all of it is grazing-angle surface, and an unfloored term would fade out
	// exactly the part of the flame that should be brightest.
	vec3 N = normalize(fragNorm);
	vec3 V = normalize(gubo.eyePos - fragPos);
	float facing = abs(dot(N, V));
	float edge = mix(0.5, 1.0, smoothstep(0.0, 0.5, facing));

	// Softens toward the tip so the top dissolves into licking tongues instead
	// of ending on a hard line. The smoothstep deliberately ends past h = 1,
	// so even the topmost vertex keeps some opacity -- ending it AT 1 is what
	// made the tip vanish entirely.
	float alpha = heat * edge * (1.0 - smoothstep(0.85, 1.35, h));
	alpha = clamp(alpha * 1.6, 0.0, 1.0);

	// Written straight, with no tone map: this is a light source, and the
	// swapchain does the linear-to-sRGB encode on write as it does everywhere
	// else. The colour is not premultiplied -- the pipeline blends with
	// SRC_ALPHA / ONE_MINUS_SRC_ALPHA (Starter.hpp), which expects it straight.
	outColor = vec4(fireColor(heat), alpha);
}
