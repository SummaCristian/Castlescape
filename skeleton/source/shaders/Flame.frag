// FRAGMENT SHADER for the low-poly flame (see custom/Flame.hpp). Unlit and
// opaque on purpose: a flame emits light, it doesn't reflect the scene's, so
// running it through the BRDF would need a light-facing normal a wobbling
// faceted mesh doesn't really have. Colour is a plain height gradient instead.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// Only gubo.time is read here, but the block must be declared with every
// field UP TO it in the same order as main.cpp's real struct: std140 offsets
// are purely positional, so a shader can stop declaring early (the trailing
// lights[] is never read here) but can't skip or reorder anything before
// what it does read.
layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	vec3 ambientUpper;
	vec3 ambientLower;
	vec3 ambientDir;
	int debugFlags;
	float time;
} gubo;

// flat, matching Flame.vert: one tier/seed value per triangle, so both the
// gradient and the flicker step facet-to-facet instead of smoothly
// blending, which is the point of a low-poly look.
layout(location = 0) flat in float tier;
layout(location = 1) flat in float swaySeed;
// NOT flat -- see Flame.vert. Read only through dFdx/dFdy below, never used
// as a position by itself.
layout(location = 2) in vec3 fragLocalPos;

layout(location = 0) out vec4 outColor;

void main() {
	// The look this is chasing: cut-glass/gem faceting, where each flat
	// triangle catches a fake light differently depending on which way it
	// happens to face, instead of a smooth height gradient with no
	// per-facet variation at all. Exactly CookTorrance.frag's flatNormals
	// trick (notes.md): the derivatives of an interpolated position across a
	// triangle are two vectors lying in its own plane, so their cross
	// product is a normal for that triangle, no vertex normals needed.
	vec3 faceN = normalize(cross(dFdx(fragLocalPos), dFdy(fragLocalPos)));
	// No vertex normal to compare against for a consistent outward flip
	// (unlike CookTorrance.frag, this mesh doesn't carry one) -- pushed
	// against the local axis instead, positive where the facet already
	// faces away from the flame's own centerline.
	vec3 axisOut = normalize(vec3(fragLocalPos.x, 0.0, fragLocalPos.z) + 1e-4);
	faceN = dot(faceN, axisOut) < 0.0 ? -faceN : faceN;

	vec3 fakeLightDir = normalize(vec3(0.35, 1.0, 0.55));
	float facetLight = dot(faceN, fakeLightDir) * 0.5 + 0.5;	// -1..1 to 0..1
	// A wide range (0.5x..1.6x): this is what makes facets read as
	// individually catching or missing the light, the actual "faceted gem"
	// look, rather than a subtle variation nobody would notice.
	float facetShade = mix(0.5, 1.6, facetLight);

	// Four stops instead of two: a two-color lerp reads as a single smooth
	// gradient no matter how it's clamped, four gives the eye a near-white
	// hot core, a distinct yellow-orange body and a soot-red fringe instead
	// of one blur from pale to dark.
	vec3 c0 = vec3(1.0, 0.98, 0.85);	// near-white, hottest (the base)
	vec3 c1 = vec3(1.0, 0.78, 0.25);	// bright yellow-orange
	vec3 c2 = vec3(1.0, 0.42, 0.05);	// orange
	vec3 c3 = vec3(0.55, 0.07, 0.02);	// deep red, sooty (the tips)

	vec3 color;
	if(tier < 0.33) {
		color = mix(c0, c1, tier / 0.33);
	} else if(tier < 0.66) {
		color = mix(c1, c2, (tier - 0.33) / 0.33);
	} else {
		color = mix(c2, c3, (tier - 0.66) / 0.34);
	}

	// Per-facet brightness shimmer, out of phase per triangle the same way
	// Flame.vert's sway is (same swaySeed, same TAU spread): a flat color
	// with no variation at all reads as a painted surface rather than fire,
	// even while it's swaying.
	float flicker = 0.88 + 0.18 * sin(gubo.time * 9.0 + swaySeed * 6.2831853);

	// Overexposure: there's no HDR render target or bloom pass in this
	// renderer (CookTorrance.frag tone-maps everything down to [0,1] before
	// the swapchain's own sRGB write clips it there anyway), so the only way
	// to get a "too bright to look at" core is to write values that clip on
	// purpose. Strongest at the base (tier 0, where c0 is already
	// near-white) and fading to none by the tips, so the red fringe stays a
	// real color instead of also blowing out to white.
	float overexposure = 1.0 + 2.0 * (1.0 - tier) * (1.0 - tier);
	outColor = vec4(color * flicker * overexposure * facetShade, 1.0);
}
