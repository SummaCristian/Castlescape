// FRAGMENT SHADER: final stage of the bloom chain. Combines full-res HDR
// scene with blurred quarter-res bloom mask, exposes, tone-maps, grades,
// writes to swapchain.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// LIGHT_DEBUG_NO_TONEMAP, shared so one cheat toggle covers this pass too.
#include "custom/LightConstants.glsl"

layout(binding = 0, set = 0) uniform PostUniformBufferObject {
	vec2  texelSize;
	vec2  blurDir;
	float threshold;
	float knee;
	float bloomIntensity;
	float exposure;
	int   debugFlags;
	float time;
	float escapeFlash;
	float spectralVeil;
} post;

layout(binding = 1, set = 0) uniform sampler2D srcTex;
layout(binding = 2, set = 0) uniform sampler2D bloomTex;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

// Tone map, L09 s.45: divide by luminance, not per-channel (avoids
// desaturating highlights towards white).
vec3 toneMap(vec3 c) {
	float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
	return c / (Y + 1.0);
}

// Radial R/B channel split at the corner, in uv units, a few pixels wide.
const float CA_STRENGTH = 0.008;

void main() {
	// bloomTex is quarter-res; also softens blur blockiness.
	// Clamped half a texel in: REPEAT sampler would else blend the opposite edge.
	vec2 bloomTexel = 1.0 / vec2(textureSize(bloomTex, 0));
	vec2 bloomUV = clamp(uv, bloomTexel * 0.5, 1.0 - bloomTexel * 0.5);

	// Chromatic aberration: pull R/B along radial dir, scaled by dist^2 so
	// only edges smear. G keeps the true sample so crosshair/text stay aligned.
	vec2 caDir = uv - vec2(0.5);
	float caAmt = dot(caDir, caDir) * CA_STRENGTH;
	vec3 scene = vec3(
		texture(srcTex, uv + caDir * caAmt).r,
		texture(srcTex, uv).g,
		texture(srcTex, uv - caDir * caAmt).b);
	vec3 bloom = texture(bloomTex, bloomUV).rgb;

	vec3 color = scene + bloom * post.bloomIntensity;
	color *= post.exposure;

	// Debug view: skip tone map so values above 1 clip instead of compressing.
	if((post.debugFlags & LIGHT_DEBUG_NO_TONEMAP) == 0) {
		color = toneMap(color);
	}

	// Split-tone grade: shadows cool, highlights warm, so torchlight reads
	// hotter without changing the actual lighting. Gentle.
	{
		float gradeLuma = dot(color, vec3(0.2126, 0.7152, 0.0722));
		const vec3  GRADE_SHADOW = vec3(0.96, 1.00, 1.06);
		const vec3  GRADE_HIGH   = vec3(1.06, 1.01, 0.92);
		const float GRADE_AMOUNT = 0.5;
		vec3 grade = mix(GRADE_SHADOW, GRADE_HIGH, smoothstep(0.0, 0.6, gradeLuma));
		color *= mix(vec3(1.0), grade, GRADE_AMOUNT);
	}

	// Vignette: darkens corners, screen-space.
	// Pairs with distance fog (fog hides draw distance ahead, vignette hides
	// screen edges where cull cone is narrowest). Post-tonemap, before whiteout.
	{
		// uv-space distance from centre.
		float vignetteDist = length(uv - vec2(0.5));
		const float VIGNETTE_INNER = 0.35;
		const float VIGNETTE_OUTER = 0.75;
		float vignette = smoothstep(VIGNETTE_INNER, VIGNETTE_OUTER, vignetteDist);
		// 1.0 would crush corners to black (reads as a hole in the screen).
		const float VIGNETTE_STRENGTH = 0.6;
		color *= (1.0 - vignette * VIGNETTE_STRENGTH);
	}

	// Spectral veil: what standing inside a ghost looks like once
	// spectralFade() dissolves the mesh. Ramped over the same dissolve range.
	// After tone map: the room seen through something, not more light in it.
	if(post.spectralVeil > 0.0) {
		// Desaturate first, then tint -- blue straight over torchlight would
		// leave flames orange and read as a bad filter.
		const vec3  VEIL_TINT = vec3(0.62, 0.86, 1.10);
		// Avoid seeing pitch black over veil (ADDED, not multiplied).
		const vec3  VEIL_LIFT = vec3(0.014, 0.030, 0.050);
		// Strength centre vs edge; weighted outwards so the middle stays
		// walkable.
		const float VEIL_CENTRE = 0.42;
		const float VEIL_EDGE   = 0.92;

		float veil = clamp(post.spectralVeil, 0.0, 1.0);
		// Squared radius, normalised so an edge midpoint is 1.
		vec2  veilOffset = uv - 0.5;
		float veilRadiusSq = clamp(dot(veilOffset, veilOffset) * 4.0, 0.0, 1.0);
		float veilWeight = veil * mix(VEIL_CENTRE, VEIL_EDGE, veilRadiusSq);
		// Apply color (blue ghost)
		float sceneLuma = dot(color, vec3(0.2126, 0.7152, 0.0722));
		color = mix(color, vec3(sceneLuma) * VEIL_TINT + VEIL_LIFT * veil, veilWeight);
	}


	// Escape whiteout, after tone map (else the curve never quite reaches
	// white). Rides on main.cpp's exposure/bloom ramp: blow out and bloom
	// first, then wash away.
	color = mix(color, vec3(1.0), clamp(post.escapeFlash, 0.0, 1.0));

	// Linear, not gamma-encoded: swapchain is B8G8R8A8_SRGB, hardware does
	// the encode on write. A manual curve here would double-encode.
	outColor = vec4(color, 1.0);
}
