// FRAGMENT SHADER: final stage of the bloom chain. Combines the full-res HDR
// scene with the blurred quarter-res bloom mask, exposes, tone-maps, grades,
// and writes to the swapchain.
//
//   uv          from Post.vert
//   post (set 0)  per frame: bloom strength, exposure, cheat flags, veil/flash ramps
//   srcTex       the scene's HDR offscreen target, full res, unblurred
//   bloomTex     BloomBlur.frag's vertical-pass output: quarter-res, twice blurred

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

// Tone map, L09 s.45: divides by luminance, not per-channel (which would
// desaturate highlights towards white).
vec3 toneMap(vec3 c) {
	float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
	return c / (Y + 1.0);
}

// Radial R/B channel split at the corner, in uv units, a few pixels wide.
const float CA_STRENGTH = 0.008;

void main() {
	// bloomTex is quarter-res; the bilinear sampler upsamples it back to full
	// res, which also softens the blur's own blockiness. Clamped half a bloom
	// texel inside the border: the upsample's bilinear footprint reaches
	// outside the image at the edge, and these attachments carry a REPEAT
	// sampler (see BloomBlur.frag), so it would blend in the opposite edge --
	// a thin bright line at the top or bottom whenever something overbright
	// sits at the other end of the frame.
	vec2 bloomTexel = 1.0 / vec2(textureSize(bloomTex, 0));
	vec2 bloomUV = clamp(uv, bloomTexel * 0.5, 1.0 - bloomTexel * 0.5);

	// Chromatic aberration: pull R and B along the radial direction, scaled by
	// distance-from-centre squared so only the edges smear. G keeps the true
	// sample, so the world still lines up with the crosshair and text.
	vec2 caDir = uv - vec2(0.5);
	float caAmt = dot(caDir, caDir) * CA_STRENGTH;
	vec3 scene = vec3(
		texture(srcTex, uv + caDir * caAmt).r,
		texture(srcTex, uv).g,
		texture(srcTex, uv - caDir * caAmt).b);
	vec3 bloom = texture(bloomTex, bloomUV).rgb;

	vec3 color = scene + bloom * post.bloomIntensity;
	color *= post.exposure;

	// Debug view: skip the tone map so anything above 1 clips instead of
	// compressing. Same bit as CookTorrance.frag, so one toggle covers both.
	if((post.debugFlags & LIGHT_DEBUG_NO_TONEMAP) == 0) {
		color = toneMap(color);
	}

	// Split-tone grade: shadows a touch cool, highlights a touch warm, so
	// torchlight reads hotter against the stone without changing the lighting
	// itself. Weighted by luminance and kept gentle.
	{
		float Yc = dot(color, vec3(0.2126, 0.7152, 0.0722));
		const vec3  GRADE_SHADOW = vec3(0.96, 1.00, 1.06);
		const vec3  GRADE_HIGH   = vec3(1.06, 1.01, 0.92);
		const float GRADE_AMOUNT = 0.5;
		vec3 grade = mix(GRADE_SHADOW, GRADE_HIGH, smoothstep(0.0, 0.6, Yc));
		color *= mix(vec3(1.0), grade, GRADE_AMOUNT);
	}

	// Vignette: darkens the corners, screen-space, resolution-independent.
	// Pairs with the distance fog -- fog hides the draw distance ahead, the
	// vignette hides the screen edges, where the cull cone is narrowest and
	// most likely caught mid-fade by a glance to the corner. Reads as moody
	// too. Post-tonemap (a display-referred vignette, like a real lens), and
	// before the whiteout so escapeFlash washes evenly with no dark ring left.
	{
		// uv-space distance from centre: 0 middle, ~0.707 corner. Not
		// aspect-corrected on purpose -- the resulting ellipse hugs the
		// frame's proportions, which is what a vignette should do.
		float vignetteDist = length(uv - vec2(0.5));
		const float VIGNETTE_INNER = 0.35;
		const float VIGNETTE_OUTER = 0.75;
		float vignette = smoothstep(VIGNETTE_INNER, VIGNETTE_OUTER, vignetteDist);
		// 1.0 would crush the corners to black, reading as a hole in the screen.
		const float VIGNETTE_STRENGTH = 0.6;
		color *= (1.0 - vignette * VIGNETTE_STRENGTH);
	}

	// The escape whiteout, AFTER the tone map: before it, the curve
	// (c / (Y + 1), which never quite reaches white) would eat it. It rides on
	// top of main.cpp's exposure/bloom ramp, so the scene blows out and blooms
	// first and only then washes away -- being blinded, not a fade to white.

	// THE SPECTRAL VEIL: what standing inside a ghost looks like once
	// spectralFade() has taken the mesh away. Ramped over the same range the
	// mesh dissolves on, so the presence moves onto the frame. After the tone
	// map: the room seen through something, not more light in it.
	if(post.spectralVeil > 0.0) {
		// Desaturate first, then tint. Blue straight over the torchlight
		// leaves the flames orange and reads as a bad filter; taking the
		// colour out first is what makes the room go cold.
		const vec3  VEIL_TINT = vec3(0.62, 0.86, 1.10);
		// A floor under the blacks -- nothing is fully dark seen through
		// something translucent. Small: the darkness is the atmosphere.
		const vec3  VEIL_LIFT = vec3(0.014, 0.030, 0.050);
		// Strength centre vs edge. Weighted outwards it reads as something
		// wrapped around the player; the middle stays walkable. Lower
		// VEIL_CENTRE if players lose their bearings.
		const float VEIL_CENTRE = 0.42;
		const float VEIL_EDGE   = 0.92;

		float veil = clamp(post.spectralVeil, 0.0, 1.0);
		// Squared radius, normalised so an edge midpoint is 1 (the falloff
		// wanted is quadratic anyway).
		vec2  d = uv - 0.5;
		float r = clamp(dot(d, d) * 4.0, 0.0, 1.0);
		float w = veil * mix(VEIL_CENTRE, VEIL_EDGE, r);

		float Y = dot(color, vec3(0.2126, 0.7152, 0.0722));
		color = mix(color, vec3(Y) * VEIL_TINT + VEIL_LIFT * veil, w);
	}

	color = mix(color, vec3(1.0), clamp(post.escapeFlash, 0.0, 1.0));

	// Linear, not gamma-encoded: the swapchain is B8G8R8A8_SRGB, so the
	// hardware does the linear-to-sRGB encode on write. A manual curve here
	// would double-encode and wash the image out.
	outColor = vec4(color, 1.0);
}
