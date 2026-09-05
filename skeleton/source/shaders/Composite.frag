// FRAGMENT SHADER: final stage of the bloom chain. Combines the full-res HDR
// scene with the blurred quarter-res bloom mask, exposes and tone-maps the
// result, and writes it to the swapchain.
//
// Where its inputs come from:
//   uv              from Post.vert, the shared full-screen-quad vertex shader.
//   post (set 0)    written once per frame by main.cpp: bloom strength,
//                   exposure and the cheat-menu debug flags.
//   srcTex          the scene's HDR offscreen target, same one BloomBright.frag
//                   read from -- full resolution, unblurred.
//   bloomTex        BloomBlur.frag's second (vertical) pass output: the
//                   quarter-res, twice-blurred bright-pass mask.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// LIGHT_DEBUG_NO_TONEMAP, shared with SceneLights.hpp and CookTorrance.frag,
// so the cheat menu's "no tonemap" view works identically on this pass too.
// Found via glslc -I.
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
} post;

layout(binding = 1, set = 0) uniform sampler2D srcTex;
layout(binding = 2, set = 0) uniform sampler2D bloomTex;

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

// Same tone map as CookTorrance.frag, L09 s.45: divides by luminance rather
// than per-channel, which would desaturate highlights towards white. Kept
// identical between the two shaders so a tone-mapped bloom highlight and a
// tone-mapped direct-lit highlight compress the same way.
vec3 toneMap(vec3 c) {
	float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
	return c / (Y + 1.0);
}

void main() {
	// bloomTex is quarter-res; sampling it at the full-res uv relies on its
	// sampler's bilinear filtering to upsample smoothly back to full
	// resolution, which also happens to soften the blur's own quarter-res
	// blockiness rather than exposing it.
	//
	// The bloom fetch is clamped half a BLOOM texel inside the border. Even
	// with no offset added, the upsample's own bilinear footprint reaches
	// outside the image in the outermost half-texel -- and these attachments
	// carry a REPEAT sampler (Starter.hpp's default; see the note in
	// BloomBlur.frag), so out there it blends in the row from the OPPOSITE
	// edge. At quarter resolution that half texel is two full-res pixels: a
	// thin bright line along the top or bottom of the screen whenever
	// something overbright sits at the other end of the frame.
	vec2 bloomTexel = 1.0 / vec2(textureSize(bloomTex, 0));
	vec2 bloomUV = clamp(uv, bloomTexel * 0.5, 1.0 - bloomTexel * 0.5);

	vec3 scene = texture(srcTex, uv).rgb;
	vec3 bloom = texture(bloomTex, bloomUV).rgb;

	vec3 color = scene + bloom * post.bloomIntensity;
	color *= post.exposure;

	// Debug view: skip the tone map so anything above 1 clips instead of
	// being compressed back into range -- the same escape hatch
	// CookTorrance.frag offers, reused here via the same bit so one cheat
	// menu toggle affects direct lighting and bloom together.
	if((post.debugFlags & LIGHT_DEBUG_NO_TONEMAP) == 0) {
		color = toneMap(color);
	}

	// Vignette: darkens the corners/edges of the frame, screen-space and
	// resolution-independent (computed from uv, not pixel coordinates).
	// Pairs with CookTorrance.frag's distance fog for the same reason RE-era
	// survival horror leaned on both together -- fog hides the draw
	// distance AHEAD of the player, the vignette hides the SCREEN EDGES,
	// where the geometry visibility cull's cone (GEOM_CULL_CONE_COS in
	// main.cpp) is narrower than the actual camera frustum and so is at its
	// most likely to be caught mid-fade by a glance toward the corner of the
	// view. Also just reads as moody in a dungeon, independent of any of
	// that.
	//
	// Applied post-tonemap (like the whiteout below) since darkening the
	// corners is a display-referred vignette, the same as a real lens's,
	// not a physical light falloff CookTorrance.frag's fog already is.
	// Applied BEFORE the whiteout mix, not after: escapeFlash washing the
	// whole frame to white should read as an even, blinding flash, not a
	// flash with a dark ring still sitting around it.
	{
		// Distance from centre in uv space: 0 at the middle, ~0.707 at a
		// corner. Not aspect-corrected on purpose -- a plain uv-space circle
		// comes out as an ellipse hugging the frame's actual proportions
		// (wider on a wide window), which is what a vignette should do
		// anyway, rather than a true circle that would clip harder on the
		// short (vertical) axis of a widescreen window.
		float vignetteDist = length(uv - vec2(0.5));
		// 0 within VIGNETTE_INNER of centre, ramping to 1 by VIGNETTE_OUTER
		// (and staying 1 past it, smoothstep's usual clamp).
		const float VIGNETTE_INNER = 0.35;
		const float VIGNETTE_OUTER = 0.75;
		float vignette = smoothstep(VIGNETTE_INNER, VIGNETTE_OUTER, vignetteDist);
		// How dark the corners get at full vignette: 1.0 would crush them to
		// black, which read as a hole cut in the screen rather than a haze.
		const float VIGNETTE_STRENGTH = 0.6;
		color *= (1.0 - vignette * VIGNETTE_STRENGTH);
	}

	// The escape whiteout, applied AFTER the tone map on purpose. Before it,
	// this would just be more exposure, and the tone map's own curve --
	// c / (Y + 1), which approaches white without ever arriving -- would eat
	// it: the frame would go pale and stop, with the bright parts still
	// clearly brighter. After the curve there is nothing left to compress it,
	// so escapeFlash = 1 is genuinely white.
	//
	// It rides on top of the exposure/bloom ramp main.cpp applies at the same
	// time rather than replacing it, and the order matters for how it reads:
	// the scene blows out first, blooming and losing its detail, and only then
	// does the last of it wash away. Doing this alone would be a fade to
	// white, which is a screen transition and not being blinded.
	color = mix(color, vec3(1.0), clamp(post.escapeFlash, 0.0, 1.0));

	// Written linear, not gamma-encoded: the swapchain is B8G8R8A8_SRGB, so
	// the hardware does the linear-to-sRGB encode on write. A manual curve
	// here would double-encode and wash the image out.
	outColor = vec4(color, 1.0);
}
