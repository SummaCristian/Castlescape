// FRAGMENT SHADER for the flame body (see custom/Flame.hpp and Flame.vert).
// The mesh is just flat billboard cards (Flame.vert); everything that makes
// it read as fire -- the tapered flame shape, the licking internal
// structure, the soft translucent fringe, wisps that pinch off near the tip
// -- comes from a procedural fire field evaluated per pixel and discarded
// down to that field's own silhouette. Unlit: fire emits, it doesn't reflect
// the scene's light, so there is no lit side or shadow side to compute.
//
// The renderer has an HDR (RGBA16F) target and a real bloom pass downstream,
// so this shader writes real HDR values for the hot core and lets bloom do
// the rest.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// rgb2hsv/hsv2rgb/recolorStop, shared with Spark.frag so the flame body and
// its sparks recolor onto `color` the same way.
#include "custom/FlameColor.glsl"

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

layout(location = 0) in vec2 uv;
layout(location = 1) flat in float layer;
layout(location = 2) flat in float intensity;
layout(location = 3) flat in float seed;
layout(location = 4) flat in float glare;
layout(location = 5) flat in vec3 color;

layout(location = 0) out vec4 outColor;

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

// 4 octaves, each double frequency and half amplitude of the last -- the
// standard construction for detail at multiple scales at once (coarse
// tongues AND fine licking edges) instead of one single-frequency blob.
// The 2.02 (not a clean 2.0) keeps successive octaves from ever landing on
// exactly the same grid, which would show as a faint repeating overlay.
float fbm(vec2 p) {
	float sum = 0.0;
	float amp = 0.5;
	for(int i = 0; i < 4; i++) {
		sum += noise2(p) * amp;
		p = p * 2.02 + 11.0;
		amp *= 0.5;
	}
	return sum;
}

// 2-octave variant for the spine and bubble fields below: both only need
// low-frequency structure (a slow wander, fist-sized pockets), so paying for
// fbm's two fine octaves there would buy nothing visible.
float noiseLo(vec2 p) {
	return noise2(p) * 0.65 + noise2(p * 2.13 + 7.31) * 0.35;
}

void main() {
	float y = uv.y;

	// Two time bases, both scaled up from real seconds. ft drives the fast
	// per-pixel shimmer (the flame's flicker, `ft * 7.0` below); fm drives
	// the bulk movement -- spine sway, the rising advection, the warp morph,
	// the bubbles -- so the flame licks and travels without the flicker
	// turning frantic.
	//
	// fm's rate is not constant: it swells and eases over ~5-15 s, so the
	// flame surges on a draft and settles again. fm is the exact integral of
	//   R0 * (1 + A1*sin(W1 t) + A2*sin(W2 t))
	// which keeps the phase continuous -- scaling time by a moving factor
	// directly would make the whole field stutter as the factor changed.
	// A1 + A2 < 1 so the rate never crosses zero and the flame never flows
	// backward. seed offsets each torch so they don't all breathe in step.
	float ft = gubo.time * 1.3;
	const float R0 = 2.2;
	const float A1 = 0.30, W1 = 0.55;
	const float A2 = 0.15, W2 = 1.30;
	float st = gubo.time + seed * 6.0;
	float fm = R0 * (st - (A1 / W1) * cos(W1 * st) - (A2 / W2) * cos(W2 * st));

	// WANDERING SPINE. The centreline itself sways: the sample x is shifted
	// by slow noise advected down the flame axis, zero at the wick (a flame
	// is pinned to its fuel) and strongest at the tip (y*y, same reasoning
	// as the lean's h*h in Flame.vert). Everything downstream -- the profile
	// mask, the advection, the bubbles -- works in this spine-relative x, so
	// the WHOLE field curls, silhouette and internal structure together,
	// instead of the texture sliding around inside a static cutout. This is
	// the single biggest difference between "pulsing in place" and burning:
	// without it the flame's outline never travels at all.
	// Worst case |offset| at the tip is 0.35 and the tip half-width is well
	// under that from 1.0, so the swayed field always stays inside the card.
	vec2 spineP = vec2(seed * 7.0 + layer * 1.7, y * 1.6 - fm * 0.9);
	float sway = (noiseLo(spineP) - 0.5) * 2.0;	// ~[-1,1], low frequency
	float x = uv.x - sway * 0.35 * y * y;

	// Flame profile mask: half-width w(y), necked at the wick, widest just
	// above it, tapering to a point at the tip.
	//
	// wFall is a POWER curve rather than a smoothstep, and that is the whole
	// difference between a flame and a kite. A linear or smoothstep taper
	// gives straight sides meeting the widest point at an angle, which reads
	// as a diamond no matter what colour it is; pow(1-y, 0.65) falls slowly
	// at first and then increasingly fast, so the sides bow outward low down
	// and draw in to a point at the top. Floored just above 0, never 0, so
	// the division below can't blow up.
	float wRise = smoothstep(0.0, 0.16, y);
	float wFall = pow(max(1.0 - y, 0.0), 0.65);
	float w = max(mix(0.30, 1.0, wRise) * wFall, 0.015);

	// Parabolic rather than linear across the width: keeps the core full while
	// dropping off faster as it approaches the edge, so the body reads as
	// rounded instead of as a wedge with a bright crease down the middle.
	float r = abs(x) / w;
	float shape = clamp(1.0 - r * r, 0.0, 1.0);

	// Extra fade right at the wick and right at the very tip, on top of the
	// width falloff: without it the base has a visible hard seam where it
	// meets the torch head, and the tip -- where w is already tiny -- can
	// still show a flat-topped sliver instead of narrowing to nothing.
	float baseFade = smoothstep(0.0, 0.06, y);
	float tipFade = 1.0 - smoothstep(0.90, 1.0, y);
	shape *= baseFade * tipFade;

	// Advect upward: each layer gets its own scroll speed and phase (from
	// `layer` and `seed`) so the three cards never sync into one flat
	// pulsing sheet. Subtracting time from the sample's y coordinate (not
	// adding) is what makes the PATTERN travel up the card instead of
	// merely pulsing in place: a fixed screen point sees, as time advances,
	// the noise value that previously sat lower down -- i.e. fuel visibly
	// rising through the flame.
	float scrollSpeed = 0.55 + layer * 0.18;
	float scrollPhase = seed * 9.0 + layer * 3.1;
	vec2 p = vec2(x * 2.2, y * 3.4 - fm * scrollSpeed - scrollPhase);

	// Domain warp: offset the FBM lookup by a second, lower-frequency FBM
	// (itself slowly advected) instead of sampling the first FBM directly.
	// This is the actual difference between "licking tongues" and a fizzy
	// gradient -- without it, brightness just varies smoothly in place;
	// warping the SAMPLE POINT makes the bright regions themselves curl and
	// travel sideways as they rise, which is what a real flame's turbulent
	// tongues look like.
	// The warp coordinate has a LATERAL time term too, so the warp field
	// morphs as well as translates: a tongue changes shape as it rises
	// instead of the same frozen curl riding up the card unchanged. And the
	// warp's bite grows with height -- near-laminar at the wick, where a
	// real flame is a smooth cone, fully turbulent by the tip.
	vec2 warpCoord = p * 0.4 + vec2(fm * 0.16, -fm * scrollSpeed * 0.5);
	vec2 warp = vec2(fbm(warpCoord), fbm(warpCoord + 19.3)) - 0.5;
	float heatNoise = fbm(p + warp * mix(0.85, 1.55, y));

	// How hard the noise is allowed to bite into the silhouette, as a function
	// of height. This is what stops the mask above from reading as a solid
	// cutout shape: near the wick a flame is dense and steady, so the noise
	// barely perturbs it, but toward the tip it is thin and fully turbulent,
	// so up there the noise is allowed to swing the field far enough negative
	// to tear pieces off the silhouette entirely -- which is exactly how wisps
	// come to detach and float free of the body.
	//
	// Centred on 0.55 rather than 0.5 because a 4-octave fbm sum sits slightly
	// above the midpoint; centring on its true mean is what keeps the carving
	// symmetric instead of biased toward eroding everything.
	float carve = mix(0.35, 1.30, y);
	float noiseTerm = clamp(1.0 + (heatNoise - 0.55) * 2.0 * carve, 0.0, 1.5);

	float heat = shape * noiseTerm;

	// BUBBLING: pockets of extra-hot gas that form low in the flame, ride the
	// advection up, and die out toward the tip where `shape` pinches away.
	// A thresholded low-frequency field rather than more fbm detail: the
	// smoothstep turns noise into distinct blobs with in-between gaps, which
	// is what "bubbles" are -- fbm octaves alone only ever make the existing
	// texture busier. The field scrolls ~35% FASTER than the body field, so
	// the pockets visibly overtake the texture they ride through, reading as
	// buoyant volumes rather than painted-on brightness. Sharing `warp` (at
	// reduced strength) keeps them curling with the same turbulence as the
	// tongues around them. Added to heat, a pocket both brightens (the
	// temperature ramp below) and locally bulges the silhouette (the alpha
	// window reads heat too).
	vec2 bp = vec2(x * 3.0, y * 2.2 - fm * scrollSpeed * 1.35 - scrollPhase * 1.3);
	float bubbles = smoothstep(0.58, 0.80, noiseLo(bp + warp * 0.6));
	heat += bubbles * shape * mix(0.55, 0.20, y);

	// Outer fringe genuinely translucent, not just dim. The window starts
	// well above 0 so that the fringe is a real gradient several pixels
	// wide rather than saturating to opaque almost immediately.
	float alpha = smoothstep(0.15, 0.50, heat);

	// The framework hardcodes depthWriteEnable = VK_TRUE on every pipeline,
	// transparent ones included (Starter.hpp is not touched, see notes.md),
	// so a low-alpha fringe that isn't discarded still writes depth and
	// punches a silhouette-shaped hole through whatever's behind it -- the
	// other two flame layers and the sparks included. Discarding is not an
	// optimization here, it's required for correctness.
	if(alpha < 0.04) {
		discard;
	}

	// Temperature ramp keyed on `heat` (the noise-carved fire field), not on
	// mesh height -- the hottest, whitest pixels are wherever the noise says
	// the core currently is, which drifts and licks upward instead of
	// always sitting at a fixed height on the card.
	// The 4 stops below are authored for realistic orange fire. Every torch
	// picks a `color` (main.cpp's TorchFlame::color, default = the same
	// orange the point light casts): rather than replacing the stops
	// wholesale per color -- which either can't reproduce today's exact look
	// or needs fragile special-casing for the default -- each stop is
	// hue-rotated onto `color` by the same delta, which preserves the
	// value/saturation progression (dark -> hot-white) that makes the
	// gradient read as fire at all, and reproduces today's palette exactly
	// when `color` is the default orange (hueDelta == 0).
	const vec3 REF_ORANGE = vec3(1.00, 0.42, 0.05);	// today's cOrange stop
	vec3 targetHsv = rgb2hsv(color);
	float hueDelta = targetHsv.x - rgb2hsv(REF_ORANGE).x;
	float hueDist = min(abs(hueDelta), 1.0 - abs(hueDelta));

	vec3 cCold   = recolorStop(vec3(0.55, 0.06, 0.02), hueDelta, hueDist, targetHsv.y, 0.25);
	vec3 cOrange = recolorStop(vec3(1.00, 0.42, 0.05), hueDelta, hueDist, targetHsv.y, 0.0);
	vec3 cYellow = recolorStop(vec3(1.00, 0.78, 0.25), hueDelta, hueDist, targetHsv.y, 0.45);
	vec3 cCore   = recolorStop(vec3(1.00, 0.97, 0.88), hueDelta, hueDist, targetHsv.y, 0.60);

	vec3 color;
	if(heat < 0.35) {
		color = mix(cCold, cOrange, smoothstep(0.05, 0.35, heat));
	} else if(heat < 0.65) {
		color = mix(cOrange, cYellow, (heat - 0.35) / 0.30);
	} else {
		color = mix(cYellow, cCore, smoothstep(0.65, 1.0, heat));
	}

	// Push the hot end well above 1.0 so the bloom pass downstream has real
	// energy to find; the fringe stays near unit brightness so it doesn't
	// also blow out to white and lose the red/orange color entirely.
	float hdrBoost = mix(1.0, 6.0, smoothstep(0.3, 1.0, heat));
	color *= hdrBoost;

	// SHIMMER: a fast per-pixel flicker that varies ALONG the flame --
	// different heights twitch at different moments, which reads as
	// combustion rather than a brightness dial being wiggled. The CPU
	// envelope (`intensity`) only carries the slower breathing/guttering,
	// because the point light must ride that same signal (see main.cpp),
	// and a light can't flicker per-pixel anyway.
	float shimmer = 0.88 + 0.24 * noise2(vec2(y * 2.0 + seed * 31.0 + layer,
	                                          ft * 7.0));

	// intensity is the same CPU-driven envelope that drives the torch's own
	// point light, so flame and light dim together; glare is the stare-at
	// boost (1.0 unless this torch is being looked at dead-on), which
	// overdrives the HDR output so bloom flares exactly when the player
	// stares into the flame.
	color *= intensity * shimmer * glare;

	outColor = vec4(color, alpha);
}
