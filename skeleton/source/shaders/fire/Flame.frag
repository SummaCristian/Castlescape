// FRAGMENT SHADER for the flame body (custom/Flame.hpp, Flame.vert). The mesh
// is flat billboard cards; the tapered shape, the licking structure, the soft
// fringe, the detaching wisps all come from a procedural fire field evaluated
// per pixel and discarded down to its own silhouette. Unlit: fire emits.
// Writes real HDR values for the hot core and lets the bloom pass do the rest.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// rgb2hsv/hsv2rgb/recolorStop, shared with Spark.frag so the flame body and
// its sparks recolor onto `color` the same way.
#include "custom/FlameColor.glsl"

// Only gubo.time is read, but every field up to it must be declared in order
// (std140 offsets are positional); the trailing lights[] is dropped.
layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
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

// Cheap 2D value noise: hash the cell's four corners, blend with a smoothstep
// so no grid shows. No texture lookups.
float hash21(vec2 p) {
	p = fract(p * vec2(123.34, 456.21));
	p += dot(p, p + 45.32);
	return fract(p.x * p.y);
}

float noise2(vec2 p) {
	vec2 cell = floor(p);
	vec2 cellFrac = fract(p);
	float cornerBL = hash21(cell);
	float cornerBR = hash21(cell + vec2(1.0, 0.0));
	float cornerTL = hash21(cell + vec2(0.0, 1.0));
	float cornerTR = hash21(cell + vec2(1.0, 1.0));
	vec2 smoothFrac = cellFrac * cellFrac * (3.0 - 2.0 * cellFrac);
	return mix(mix(cornerBL, cornerBR, smoothFrac.x), mix(cornerTL, cornerTR, smoothFrac.x), smoothFrac.y);
}

// 4 octaves, each double frequency and half amplitude: detail at many scales
// at once (coarse tongues AND fine edges). 2.02, not 2.0, so successive
// octaves never land on the same grid and show a repeating overlay.
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

// 2-octave variant for the spine and bubble fields: both only need
// low-frequency structure, so fbm's fine octaves would buy nothing there.
float noiseLo(vec2 p) {
	return noise2(p) * 0.65 + noise2(p * 2.13 + 7.31) * 0.35;
}

void main() {
	float y = uv.y;

	// Two time bases. ft drives the fast per-pixel shimmer; fm drives the bulk
	// movement (spine sway, advection, warp morph, bubbles), so the flame
	// travels without the flicker turning frantic.
	//
	// fm's rate swells and eases over ~5-15 s, so the flame surges on a draft
	// and settles. It's the exact integral of R0 * (1 + A1*sin(W1 t) +
	// A2*sin(W2 t)), which keeps the phase continuous -- scaling time by a
	// moving factor directly would stutter. A1 + A2 < 1, so the flame never
	// flows backward. seed offsets each torch.
	float ft = gubo.time * 1.3;
	const float R0 = 2.2;
	const float A1 = 0.30, W1 = 0.55;
	const float A2 = 0.15, W2 = 1.30;
	float st = gubo.time + seed * 6.0;
	float fm = R0 * (st - (A1 / W1) * cos(W1 * st) - (A2 / W2) * cos(W2 * st));

	// WANDERING SPINE. The centreline sways: x is shifted by slow noise
	// advected down the axis, zero at the wick, strongest at the tip (y*y,
	// like the lean's h*h). Everything downstream works in this spine-relative
	// x, so the WHOLE field curls, silhouette and structure together, instead
	// of the texture sliding inside a static cutout -- the biggest difference
	// between "pulsing in place" and burning. Worst-case |offset| stays inside
	// the card.
	vec2 spineP = vec2(seed * 7.0 + layer * 1.7, y * 1.6 - fm * 0.9);
	float sway = (noiseLo(spineP) - 0.5) * 2.0;	// ~[-1,1], low frequency
	float x = uv.x - sway * 0.35 * y * y;

	// Flame profile mask: half-width w(y), necked at the wick, widest just
	// above it, tapering to a point. wFall is a POWER curve, not a smoothstep:
	// a linear taper gives straight sides and reads as a diamond, while
	// pow(1-y, 0.65) bows the sides outward low down and draws them to a
	// point. Floored just above 0 so the division below can't blow up.
	float wRise = smoothstep(0.0, 0.16, y);
	float wFall = pow(max(1.0 - y, 0.0), 0.65);
	float w = max(mix(0.30, 1.0, wRise) * wFall, 0.015);

	// Parabolic across the width: keeps the core full and drops off faster at
	// the edge, so the body reads as rounded, not a wedge with a bright crease.
	float r = abs(x) / w;
	float shape = clamp(1.0 - r * r, 0.0, 1.0);

	// Extra fade at the wick and the tip: without it the base has a hard seam
	// against the torch head and the tip shows a flat-topped sliver.
	float baseFade = smoothstep(0.0, 0.06, y);
	float tipFade = 1.0 - smoothstep(0.90, 1.0, y);
	shape *= baseFade * tipFade;

	// Advect upward: each layer gets its own scroll speed and phase so the
	// three cards never sync into one flat sheet. Subtracting time from the
	// sample's y (not adding) makes the PATTERN travel up the card -- a fixed
	// screen point sees the noise that previously sat lower, i.e. fuel rising.
	float scrollSpeed = 0.55 + layer * 0.18;
	float scrollPhase = seed * 9.0 + layer * 3.1;
	vec2 p = vec2(x * 2.2, y * 3.4 - fm * scrollSpeed - scrollPhase);

	// Domain warp: offset the FBM lookup by a second, lower-frequency FBM
	// instead of sampling directly. Warping the SAMPLE POINT makes the bright
	// regions curl and drift sideways as they rise -- the difference between
	// licking tongues and a fizzy gradient. The warp coordinate has a lateral
	// time term so a tongue changes shape as it rises, and its bite grows with
	// height: near-laminar at the wick, fully turbulent by the tip.
	vec2 warpCoord = p * 0.4 + vec2(fm * 0.16, -fm * scrollSpeed * 0.5);
	vec2 warp = vec2(fbm(warpCoord), fbm(warpCoord + 19.3)) - 0.5;
	float heatNoise = fbm(p + warp * mix(0.85, 1.55, y));

	// How hard the noise may bite into the silhouette, by height: near the
	// wick it barely perturbs the dense base, near the tip it can swing the
	// field negative enough to tear pieces off -- how wisps detach. Centred on
	// 0.55, not 0.5, because a 4-octave fbm sum sits above the midpoint.
	float carve = mix(0.35, 1.30, y);
	float noiseTerm = clamp(1.0 + (heatNoise - 0.55) * 2.0 * carve, 0.0, 1.5);

	float heat = shape * noiseTerm;

	// BUBBLING: pockets of hot gas that form low, ride the advection up, and
	// die toward the tip. A thresholded low-frequency field, not more fbm: the
	// smoothstep turns noise into distinct blobs with gaps. Scrolls ~35%
	// faster than the body, so the pockets overtake the texture and read as
	// buoyant volumes. Shares `warp` so they curl with the same turbulence.
	// Added to heat, a pocket both brightens and bulges the silhouette.
	vec2 bp = vec2(x * 3.0, y * 2.2 - fm * scrollSpeed * 1.35 - scrollPhase * 1.3);
	float bubbles = smoothstep(0.58, 0.80, noiseLo(bp + warp * 0.6));
	heat += bubbles * shape * mix(0.55, 0.20, y);

	// Outer fringe genuinely translucent, not just dim: the window starts well
	// above 0 so the fringe is a gradient several pixels wide.
	float alpha = smoothstep(0.15, 0.50, heat);

	// depthWriteEnable is hardcoded on for every pipeline, transparent ones
	// included, so an undiscarded low-alpha fringe still writes depth and
	// punches a hole through the other flame layers and the sparks. The
	// discard is required for correctness, not an optimization.
	if(alpha < 0.04) {
		discard;
	}

	// Temperature ramp keyed on `heat`, not mesh height -- the whitest pixels
	// are wherever the noise puts the core, which drifts and licks upward.
	// The 4 stops are authored for orange fire; each is hue-rotated onto the
	// torch's `color` by the same delta, which keeps the dark -> hot-white
	// value/saturation progression that makes it read as fire, and reproduces
	// the default palette exactly when hueDelta == 0.
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

	// Push the hot end above 1.0 so the bloom pass has real energy to find;
	// the fringe stays near unit brightness so it keeps its red/orange.
	float hdrBoost = mix(1.0, 6.0, smoothstep(0.3, 1.0, heat));
	color *= hdrBoost;

	// SHIMMER: a fast per-pixel flicker that varies ALONG the flame, so
	// different heights twitch at different moments and it reads as
	// combustion. The CPU `intensity` envelope only carries the slower
	// breathing, since the point light rides that same signal.
	float shimmer = 0.88 + 0.24 * noise2(vec2(y * 2.0 + seed * 31.0 + layer,
	                                          ft * 7.0));

	// intensity drives the torch's point light too, so flame and light dim
	// together; glare is the stare-at boost (1.0 unless looked at dead-on),
	// overdriving the HDR output so bloom flares when the player stares in.
	color *= intensity * shimmer * glare;

	outColor = vec4(color, alpha);
}
