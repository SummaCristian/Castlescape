// FRAGMENT SHADER: flame body (Flame.vert). Flat billboard cards; shape,
// licking structure, fringe and wisps all from a procedural fire field
// evaluated per pixel. Unlit, HDR, bloom picks up the hot core.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// rgb2hsv/hsv2rgb/recolorStop, shared with Spark.frag.
#include "custom/FlameColor.glsl"

// Only gubo.time is read; fields kept in order for std140, lights[] dropped.
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

// Cheap 2D value noise: hash cell corners, smoothstep blend, no grid artifacts.
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

// 4 octaves, double freq/half amp each: coarse tongues + fine edges.
// 2.02 not 2.0 avoids octaves landing on the same grid.
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

// 2-octave variant for spine/bubble fields: only low-frequency structure needed.
float noiseLo(vec2 p) {
	return noise2(p) * 0.65 + noise2(p * 2.13 + 7.31) * 0.35;
}

void main() {
	float y = uv.y;

	// ft: fast per-pixel shimmer. fm: bulk movement (spine sway, advection,
	// warp, bubbles). fm's rate swells/eases over ~5-15s; it's the exact
	// integral of R0*(1+A1*sin(W1 t)+A2*sin(W2 t)), keeping phase continuous.
	// A1+A2 < 1 so flame never flows backward. seed offsets each torch.
	float ft = gubo.time * 1.3;
	const float R0 = 2.2;
	const float A1 = 0.30, W1 = 0.55;
	const float A2 = 0.15, W2 = 1.30;
	float st = gubo.time + seed * 6.0;
	float fm = R0 * (st - (A1 / W1) * cos(W1 * st) - (A2 / W2) * cos(W2 * st));

	// Wandering spine: x shifted by slow noise, zero at wick, strongest at tip
	// (y*y). Downstream works in spine-relative x so silhouette+structure curl
	// together (not a static cutout). Worst-case offset stays inside the card.
	vec2 spineP = vec2(seed * 7.0 + layer * 1.7, y * 1.6 - fm * 0.9);
	float sway = (noiseLo(spineP) - 0.5) * 2.0;	// ~[-1,1], low frequency
	float x = uv.x - sway * 0.35 * y * y;

	// Flame profile mask: half-width w(y), necked at wick, tapering to a point.
	// Power curve (not smoothstep) bows sides outward instead of a diamond.
	float wRise = smoothstep(0.0, 0.16, y);
	float wFall = pow(max(1.0 - y, 0.0), 0.65);
	float w = max(mix(0.30, 1.0, wRise) * wFall, 0.015);

	// Parabolic across width: rounded core, sharper edge falloff.
	float r = abs(x) / w;
	float shape = clamp(1.0 - r * r, 0.0, 1.0);

	// Fade at wick and tip to avoid a hard base seam / flat-topped tip.
	float baseFade = smoothstep(0.0, 0.06, y);
	float tipFade = 1.0 - smoothstep(0.90, 1.0, y);
	shape *= baseFade * tipFade;

	// Advect upward; per-layer speed/phase keeps the 3 cards desynced.
	// Subtracting time from y makes the pattern travel up (fuel rising).
	float scrollSpeed = 0.55 + layer * 0.18;
	float scrollPhase = seed * 9.0 + layer * 3.1;
	vec2 p = vec2(x * 2.2, y * 3.4 - fm * scrollSpeed - scrollPhase);

	// Domain warp: offset the FBM sample point by a second, lower-freq FBM,
	// so bright regions curl/drift instead of a static gradient. Bite grows
	// with height: near-laminar at wick, turbulent by tip.
	vec2 warpCoord = p * 0.4 + vec2(fm * 0.16, -fm * scrollSpeed * 0.5);
	vec2 warp = vec2(fbm(warpCoord), fbm(warpCoord + 19.3)) - 0.5;
	float heatNoise = fbm(p + warp * mix(0.85, 1.55, y));

	// Noise's bite into the silhouette, by height: barely perturbs near wick,
	// can go negative near tip (wisps detach). Centred 0.55 since 4-octave fbm
	// sits above 0.5.
	float carve = mix(0.35, 1.30, y);
	float noiseTerm = clamp(1.0 + (heatNoise - 0.55) * 2.0 * carve, 0.0, 1.5);

	float heat = shape * noiseTerm;

	// Bubbling: thresholded low-freq field (not fbm) gives distinct blobs.
	// Scrolls faster than body so pockets read as buoyant volumes; shares
	// `warp` for matching turbulence. Brightens + bulges the silhouette.
	vec2 bp = vec2(x * 3.0, y * 2.2 - fm * scrollSpeed * 1.35 - scrollPhase * 1.3);
	float bubbles = smoothstep(0.58, 0.80, noiseLo(bp + warp * 0.6));
	heat += bubbles * shape * mix(0.55, 0.20, y);

	// Fringe alpha: window starts above 0 for a soft multi-pixel edge.
	float alpha = smoothstep(0.15, 0.50, heat);

	// depthWriteEnable is on for all pipelines; an undiscarded low-alpha
	// fringe would still punch a hole through other layers/sparks.
	if(alpha < 0.04) {
		discard;
	}

	// Temperature ramp keyed on `heat` (not mesh height), so hottest pixels
	// track the noise, not geometry. 4 stops authored for orange, hue-rotated
	// onto torch `color`; reproduces default palette when hueDelta == 0.
	const vec3 REF_ORANGE = vec3(1.00, 0.42, 0.05);
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

	// Push hot end above 1.0 for bloom; fringe stays near unit brightness.
	float hdrBoost = mix(1.0, 6.0, smoothstep(0.3, 1.0, heat));
	color *= hdrBoost;

	// Shimmer: fast per-pixel flicker varying along the flame height.
	float shimmer = 0.88 + 0.24 * noise2(vec2(y * 2.0 + seed * 31.0 + layer,
	                                          ft * 7.0));

	// intensity also drives the point light; glare is stare-at boost.
	color *= intensity * shimmer * glare;

	outColor = vec4(color, alpha);
}
