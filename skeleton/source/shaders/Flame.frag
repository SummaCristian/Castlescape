// FRAGMENT SHADER for the flame body (see custom/Flame.hpp and Flame.vert).
// This REPLACES the old low-poly fragment shader, which shaded a static
// faceted mesh with a fake per-facet "light" (a cross-product face normal
// against a made-up light direction) and a height-based color gradient. That
// approach's silhouette was whatever the mesh's triangles happened to be --
// hard-edged, closed, and opaque. Here the mesh is just three flat billboard
// cards (Flame.vert); everything that makes it read as fire -- the tapered
// flame shape, the licking internal structure, the soft translucent fringe,
// wisps that pinch off near the tip -- comes from a procedural fire field
// evaluated per pixel and discarded down to that field's own silhouette.
// Still unlit and still no fake lighting: fire emits, it doesn't reflect the
// scene's light, and there is no lit side or shadow side to compute.
//
// The renderer now has an HDR (RGBA16F) target and a real bloom pass
// downstream, so unlike the old mesh (which had to fake "too bright to look
// at" by clipping color at 1.0, see the removed `overexposure` hack) this
// shader writes real HDR values for the hot core and lets bloom do the rest.

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

layout(location = 0) in vec2 uv;
layout(location = 1) flat in float layer;
layout(location = 2) flat in float intensity;
layout(location = 3) flat in float seed;

layout(location = 0) out vec4 outColor;

// Cheap 2D value noise: hash the four corners of the cell p falls in, blend
// with a smoothstep so there's no visible grid, no texture lookups needed.
// Same construction as the old Flame.vert used for mesh sway -- kept here
// unchanged because it's cheap and has no visible periodicity at the scales
// this shader samples it at.
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

void main() {
	float y = uv.y;

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
	float r = abs(uv.x) / w;
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
	// the noise value that used to sit lower down -- i.e. fuel visibly
	// rising through the flame. Same trick the old Flame.vert used for its
	// mesh sway, just applied to a noise lookup instead of a vertex offset.
	float scrollSpeed = 0.55 + layer * 0.18;
	float scrollPhase = seed * 9.0 + layer * 3.1;
	vec2 p = vec2(uv.x * 2.2, y * 3.4 - gubo.time * scrollSpeed - scrollPhase);

	// Domain warp: offset the FBM lookup by a second, lower-frequency FBM
	// (itself slowly advected) instead of sampling the first FBM directly.
	// This is the actual difference between "licking tongues" and a fizzy
	// gradient -- without it, brightness just varies smoothly in place;
	// warping the SAMPLE POINT makes the bright regions themselves curl and
	// travel sideways as they rise, which is what a real flame's turbulent
	// tongues look like.
	vec2 warpCoord = p * 0.4 + vec2(0.0, -gubo.time * scrollSpeed * 0.5);
	vec2 warp = vec2(fbm(warpCoord), fbm(warpCoord + 19.3)) - 0.5;
	float heatNoise = fbm(p + warp * 1.15);

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

	// Outer fringe genuinely translucent, not just dim -- this is what a
	// billboard card can do that the old closed mesh never could. The window
	// starts well above 0 so that the fringe is a real gradient several pixels
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

	// Temperature ramp keyed on `heat` (the noise-carved fire field), NOT on
	// mesh height like the old shader's tier-based gradient -- the hottest,
	// whitest pixels are wherever the noise says the core currently is,
	// which drifts and licks upward instead of always sitting at a fixed
	// height on the card.
	vec3 cCold   = vec3(0.55, 0.06, 0.02);   // deep red, coolest visible edge
	vec3 cOrange = vec3(1.00, 0.42, 0.05);
	vec3 cYellow = vec3(1.00, 0.78, 0.25);
	vec3 cCore   = vec3(1.00, 0.97, 0.88);   // near-white, hottest

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

	// intensity is the same CPU-driven flicker/guttering envelope that
	// shortened the card in Flame.vert -- applying it to color too means a
	// guttering flame visibly dims as well as shrinks.
	color *= intensity;

	outColor = vec4(color, alpha);
}
