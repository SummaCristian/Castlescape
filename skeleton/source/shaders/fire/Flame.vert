// VERTEX SHADER for the flame body (custom/Flame.hpp): camera-facing billboard
// quads. The silhouette is punched per pixel in Flame.frag, which is what lets
// the flame fray and shed wisps; this shader just places flat cards in the
// flame's local space and lets the envelopes bend them.
//
// set 0 is the SAME global uniform the main pass uses, bound whole so eyePos
// etc. stay in step without a copy. set 1 is one per-torch block: a CPU-built
// billboard mvp, a seed so torches don't flicker in lockstep, the brightness
// and height envelopes (separate, see below), and a lean from hand motion.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	int debugFlags;
	float time;
	// lights[] follows in the real block; unread here, so left undeclared.
} gubo;

layout(binding = 0, set = 1) uniform FlameUniformBufferObject {
	mat4  mvpMat;      // billboard basis * ViewPrj, built CPU-side per frame
	float seed;        // per-torch phase offset so torches don't move in lockstep
	float intensity;   // BRIGHTNESS envelope, ~0.30..1.40, spring-smoothed; not height
	vec2  lean;        // billboard-LOCAL lean from hand motion (x right, y forward)
	float heightScale; // HEIGHT envelope, ~0.78..1.09; lags intensity (FLAME_HEIGHT_TAU)
	float glareBoost;  // 1.0 + stare-at emphasis, for Flame.frag
	vec3  color;       // target hue; Flame.frag hue-rotates the fire gradient onto it
} fubo;

// Quad corner in local units, not yet placed: x is the flame's width, y runs
// 0 (wick) to 1 (natural full-height tip) before heightScale guts it.
layout(location = 0) in vec2 inCorner;
// Which of the three depth-offset cards: 0 (far) to 2 (near). A float so it
// flats out like every other per-instance value here.
layout(location = 1) in float inLayer;

// x in [-1,1] across the card, y in [0,1] up it -- the normalized space
// Flame.frag's shape function works in.
layout(location = 0) out vec2 uv;
layout(location = 1) flat out float layer;
layout(location = 2) flat out float intensity;
layout(location = 3) flat out float seed;
layout(location = 4) flat out float glare;
layout(location = 5) flat out vec3 color;

void main() {
	// h=0 at the wick, h=1 at the tip. Separate from the scaled height below
	// because the lean and the fragment shape mask both want the RAW fraction.
	float h = inCorner.y;

	// The three cards get a little size variance so they don't perfectly
	// overlap and read as one flat card. Far layer slightly larger, near one
	// smaller -- for a convex flame that's the direction that looks like
	// thickness.
	float layerT = inLayer / 2.0;               // 0, 0.5, 1
	float widthScale       = mix(1.12, 0.88, layerT);
	float layerHeightScale = mix(1.06, 0.94, layerT);

	vec3 pos;
	pos.x = inCorner.x * widthScale;
	// heightScale scales the WHOLE height, not just colour/alpha: a guttering
	// flame is visibly shorter, not just dimmer.
	pos.y = h * fubo.heightScale * layerHeightScale;
	// Depth-separate the cards along local z so they aren't coplanar; 0.10 is
	// enough for parallax around the torch.
	pos.z = (inLayer - 1.0) * 0.10;

	// Lean pinned at the base, hardest at the tip: h*h keeps the lower flame
	// upright and concentrates the sway at the loose end.
	pos.xz += fubo.lean * h * h;

	gl_Position = fubo.mvpMat * vec4(pos, 1.0);

	uv = vec2(inCorner.x, h);

	layer = inLayer;
	intensity = fubo.intensity;
	seed = fubo.seed;
	glare = fubo.glareBoost;
	color = fubo.color;
}
