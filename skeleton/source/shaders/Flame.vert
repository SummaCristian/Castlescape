// VERTEX SHADER for the flame body (see custom/Flame.hpp): camera-facing
// billboard quads. A billboard punches its silhouette per pixel in the
// fragment shader (Flame.frag), rather than the mesh itself defining a fixed
// closed outline, which is what lets the flame fray at the edges and let
// wisps detach. All this shader does is place flat cards in the flame's own
// local space and let the envelopes bend them; every bit of the actual look
// lives in the fragment shader.
//
// set 0 is the SAME global uniform the main pass uses (DSglobal in main.cpp,
// extended with a "time" field); the flame doesn't need eyePos or the light
// array, but binding the very same descriptor set means it doesn't need its
// own copy of eyePos/lightCount/etc kept in step.
// set 1 is one small per-torch block: an mvp built CPU-side from the
// camera's own right/up/forward vectors (a billboard has to face the camera,
// it can't ride the torch's own orientation), plus a seed so several torches
// don't flicker in lockstep, the CPU-driven envelopes -- brightness and
// height SEPARATELY, see below -- and a lean the CPU derives from hand motion
// so the flame visibly responds to the torch being swung rather than just
// sitting there.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	vec3 ambientUpper;
	vec3 ambientLower;
	vec3 ambientDir;
	int debugFlags;
	float time;
	// lights[] follows in the real block; unread here, so left undeclared.
} gubo;

layout(binding = 0, set = 1) uniform FlameUniformBufferObject {
	mat4  mvpMat;      // billboard basis * ViewPrj, built CPU-side per frame
	float seed;        // per-torch phase offset so torches don't move in lockstep
	float intensity;   // CPU-driven BRIGHTNESS envelope, ~0.30 .. 1.40,
	                   // spring-smoothed; does not scale height (see below)
	vec2  lean;        // billboard-LOCAL lean from hand motion: x = along the
	                   // billboard's right axis, y = along its forward axis
	float heightScale; // slow HEIGHT envelope, ~0.78 .. 1.09, kept separate
	                   // from intensity: light output can flicker fast, but
	                   // height follows the fuel column and lags it, so the
	                   // CPU chases this one much more slowly (see main.cpp's
	                   // FLAME_HEIGHT_TAU).
	float glareBoost;  // 1.0 + stare-at emphasis, passed through to Flame.frag
} fubo;

// Quad corner in the billboard's own local units, NOT yet placed: x spans
// the flame's width, y spans 0 (wick) to 1 (the flame's natural full-height
// tip) before heightScale scales it down for guttering.
layout(location = 0) in vec2 inCorner;
// Which of the three depth-offset cards this vertex belongs to: 0 (farthest)
// to 2 (nearest). A float, not an int, purely so it interpolates/flat-outs
// the same way every other per-instance value here does.
layout(location = 1) in float inLayer;

// x in [-1,1] across the card, y in [0,1] up it -- the SAME normalized space
// Flame.frag's shape function works in, regardless of how this vertex shader
// actually scales/skews the card in world space below.
layout(location = 0) out vec2 uv;
// flat: one value per card, so the fragment shader can tell which of the
// three layers a pixel belongs to without a fourth attribute.
layout(location = 1) flat out float layer;
layout(location = 2) flat out float intensity;
layout(location = 3) flat out float seed;
layout(location = 4) flat out float glare;

void main() {
	// h=0 at the wick (pinned to the torch head), h=1 at the untouched tip.
	// Kept separate from the envelope-scaled height below because the lean
	// formula and the fragment shader's shape mask both want the RAW
	// fraction up the card, not the guttered one.
	float h = inCorner.y;

	// The three cards aren't identical rectangles stacked on the same spot:
	// a little size variance keeps them from perfectly overlapping, which
	// would read as one flat card face-on instead of a volume with depth.
	// The far layer (0) is drawn slightly larger, the near one (2) slightly
	// smaller -- for a flame that's mostly convex, that's the direction that
	// actually looks like thickness rather than a cutout stack.
	float layerT = inLayer / 2.0;               // 0, 0.5, 1
	float widthScale       = mix(1.12, 0.88, layerT);
	float layerHeightScale = mix(1.06, 0.94, layerT);

	vec3 pos;
	pos.x = inCorner.x * widthScale;
	// heightScale scales the WHOLE height, not just the color/alpha: a
	// guttering flame is visibly shorter, not just dimmer, exactly like a
	// real one starved of fuel or caught by a draft.
	pos.y = h * fubo.heightScale * layerHeightScale;
	// Depth-separate the three cards along local z so they aren't coplanar;
	// 0.10 units is small next to the flame's own ~1-unit half-width, just
	// enough for parallax as the camera moves around the torch.
	pos.z = (inLayer - 1.0) * 0.10;

	// Lean is pinned at the base and swings hardest at the tip -- h*h rather
	// than h keeps the lower two-thirds of the flame nearly upright and
	// concentrates the sway exactly where a real flame's is, at the loose
	// end away from the wick.
	pos.xz += fubo.lean * h * h;

	gl_Position = fubo.mvpMat * vec4(pos, 1.0);

	uv = vec2(inCorner.x, h);

	layer = inLayer;
	intensity = fubo.intensity;
	seed = fubo.seed;
	glare = fubo.glareBoost;
}
