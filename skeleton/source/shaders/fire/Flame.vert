// VERTEX SHADER: flame body billboard quads (custom/Flame.hpp).
// Silhouette carved in Flame.frag; this just places/bends the cards.
// set 0: global uniform. set 1: per-torch (mvp, seed, envelopes, lean).

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	int debugFlags;
	float time;
	// lights[] follows; unread here.
} gubo;

layout(binding = 0, set = 1) uniform FlameUniformBufferObject {
	mat4  mvpMat;      // billboard basis * ViewPrj, CPU-built
	float seed;        // per-torch phase offset
	float intensity;   // brightness envelope, ~0.30..1.40
	vec2  lean;        // billboard-local lean from hand motion
	float heightScale; // height envelope, ~0.78..1.09, lags intensity
	float glareBoost;  // stare-at emphasis, for Flame.frag
	vec3  color;       // target hue, hue-rotated in Flame.frag
} fubo;

// x: flame width. y: 0 (wick) to 1 (tip), before heightScale.
layout(location = 0) in vec2 inCorner;
// Depth-offset card index: 0 (far) to 2 (near).
layout(location = 1) in float inLayer;

// Normalized space Flame.frag's shape function works in.
layout(location = 0) out vec2 uv;
layout(location = 1) flat out float layer;
layout(location = 2) flat out float intensity;
layout(location = 3) flat out float seed;
layout(location = 4) flat out float glare;
layout(location = 5) flat out vec3 color;

void main() {
	// Raw wick(0)-to-tip(1) fraction; lean and shape mask need it unscaled.
	float h = inCorner.y;

	// Slight per-card size variance so the 3 cards don't overlap as one flat plane.
	float layerT = inLayer / 2.0;               // 0, 0.5, 1
	float widthScale       = mix(1.12, 0.88, layerT);
	float layerHeightScale = mix(1.06, 0.94, layerT);

	vec3 pos;
	pos.x = inCorner.x * widthScale;
	// heightScale scales the whole height, not just color/alpha.
	pos.y = h * fubo.heightScale * layerHeightScale;
	// Depth-separate cards along local z so they aren't coplanar.
	pos.z = (inLayer - 1.0) * 0.10;

	// Lean pinned at base, strongest at tip (h*h).
	pos.xz += fubo.lean * h * h;

	gl_Position = fubo.mvpMat * vec4(pos, 1.0);

	uv = vec2(inCorner.x, h);

	layer = inLayer;
	intensity = fubo.intensity;
	seed = fubo.seed;
	glare = fubo.glareBoost;
	color = fubo.color;
}
