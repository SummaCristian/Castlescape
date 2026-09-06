// VERTEX SHADER for the 2D shadow pass: depth-only, from the sun's point of
// view (see PShadow / RPShadow in main.cpp). Run once per 2D-shadow light per
// frame over every opaque instance -- the flames are skipped, see Flame.vert.
//
// `ubo` is the SAME per-instance buffer the main pass reads (set 1 there,
// set 0 here since this pipeline has no DSLglobal). It is re-mapped with the
// instance's current Wm every frame, which is what keeps a moving occluder
// casting a shadow that follows it.
//
// The light's view-projection arrives as a PUSH CONSTANT, safe only because
// the sun never moves: the value is baked into the command buffer at record
// time (once per swapchain image) and reused. A push constant that had to
// change per frame would NOT work here.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// Field-for-field the block PosNormUV.vert/CookTorrance.frag declare at set 1.
// Only mMat is read; the rest is here to share that exact layout.
layout(binding = 0, set = 0) uniform UniformBufferObject {
	mat4 mvpMat;
	mat4 mMat;
	mat4 nMat;
	vec3 mS;
	float roughness;
	float F0;
	float k;
	int flatNormals;
	int interiorAmbient;
	float time;
	float ambientWeight;
	float glow;
	int metallic;
} ubo;

layout(push_constant) uniform ShadowPushConstant {
	mat4 lightViewProj;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

void main() {
	gl_Position = pc.lightViewProj * ubo.mMat * vec4(inPosition, 1.0);
}
