// VERTEX SHADER for the shadow pass: renders depth-only, from a light's point
// of view, into one of the six shadow maps (see PShadow / RPShadow in
// main.cpp). Run once per shadow-casting light per frame, over every opaque
// scene instance (the CookTorrance technique only -- the flames are
// deliberately not drawn here, see Flame.vert's header).
//
// This is the one place in the renderer that reuses a descriptor set across
// two different pipelines: `ubo` below is bound to the SAME per-instance
// buffer the main pass's PosNormUV.vert/CookTorrance.frag already read every
// frame (DSLlocal, set 1 there), just declared at set 0 here since this
// pipeline has no DSLglobal in front of it. That buffer is re-mapped with the
// instance's current Wm every frame regardless of which pipeline reads it
// (see updateUniformBuffer in main.cpp), which is what keeps a moving
// occluder -- the door, the watching skulls -- casting a shadow that follows
// it instead of the position it was first drawn at.
//
// The light's view-projection matrix, by contrast, arrives as a PUSH
// CONSTANT rather than through a UBO. That is safe only because it is
// genuinely constant: the sun and the torches never move, so the value baked
// into the command buffer the first time this pipeline is recorded (see
// BaseProject::submitCommandBuffer -- a command buffer is built once per
// swapchain image and reused after that) stays correct forever. A push
// constant that needed to change frame to frame would NOT work here for
// exactly that reason.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// Field-for-field the same block PosNormUV.vert and CookTorrance.frag declare
// at set 1 -- see the header comment above for why. Only mMat is read; the
// rest exists so this shares that exact buffer layout instead of needing one
// of its own.
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
