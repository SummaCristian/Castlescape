// VERTEX SHADER for one FACE of a point light's cube shadow map (see
// CubeShadowMap.hpp / PShadowCube in main.cpp). Run 6 times per torch per
// frame (once per face), over every opaque scene instance -- same
// occluder set Shadow.vert uses for the sun, same reason the flames are
// skipped (see Shadow.vert's header).
//
// Field-for-field the same reuse trick Shadow.vert documents: `ubo` here is
// the SAME per-instance buffer the main pass's PosNormUV.vert/
// CookTorrance.frag read (DSLlocal, set 1 there, set 0 here), so a moving
// occluder still casts a shadow that follows it.
//
// Unlike Shadow.vert this stage also has to hand the fragment shader the
// world-space position of the vertex: ShadowCube.frag needs it to compute
// the linear distance to the light (see that file's header for why linear
// distance, not raw projective depth, is what gets stored here).
//
// lightViewProj/lightPos come from a UNIFORM BUFFER (set 1, cubeData below),
// not a push constant, even though every torch except the held one is
// static and a push constant would work fine for those: the "main" command
// buffer is recorded ONCE per swapchain image and then reused every frame
// (see submitCommandBuffer()/updateCommandBuffers() in Starter.hpp) -- a
// push constant's value is baked in at THAT record time and never
// re-evaluated, so it can't track a torch that moves after the buffer was
// first recorded. A uniform buffer's CONTENTS, by contrast, are read fresh
// at draw time regardless of when the command that binds it was recorded --
// the same reason every per-instance Wm survives being re-mapped every
// frame instead of re-recorded. See main.cpp's updateUniformBuffer(), which
// maps cubeData for every torch (not just the held one) each frame, the
// same way ShadowUniformBufferObject gets re-mapped for the always-static
// sun.

#version 450
#extension GL_ARB_separate_shader_objects : enable

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
	int metallic;
} ubo;

layout(binding = 0, set = 1) uniform ShadowCubeUniformBufferObject {
	mat4 lightViewProj[6];
	vec4 lightPos;	// xyz used, w is padding to keep the block 16-aligned
} cubeData;

// Which of the 6 faces this draw is for. Safe as a push constant unlike the
// matrix/position above: it's a fixed property of WHERE in the recorded
// command buffer this draw call sits (always face 0, then face 1, ...),
// never something that needs to change after the buffer is recorded.
layout(push_constant) uniform ShadowCubeFacePushConstant {
	int face;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPos;

void main() {
	vec4 worldPos = ubo.mMat * vec4(inPosition, 1.0);
	outWorldPos = worldPos.xyz;
	gl_Position = cubeData.lightViewProj[pc.face] * worldPos;
}
