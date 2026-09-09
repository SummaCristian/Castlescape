// VERTEX SHADER for one FACE of a point light's cube shadow map (see
// CubeShadowMap.hpp / PShadowCube in main.cpp). Run 6 times per torch per
// frame, over the same occluder set Shadow.vert uses, flames skipped.
//
// `ubo` is the same per-instance buffer the main pass reads (set 0 here), so
// a moving occluder still casts a following shadow. Unlike Shadow.vert this
// also hands the fragment shader the world position: ShadowCube.frag needs it
// for the linear distance to the light (see that file for why linear).
//
// lightViewProj/lightPos come from a UNIFORM BUFFER, not a push constant:
// the main command buffer is recorded once and reused, and a push constant
// would freeze at record time and never track the held torch. A UBO's
// contents are read fresh at draw time. updateUniformBuffer() maps cubeData
// for every torch each frame.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform UniformBufferObject {
	mat4 mvpMat;
	mat4 mMat;
	mat4 nMat;
	vec3 specularColor;
	float roughness;
	float F0;
	float diffuseShare;
	int flatNormals;
	int interiorAmbient;
	float time;
	float ambientWeight;
	float glow;
	int metallic;
} ubo;

layout(binding = 0, set = 1) uniform ShadowCubeUniformBufferObject {
	mat4 lightViewProj[6];
	vec4 lightPos;	// xyz used, w is padding to keep the block 16-aligned
} cubeData;

// Which of the 6 faces this draw is for. Safe as a push constant, unlike the
// matrix above: it's fixed by where this draw sits in the command buffer.
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
