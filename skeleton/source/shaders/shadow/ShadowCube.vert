// One FACE of a point light's cube shadow map (CubeShadowMap.hpp / PShadowCube,
// main.cpp). Run 6x per torch per frame, flames skipped.
// ubo = same per-instance buffer as the main pass, so moving occluders track.
// Also outputs world position for ShadowCube.frag's linear distance calc.
// lightViewProj/lightPos via UBO not push constant: command buffer is
// recorded once, a push constant would freeze at record time; UBO is read
// fresh per draw (updateUniformBuffer() remaps cubeData each frame).

#version 450
#extension GL_ARB_separate_shader_objects : enable

// Same per-instance buffer as the main pass, so moving occluders track.
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
	vec4 lightPos;	// xyz used, w padding for 16-byte alignment
} cubeData;

// Which of the 6 faces; safe as push constant since it's fixed per draw.
layout(push_constant) uniform ShadowCubeFacePushConstant {
	int face;
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPos;

void main() {
	vec4 worldPos = ubo.mMat * vec4(inPosition, 1.0);
	outWorldPos = worldPos.xyz; // no w
	gl_Position = cubeData.lightViewProj[pc.face] * worldPos; // project from the light's view, this cube face
}
