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
	float time;
} ubo;

// Vertex stage reads lightViewProj, fragment stage reads lightPos -- one
// push constant block, both stages, same layout Shadow.vert's is a strict
// subset of.
layout(push_constant) uniform ShadowCubePushConstant {
	mat4 lightViewProj;
	vec4 lightPos;	// xyz used, w is padding to keep the block 16-aligned
} pc;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 outWorldPos;

void main() {
	vec4 worldPos = ubo.mMat * vec4(inPosition, 1.0);
	outWorldPos = worldPos.xyz;
	gl_Position = pc.lightViewProj * worldPos;
}
