// VERTEX SHADER: once per vertex. Places the vertex on screen (gl_Position)
// and hands the fragment shader what it needs about it. The "out" variables
// are interpolated across the triangle, which is how a smoothly-lit surface
// comes out of three corner normals.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// The material fields are unused here, but the block must be declared
// identically in both stages: one buffer, one binding, shared.
layout(binding = 0, set = 1) uniform UniformBufferObject {
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

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNorm;
layout(location = 2) out vec2 fragUV;

void main() {
	gl_Position = ubo.mvpMat * vec4(inPosition, 1.0);
	fragPos = (ubo.mMat * vec4(inPosition, 1.0)).xyz;
	// nMat, not mMat: a non-uniform scale would tilt the normal off the
	// surface. Not normalized -- the fragment stage does it anyway.
	fragNorm = mat3(ubo.nMat) * inNormal;
	fragUV  = inUV;
}
