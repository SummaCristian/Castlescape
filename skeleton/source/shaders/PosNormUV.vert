// VERTEX SHADER. Runs on the GPU once per vertex, before the fragment shader.
// Two jobs:
//   1. decide where the vertex ends up on screen (gl_Position)
//   2. hand the fragment shader whatever it needs about this vertex
//
// The "out" variables below are not delivered as-is: the GPU interpolates them
// across the triangle, so a pixel in the middle gets a blend of the three
// corners. That is how a smoothly-lit surface comes out of three corner normals.
//
// It exists as a separate file from the fragment shader because the two run at
// completely different rates: this one a few thousand times per frame, the other
// a few million.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// The material fields go unused here, but the block must be declared identically
// in both stages: one buffer at one binding, shared by the two.
layout(binding = 0, set = 1) uniform UniformBufferObject {
	mat4 mvpMat;
	mat4 mMat;
	mat4 nMat;
	vec3 mS;
	float roughness;
	float F0;
	float k;
	int flatNormals;
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
	// nMat, not mMat: a non-uniform scale would leave the normal no longer
	// perpendicular. Not normalized, the fragment stage has to do it anyway.
	fragNorm = mat3(ubo.nMat) * inNormal;
	fragUV  = inUV;
}
