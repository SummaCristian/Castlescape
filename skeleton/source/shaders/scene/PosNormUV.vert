// Places vertex on screen and passes interpolated data to fragment shader.

#version 450
#extension GL_ARB_separate_shader_objects : enable

// Material fields unused here, but block must match the fragment stage's layout.
layout(binding = 0, set = 1) uniform UniformBufferObject {
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

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNorm;
layout(location = 2) out vec2 fragUV;

void main() {
	gl_Position = ubo.mvpMat * vec4(inPosition, 1.0);
	fragPos = (ubo.mMat * vec4(inPosition, 1.0)).xyz;
	// nMat not mMat: non-uniform scale would tilt normal off surface.
	fragNorm = mat3(ubo.nMat) * inNormal;
	fragUV  = inUV;
}
