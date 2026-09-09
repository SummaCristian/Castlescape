// One FACE of a point light's cube shadow map. R32_SFLOAT attachment holding
// LINEAR distance (world units) from light to fragment.
// Linear not projective depth: CookTorrance.frag samples by direction
// (samplerCube), doesn't know which face answered, so value must be
// face-independent -- true for Euclidean distance, not perspective depth.
// No bias folded in here; acne is handled upstream by culling front faces
// (PShadowCube.setCullMode(), main.cpp).

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 1) uniform ShadowCubeUniformBufferObject {
	mat4 lightViewProj[6];
	vec4 lightPos;
} cubeData;

layout(location = 0) in vec3 inWorldPos;

layout(location = 0) out float outDistance;

void main() {
	outDistance = length(inWorldPos - cubeData.lightPos.xyz); // distance from light to nearest occluder here (in the middle -> shadow)
}
