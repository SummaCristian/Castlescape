// FRAGMENT SHADER for one FACE of a point light's cube shadow map. It has a
// real color attachment (one R32_SFLOAT channel per face) and fills it with
// the LINEAR distance from the light to the fragment, in world units.
//
// Linear, not projective depth: CookTorrance.frag samples the cube by
// DIRECTION with an ordinary samplerCube, and that lookup doesn't know which
// face answered, so the stored value has to mean the same on every face --
// true for a Euclidean distance, false for a perspective depth (which warps
// across a face and again on the adjacent one).
//
// The distance stored is raw, with NO bias folded in. A slope-scaled bias
// here would be invisible downstream, hiding the artifact it caused. A
// shadow map should store a measurement; slack belongs where it can be
// seen. Acne is handled upstream instead, by culling FRONT faces so a lit
// surface is never in its own map (PShadowCube.setCullMode() in main.cpp).

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 1) uniform ShadowCubeUniformBufferObject {
	mat4 lightViewProj[6];
	vec4 lightPos;
} cubeData;

layout(location = 0) in vec3 inWorldPos;

layout(location = 0) out float outDistance;

void main() {
	outDistance = length(inWorldPos - cubeData.lightPos.xyz);
}
