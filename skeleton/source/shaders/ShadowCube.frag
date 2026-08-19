// FRAGMENT SHADER for one FACE of a point light's cube shadow map. Unlike
// Shadow.frag (depth-only, writes nothing itself) this pass has a real color
// attachment -- a single VK_FORMAT_R32_SFLOAT channel per face of
// CubeShadowMap.hpp's cube image -- and this is what fills it: the LINEAR
// distance from the light to the fragment, world units, not a projective
// 0..1 depth.
//
// Why linear distance instead of the depth buffer's own value (as Shadow.vert/
// frag do for the sun): CookTorrance.frag samples this cube by DIRECTION
// (fragPos - lightPos) with an ordinary samplerCube, which is what makes a
// point light's shadow a single lookup instead of the old pick-the-covering-
// map dance. That lookup has no notion of which face answered, so whatever
// gets compared against it has to mean the same thing on every face -- true
// for a Euclidean distance, false for a perspective-projected depth (which
// warps differently near the center of a face than near its edge, and
// differently again on the ADJACENT face sharing that edge).

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(push_constant) uniform ShadowCubePushConstant {
	mat4 lightViewProj;
	vec4 lightPos;
} pc;

layout(location = 0) in vec3 inWorldPos;

layout(location = 0) out float outDistance;

void main() {
	outDistance = length(inWorldPos - pc.lightPos.xyz);
}
