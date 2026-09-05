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

// lightPos comes from the same per-torch uniform buffer ShadowCube.vert
// reads (set 1, binding 0) instead of a push constant -- see that file's
// header for why: a push constant is frozen into the command buffer at
// record time, which happens once and is then reused every frame, so it
// can't track a torch (the held one) that keeps moving after that.
//
// The distance stored is the raw one, with no bias of any kind folded into
// it, and that is deliberate.
//
// A slope-scaled bias used to be added here, on the theory that shadow acne
// belongs to the surface being sampled and should be paid for by it rather
// than by whatever it shades. The theory is right and the practice was not:
// a bias baked into the map is invisible to everything downstream, including
// LIGHT_DEBUG_SHADOW_GAP, which computes its gap from the stored value and
// so reported every fragment that bias forgave as geometrically unoccluded.
// It hid the very artifact it was contributing to. Whatever a shadow map
// stores should be a measurement; slack belongs where it can still be seen.
//
// Acne is dealt with upstream of all of it now, by culling FRONT faces in
// this pass so a lit surface is never in the map to compare against itself.
// See PShadowCube.setCullMode() in main.cpp.

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
