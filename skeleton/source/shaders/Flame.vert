// VERTEX SHADER for the torch flames. Paired with Flame.frag, on their own
// pipeline (see PFlame in main.cpp) because a flame needs two things the scene
// pipeline cannot give it: alpha blending, and no back-face culling.
//
// Unlike PosNormUV.vert, which only transforms the vertex, this one MOVES it:
// the mesh is a static sculpt, so every bit of the flame's motion is made
// here, by pushing each vertex along its own normal by an amount that varies
// with position and time. That is deliberately the same operation Blender's
// Displace modifier performs -- the flame was authored that way and glTF has
// no way to export it (it animates node transforms, skins and morph targets,
// never a modifier stack), so it is reproduced on the GPU instead.
//
// Doing it per vertex rather than per pixel matters: this changes the
// silhouette, which is what makes a flame read as alive. A fragment-only
// effect can only ever repaint a shape that never moves.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "custom/Noise.glsl"

// Same block, same order, as PosNormUV.vert and CookTorrance.frag: both
// pipelines share DSLlocal, so one C++ UniformBufferObject feeds both and the
// declarations have to agree field for field. The material fields go unused
// here -- a flame has no BRDF -- but they still have to be declared to keep
// the std140 offsets right.
layout(binding = 0, set = 1) uniform UniformBufferObject {
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

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNorm;
layout(location = 2) out vec2 fragUV;
// Height up the flame, 0 at the base and 1 at the tip. Computed here from the
// local position rather than read from the UVs, so the fragment stage gets a
// value that means the same thing whatever the UV unwrap happens to look like.
layout(location = 3) out float fragHeight;

// The sculpt's local bounding box runs y = 0 .. 0.30 (read off the accessor
// min/max in Flame_01.gltf). Hard-coded rather than passed in: it is a
// property of this one mesh, and this shader is only ever used with it.
const float FLAME_TOP = 0.30;

void main() {
	float h = clamp(inPosition.y / FLAME_TOP, 0.0, 1.0);

	// Two noise lookups at different rates, so the wobble never settles into a
	// visible period. Sampled in the XZ plane and scrolled in time: sampling
	// includes y as well would make the noise slide up the flame, which reads
	// as the surface travelling rather than flickering.
	float n1 = fbm(vec2(inPosition.x * 22.0 + ubo.time * 1.3,
						inPosition.z * 22.0 - ubo.time * 0.9));
	float n2 = fbm(vec2(inPosition.z * 15.0 - ubo.time * 2.1,
						inPosition.x * 15.0 + ubo.time * 1.7));

	// Ramped by h*h: the base is sitting in the torch cup and must stay put,
	// or the flame visibly detaches from the mesh it belongs to. Only the top
	// is free to move, which is also how a real flame behaves.
	float amp = 0.020 * h * h;
	vec3 p = inPosition + normalize(inNormal) * (n1 - 0.5) * 2.0 * amp;

	// Slow sway of the whole tip, on top of the noise. Two different
	// frequencies on the two axes so it traces a wandering path instead of a
	// straight line back and forth.
	p.x += (sin(ubo.time * 2.3) + (n2 - 0.5)) * 0.018 * h * h;
	p.z += (cos(ubo.time * 1.7) + (n2 - 0.5)) * 0.014 * h * h;

	gl_Position = ubo.mvpMat * vec4(p, 1.0);
	fragPos = (ubo.mMat * vec4(p, 1.0)).xyz;
	// The displaced normal is not recomputed: the fragment stage uses it only
	// for a soft-edge term, which a slightly stale normal is plenty accurate
	// for. Deriving the true one would need the noise gradient per vertex.
	fragNorm = mat3(ubo.nMat) * inNormal;
	fragUV = inUV;
	fragHeight = h;
}
