// VERTEX SHADER for the flame's ember shard particles (see custom/Flame.hpp).
// Entirely procedural: no CPU-side particle system, no per-particle uniform
// data at all beyond what's already baked into the mesh (particleSeed) and
// already bound for the flame body (gubo.time, fubo.mvpMat/seed). Each
// shard's whole lifecycle -- rise, drift, shrink, fade -- is a function of
// those three things, looping forever via fract().

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
	vec3 eyePos;
	int lightCount;
	vec3 ambientUpper;
	vec3 ambientLower;
	vec3 ambientDir;
	int debugFlags;
	float time;
} gubo;

// Same block shape as the flame body's own FlameUniformBufferObject -- see
// Flame.hpp on why embers reuse that DS instead of getting their own.
layout(binding = 0, set = 1) uniform EmberUniformBufferObject {
	mat4 mvpMat;
	float seed;
} fubo;

layout(location = 0) in vec3 inPos;
layout(location = 1) in float inParticleSeed;

layout(location = 0) flat out float tphase;

void main() {
	// Loop period varies per shard (2.0..3.6s) so EMBER_COUNT particles
	// don't all pop back to their start at the same instant.
	float period = 2.0 + inParticleSeed * 1.6;
	float tphase_ = fract((gubo.time * 0.55 + fubo.seed * 2.7 + inParticleSeed * 13.0) / period);

	// Spawn scattered around the crown, in the SAME local space the flame
	// body's own mesh lives in (fubo.mvpMat here is that body's own mvp), so
	// embers visibly originate from it rather than some unrelated point.
	vec3 startPos = vec3((fract(inParticleSeed * 17.0) - 0.5) * 0.18,
						  0.6,
						  (fract(inParticleSeed * 31.0) - 0.5) * 0.18);

	float rise = tphase_ * 0.9;
	float driftPhase = gubo.time * 2.2 + inParticleSeed * 23.0;
	vec3 drift = vec3(sin(driftPhase) * 0.10 * tphase_,
					   rise,
					   cos(driftPhase * 1.4) * 0.10 * tphase_);

	// Shrinks as it rises and cools, on top of fading out in Ember.frag.
	float shrink = mix(1.0, 0.25, tphase_);
	vec3 p = startPos + drift + inPos * shrink;

	gl_Position = fubo.mvpMat * vec4(p, 1.0);
	tphase = tphase_;
}
