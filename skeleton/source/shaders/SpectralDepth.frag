// DEPTH PREPASS for the ghosts. Writes no colour: its only product is the
// depth of the ghost surface NEAREST the eye, which is what stops Spectral.frag
// -- run right after it, same mesh, same matrices -- from blending the ghost's
// far side into its near side.
//
// The problem it solves. Spectral.frag draws a translucent shell over whatever
// is behind it, and "whatever is behind it" includes the rest of the ghost: the
// feet inside the robe, the far wall of the torso, the inside of the hem. Those
// fragments come out of the mesh in index order, so the far ones are frequently
// rasterised FIRST, write their depth, and are then blended under the near
// surface, which passes the depth test because it is closer. The result is the
// feet's Fresnel rim glowing through the body -- and a rim is exactly what
// survives that blend most visibly, being the brightest thing this technique
// produces.
//
// Back-face culling (see Pspectral in main.cpp) already removes the far side of
// the shell. It does nothing about geometry that is genuinely inside the shell
// and front-facing, which is what the feet are.
//
// Why this works. The prepass runs with depth writes on (hardcoded in
// Starter.hpp) and the default VK_COMPARE_OP_LESS, so after it the depth buffer
// holds the MINIMUM ghost depth per pixel. The colour pass then runs with
// LESS_OR_EQUAL and only the fragments at exactly that depth survive: one layer
// per pixel, whichever is nearest, with the interior rejected instead of
// blended in. The two passes share PosNormUV.vert and the same per-instance
// UBO, so they compute bit-identical positions and the equality holds.
//
// Why it costs nothing visible. The blend is srcAlpha * src + (1 - srcAlpha) *
// dst and this shader emits alpha 0, so the destination is returned unchanged;
// the attachment comes out of the prepass exactly as it went in. It is a
// colour-write mask spelled with a blend factor, because Pipeline exposes
// setTransparency() and not the mask.
//
// It also declares no descriptors -- not the UBO, not the albedo map. Only the
// vertex stage needs anything, and the pipeline layout it is created with is
// Pspectral's, so the sets Scene binds for the instance fit either pipeline.
//
// NO DISCARD, deliberately, where the colour pass discards below ALPHA_CUTOFF.
// Replicating it would mean recomputing the rim, the noise and the face mask
// here to reach a branch that, with BODY_ALPHA at 0.46 and the noise bottoming
// out at 0.72 of it, the body cannot take: the smallest alpha the ghost can
// reach is around 0.33, twenty times the cutoff. The cutoff guards the depth
// buffer against a fully transparent fragment punching a hole through the
// flames, and there is no such fragment to guard against.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(0.0);
}
