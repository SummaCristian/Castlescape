// DEPTH PREPASS for the ghosts. Writes no colour: its product is the depth of
// the ghost surface NEAREST the eye, which stops Spectral.frag (run right
// after, same mesh) from blending the ghost's far side into its near side.
//
// Spectral.frag draws a translucent shell, and "behind it" includes the rest
// of the ghost -- feet inside the robe, the far torso wall. Those fragments
// arrive in index order, so a far one often rasterises first, writes depth,
// and is blended under the near surface: the feet's rim glowing through the
// body. Back-face culling removes the far shell but not front-facing geometry
// genuinely inside it.
//
// This pass runs with depth writes on and LESS, so afterwards the buffer holds
// the MINIMUM ghost depth per pixel. The colour pass runs LESS_OR_EQUAL, so
// only the nearest layer survives. Both share PosNormUV.vert and the same UBO,
// so positions are bit-identical and the equality holds. It costs nothing
// visible: alpha 0 returns the destination unchanged -- a colour-write mask
// spelled with a blend factor, since Pipeline exposes setTransparency() and
// not the mask.
//
// The one DISCARD replicates spectralFade() only -- not ALPHA_CUTOFF, which
// the body never comes near. The fade takes the whole shell to zero, and a
// prepass still writing depth there would punch a ghost-shaped hole through
// the flames and exit glow in exactly the case the fade exists for: the
// player standing inside a ghost. Threshold 0.015 / 0.33, where the thinnest
// body fragment drops under the colour cutoff.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "custom/SpectralFade.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Both blocks truncated at the last field read (std140 offsets are positional).
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
} ubo;

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
} gubo;

// 0.015 / 0.33 -- see the header.
const float FADE_CUTOFF = 0.045;

void main() {
    if(spectralFade(fragPos, gubo.eyePos, ubo.mMat[3].xyz) < FADE_CUTOFF) {
        discard;
    }
    outColor = vec4(0.0);
}
