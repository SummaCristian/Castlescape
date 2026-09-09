// DEPTH PREPASS for ghosts. Writes no color: produces the depth of the ghost
// surface nearest the eye, so Spectral.frag (same mesh, runs after) doesn't
// blend the far side (e.g. feet inside robe) through the near surface.
//
// Runs with depth test LESS, buffer ends up holding MINIMUM ghost depth per
// pixel; color pass runs LESS_OR_EQUAL so only the nearest layer survives.
// Both share PosNormUV.vert/UBO so positions match bit-for-bit. Alpha 0 keeps
// this invisible (a color-write mask via blend factor).
//
// The discard replicates spectralFade() only, not ALPHA_CUTOFF: a prepass
// still writing depth during the fade would punch a ghost-shaped hole through
// bloom while the player stands inside a ghost.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "custom/SpectralFade.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Truncated at last field read (std140 offsets are positional).
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
} ubo;

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
} gubo;

const float FADE_CUTOFF = 0.045;

void main() {
    if(spectralFade(fragPos, gubo.eyePos, ubo.mMat[3].xyz) < FADE_CUTOFF) {
        discard;
    }
    outColor = vec4(0.0);
}
