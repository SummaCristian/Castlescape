// How much of a ghost is drawn, given the camera position. Shared by
// Spectral.frag and its depth prepass so the two agree exactly. Two terms,
// min()'d:
//   INSIDE  the whole instance, faded by how close the EYE is to the ghost's
//           vertical axis. The one that matters: standing inside a ghost, the
//           arms and hem points are a metre away and no per-fragment distance
//           reaches them.
//   NEAR    per fragment, by distance to the eye. Covers brushing the robe
//           off-axis, where the 0.1 near clip would slice the sheet open.
//
// main.cpp ramps the full-screen veil over the same range INSIDE uses, so what
// leaves the mesh arrives on the frame. Retune together.

#ifndef SPECTRAL_FADE_GLSL
#define SPECTRAL_FADE_GLSL

// Set well OUTSIDE the body, not at its axis: the robe is ~0.5 wide, so the
// ghost is gone a step before the player crosses into it -- arms and hem
// points included, which a per-fragment fade never reaches.
const float SPECTRAL_INSIDE_FAR  = 2.00;
const float SPECTRAL_INSIDE_NEAR = 1.05;
// Ghost.gltf's Y bounds plus a margin (ghostBodyBottom/Top in main.cpp):
// crossing them must fade, or the bob switches the ghost on and off from below.
const float SPECTRAL_INSIDE_Y_MIN = -1.80;
const float SPECTRAL_INSIDE_Y_MAX =  0.83;
const float SPECTRAL_INSIDE_Y_FADE = 0.60;

const float SPECTRAL_NEAR_FAR  = 1.30;
const float SPECTRAL_NEAR_NEAR = 0.38;

// ghostOrigin is column 3 of the instance's world matrix.
float spectralFade(vec3 fragPos, vec3 eyePos, vec3 ghostOrigin) {
    vec3 eyeRel = eyePos - ghostOrigin;

    float vy = smoothstep(SPECTRAL_INSIDE_Y_MIN - SPECTRAL_INSIDE_Y_FADE,
                          SPECTRAL_INSIDE_Y_MIN, eyeRel.y) *
               (1.0 - smoothstep(SPECTRAL_INSIDE_Y_MAX,
                                 SPECTRAL_INSIDE_Y_MAX + SPECTRAL_INSIDE_Y_FADE, eyeRel.y));
    float axis = smoothstep(SPECTRAL_INSIDE_NEAR, SPECTRAL_INSIDE_FAR, length(eyeRel.xz));
    float inside = mix(1.0, axis, vy);

    float near = smoothstep(SPECTRAL_NEAR_NEAR, SPECTRAL_NEAR_FAR,
                            length(eyePos - fragPos));

    return min(inside, near);
}

#endif
