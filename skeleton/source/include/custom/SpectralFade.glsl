// Ghost fade factor from camera position. Shared by Spectral.frag and its
// depth prepass. min() of two terms:
//   INSIDE  whole instance, faded by eye distance to ghost's vertical axis
//           (catches standing inside, where per-fragment distance can't reach
//           the arms/hem)
//   NEAR    per-fragment, by distance to eye (catches brushing the robe off-axis)
// main.cpp ramps the full-screen veil over the same range INSIDE uses; retune together.

#ifndef SPECTRAL_FADE_GLSL
#define SPECTRAL_FADE_GLSL

// Set outside the body (robe is ~0.5 wide) so arms/hem points fade too,
// not just the axis.
const float SPECTRAL_INSIDE_FAR  = 2.00;
const float SPECTRAL_INSIDE_NEAR = 1.05;
// Ghost.gltf Y bounds + margin (ghostBodyBottom/Top in main.cpp); must fade
// on crossing or the bob flickers the ghost from below.
const float SPECTRAL_INSIDE_Y_MIN = -1.80;
const float SPECTRAL_INSIDE_Y_MAX =  0.83;
const float SPECTRAL_INSIDE_Y_FADE = 0.60;

const float SPECTRAL_NEAR_FAR  = 1.30;
const float SPECTRAL_NEAR_NEAR = 0.38;

// ghostOrigin = column 3 of instance world matrix.
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
