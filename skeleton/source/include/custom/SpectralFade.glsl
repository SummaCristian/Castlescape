// How much of a ghost is drawn, given where the camera is. Shared by
// Spectral.frag and its depth prepass so the two agree exactly -- if they
// didn't, the prepass would write depth for a shell the colour pass no longer
// draws and punch a ghost-shaped hole through the flames and the exit glow.
//
// Two terms, combined with min():
//
//   INSIDE  the whole instance, faded by how close the EYE is to the ghost's
//           vertical axis. This is the one that matters: standing inside a
//           ghost, the arms and the hem's points are a metre or more away and
//           no per-fragment distance can reach them, so they stay hanging in
//           frame while the body around them thins out.
//   NEAR    per fragment, by distance to the eye. Covers brushing past the
//           edge of the robe without being at its axis, where the near clip
//           plane (0.1) would slice the sheet open.
//
// main.cpp ramps the full-screen veil (SPECTRAL_VEIL_*) over the same range
// INSIDE uses, so what leaves the mesh arrives on the frame. Retune together.

#ifndef SPECTRAL_FADE_GLSL
#define SPECTRAL_FADE_GLSL

// NEAR is set well OUTSIDE the body, not at its axis: the robe is about 0.5
// wide (ghostRadius in main.cpp), so the whole ghost is gone a step before the
// player crosses into it -- arms and hem points included, which are the parts
// a per-fragment fade never reaches. Anything tighter leaves them hanging in
// frame on the way in.
const float SPECTRAL_INSIDE_FAR  = 2.00;
const float SPECTRAL_INSIDE_NEAR = 1.05;
// Ghost.gltf's own Y bounds, the same fit main.cpp keeps in ghostBodyBottom /
// ghostBodyTop, plus a margin: crossing them has to fade, or the bob switches
// the whole ghost on and off from below.
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
