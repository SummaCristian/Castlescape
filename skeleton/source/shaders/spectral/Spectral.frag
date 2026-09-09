// FRAGMENT SHADER for ghosts (technique "Spectral"). Not shaded via
// CookTorrance.frag: a ghost reflects nothing, is unlit, casts no shadow.
// No BRDF. Three parts, in main() order:
//   1. Fresnel rim: density/emission rise at grazing angles (Schlick), making
//      a hollow shell read as a volume; the rim follows the mesh outline.
//   2. Noise: object-space value noise on body density, faded at the rim.
//   3. Emission: unlit, peaking above 1.0 at rim for bloom halo.
// BODY_ALPHA kept high so walls aren't readable through the torso.
// Alpha-blended, depthWriteEnable on, back-face culled (one translucency
// layer per pixel); discard below avoids a ghost-shaped hole through bloom.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// The camera-proximity fade, shared with SpectralDepth.frag.
#include "custom/SpectralFade.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Must match PosNormUV.vert field for field; ghosts reuse the common
// per-instance struct. Fields read here:
//   mMat: instance world matrix (col 3 = origin), transposed for object space.
//   specularColor: spectral tint (materials.json "ghost"), used as emitted color.
//   F0: repurposed as Ghost::chaseBlend (0..1 chase progress), no BRDF meaning here.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 specularColor;
    float roughness;
    float F0;
    float diffuseShare;
    int flatNormals;
    int interiorAmbient;
    float time;
    float ambientWeight;
    float glow;
    int metallic;
} ubo;

// Flat mid-grey with a painted black face; used as a mask, not a color (see faceMask).
layout(binding = 1, set = 1) uniform sampler2D albedoMap;

// Truncated at last field read (std140 offsets are positional). Only eyePos used.
layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    int debugFlags;
    float time;
} gubo;

// ---- The look, in one place ----

// Density face-on / grazing.
const float BODY_ALPHA = 0.46;
const float RIM_ALPHA  = 0.95;
// Grazing-ramp exponent: higher = thinner sharper edge.
const float RIM_POWER  = 2.6;

// Below this, discard instead of drawing invisibly.
const float ALPHA_CUTOFF = 0.015;

// Body/rim emission before tint. Rim clears bloom threshold once alpha
// scales it; body doesn't, so a distant ghost glows at the edge only.
const float BODY_EMISSION = 0.60;
const float RIM_EMISSION  = 2.35;

// Chase tint/brightness, distinct from focus-glow gold and hunt violet.
const vec3  HUNT_TINT     = vec3(1.0, 0.24, 0.20);
const float HUNT_EMISSION = 1.7;

// ---- Value noise, same construction as Flame.frag ----

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2(vec2 p) {
    vec2 cell = floor(p);
    vec2 cellFrac = fract(p);
    float cornerBL = hash21(cell);
    float cornerBR = hash21(cell + vec2(1.0, 0.0));
    float cornerTL = hash21(cell + vec2(0.0, 1.0));
    float cornerTR = hash21(cell + vec2(1.0, 1.0));
    vec2 smoothFrac = cellFrac * cellFrac * (3.0 - 2.0 * cellFrac);
    return mix(mix(cornerBL, cornerBR, smoothFrac.x), mix(cornerTL, cornerTR, smoothFrac.x), smoothFrac.y);
}

// 3 octaves (not Flame.frag's 4): finest octave lands below a pixel at the
// distance a ghost is typically viewed.
float fbm(vec2 p) {
    float sum = 0.0;
    float amp = 0.5;
    for(int i = 0; i < 3; i++) {
        sum += noise2(p) * amp;
        p = p * 2.02 + 11.0;
        amp *= 0.5;
    }
    return sum;
}

void main() {
    vec3 normal = normalize(fragNorm);
    vec3 viewDir = normalize(gubo.eyePos - fragPos);
    // abs(), not max(..,0): a sheet has no inside, folds facing away should
    // thicken too, else they'd punch holes in the silhouette.
    float facing = abs(dot(normal, viewDir));

    // 1. Fresnel rim: decides the whole outline.
    float rim = pow(1.0 - facing, RIM_POWER);
    float alpha = mix(BODY_ALPHA, RIM_ALPHA, rim);

    float chase = clamp(ubo.F0, 0.0, 1.0);

    // Object space via transpose (not inverse): mMat is translate*rotateY, no
    // scale, so mat3 is orthonormal and transpose == inverse. Keeps mottling
    // glued to the ghost as it walks/turns.
    vec3 rel = fragPos - ubo.mMat[3].xyz;
    vec3 objPos = vec3(dot(rel, ubo.mMat[0].xyz),
                       dot(rel, ubo.mMat[1].xyz),
                       dot(rel, ubo.mMat[2].xyz));

    // 2. Noise, body only. (1-rim) weight keeps mottling off the silhouette.
    float flow = fbm(vec2(objPos.x * 1.7 + objPos.z * 1.7, objPos.y * 1.15));
    alpha *= mix(1.0, mix(0.72, 1.20, flow), (1.0 - rim) * 0.7);

    // Face mask: dark texels mark eyes/mouth as solid (opposite of density
    // map), forced dense here and stripped of emission below.
    float texLum = dot(texture(albedoMap, fragUV).rgb, vec3(0.299, 0.587, 0.114));
    float faceMask = 1.0 - smoothstep(0.16, 0.46, texLum);
    alpha = mix(alpha, max(alpha, 0.94), faceMask);

    // Last alpha term: face mask forces 0.94 via max(), must apply after
    // everything else or the eyes/mouth would lose it.
    alpha *= spectralFade(fragPos, gubo.eyePos, ubo.mMat[3].xyz);

    alpha = clamp(alpha, 0.0, 1.0);
    if(alpha < ALPHA_CUTOFF) {
        discard;
    }

    // 3. Emission: tint brightened at rim, pushed toward hunt color by chase.
    vec3 tint = mix(ubo.specularColor, HUNT_TINT, chase);
    float emission = mix(BODY_EMISSION, RIM_EMISSION, rim) * mix(1.0, HUNT_EMISSION, chase);

    // Cool core under the tint (two colors, not one) so the ghost doesn't read flat.
    vec3 core = tint * 0.35 + vec3(0.05, 0.09, 0.14);
    vec3 color = mix(core, tint, rim) * emission;
    // Painted features cut out of the light.
    color *= 1.0 - 0.93 * faceMask;

    // Not premultiplied: blend is srcAlpha*src + (1-srcAlpha)*dst, so alpha
    // already scales color once; multiplying here would square it.
    outColor = vec4(color, alpha);
}
