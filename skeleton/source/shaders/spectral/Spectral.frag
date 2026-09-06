// FRAGMENT SHADER for the ghosts (technique "Spectral", Pspectral in main.cpp).
// Everything else goes through CookTorrance.frag; the ghosts don't, because a
// ghost is not a surface: it reflects nothing, no torch lights it, and it
// casts no shadow. Shaded as a dielectric it read as an opaque grey statue
// with a missing shadow. Shaded as an emissive translucent volume, the missing
// shadow becomes the point.
//
// No BRDF, no light read. Three things build the apparition, in main() order:
//   1. FRESNEL RIM. Density and emission both go up where the surface turns
//      away from the eye -- Schlick's grazing term, on ALPHA and brightness.
//      It makes a hollow shell read as a volume, and it IS the whole outline:
//      the rim follows the MESH, so the model's ragged hem lights up on its
//      own.
//   2. NOISE. Object-space value noise on the body's density, faded out at the
//      rim so the silhouette stays clean. Same construction as Flame.frag.
//   3. EMISSION. Written unlit, peaking above 1.0 at the rim so the bloom
//      bright pass picks the silhouette up as a halo.
//
// NOT here, on purpose: a vertical dissolve with a fake scalloped hem (the
// mesh already ends in points -- the outline is its job); any animation (the
// ghost already moves, a pulse on top reads as a flickering lamp; ubo.time is
// unread). Only the chase tint changes over time.
//
// BODY_ALPHA is high on purpose: low, you could read the brick courses through
// a torso, and a recognisable pattern seen through something makes it a
// window. Real translucency lives at the edges. Raise this one number if lit
// walls start showing their courses.
//
// The pipeline is alpha-blended with depthWriteEnable hardcoded on, and keeps
// BACK-FACE CULLING -- one layer of translucency per pixel, no self-blend
// order to get wrong. That's also why the discard below matters: a transparent
// fragment reaching the depth stage would still punch a ghost-shaped hole
// through the flames and exit glow.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// The camera-proximity fade, shared with SpectralDepth.frag.
#include "custom/SpectralFade.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Must match PosNormUV.vert field for field; the ghosts ride the same
// per-instance struct as every other prop. Three fields are read:
//   mMat  the instance's world matrix, transposed below for object space. Col 3 is the origin.
//   mS    the spectral tint (materials.json "ghost"). Named specularColor for
//         CookTorrance.frag; here it's the emitted colour.
//   F0    0..1, how far this ghost is into a CHASE. Reflectance is meaningless
//         without a BRDF, so main.cpp reuses the field for Ghost::chaseBlend --
//         same piggybacking `glow` does on the other technique.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 mS;
    float roughness;
    float F0;
    float k;
    int flatNormals;
    int interiorAmbient;
    float time;
    float ambientWeight;
    float glow;
    int metallic;
} ubo;

// Flat mid-grey with a black face (two eyes, a mouth) painted on. Read as a
// MASK for those marks, not as a colour -- see faceMask in main().
layout(binding = 1, set = 1) uniform sampler2D albedoMap;

// Truncated at the last field read. std140 offsets are positional, so stopping
// early is legal, reordering is not. Only eyePos matters here.
layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    vec3 ambientUpper;
    vec3 ambientLower;
    vec3 ambientDir;
    int debugFlags;
    float time;
} gubo;

// ---- The look, in one place ----

// Density face-on and at a grazing angle (see the header on the face-on one).
const float BODY_ALPHA = 0.46;
const float RIM_ALPHA  = 0.95;
// Grazing-ramp exponent, the most important number here: higher is a thinner,
// sharper edge. Raise it if the ghost reads as a blob, lower it if the glow
// creeps up the body.
const float RIM_POWER  = 2.6;

// Below this the fragment is discarded, not drawn invisibly (header).
const float ALPHA_CUTOFF = 0.015;

// Body and rim emission, before the tint. The rim clears BLOOM_THRESHOLD once
// alpha scales it, putting a halo on the silhouette; the body doesn't, so a
// distant ghost glows at its edge rather than becoming a lamp. RIM_EMISSION is
// capped by COLOUR: pushed higher it saturates to white and throws the tint
// away where the ghost is brightest.
const float BODY_EMISSION = 0.60;
const float RIM_EMISSION  = 2.35;

// Chase tint and brightness. Kept clear of the focus glow's gold and the
// flames' hunt violet -- this cue alone means "it has seen you".
const vec3  HUNT_TINT     = vec3(1.0, 0.24, 0.20);
const float HUNT_EMISSION = 1.7;

// ---- Value noise, identical construction to Flame.frag ----

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 3 octaves, not Flame.frag's 4: stretched over a whole body, the finest
// octave lands below a pixel at any distance the ghost is seen from.
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
    vec3 N = normalize(fragNorm);
    vec3 V = normalize(gubo.eyePos - fragPos);
    // abs(), not max(..., 0): a sheet has no inside, so a fold facing away
    // should thicken as much as one facing towards. Without abs those folds
    // punch holes in the silhouette.
    float facing = abs(dot(N, V));

    // 1. Fresnel rim. The outline of the ghost, and nothing else decides it.
    float rim = pow(1.0 - facing, RIM_POWER);
    float alpha = mix(BODY_ALPHA, RIM_ALPHA, rim);

    float chase = clamp(ubo.F0, 0.0, 1.0);

    // OBJECT SPACE by transposing the rotation, not inverting it: a ghost's Wm
    // is translate * rotateY with no scale, so mat3(mMat) is orthonormal and
    // its transpose is its inverse. Only the noise needs it, so the mottling
    // stays glued to the ghost as it walks and turns.
    vec3 rel = fragPos - ubo.mMat[3].xyz;
    vec3 objPos = vec3(dot(rel, ubo.mMat[0].xyz),
                       dot(rel, ubo.mMat[1].xyz),
                       dot(rel, ubo.mMat[2].xyz));

    // 2. Noise, BODY only. The (1 - rim) weight protects the silhouette: at
    // the edge the mottling is fully faded, so noise can't bite the outline.
    // Static -- see the header.
    float flow = fbm(vec2(objPos.x * 1.7 + objPos.z * 1.7, objPos.y * 1.15));
    alpha *= mix(1.0, mix(0.72, 1.20, flow), (1.0 - rim) * 0.7);

    // The face. Only the texture's dark texels carry information, and they
    // mean the opposite of a density map: a painted mark should read as solid,
    // not thin. So the mask makes the marks the densest thing on the model
    // and, below, stops them emitting -- black features floating in the glow.
    float texLum = dot(texture(albedoMap, fragUV).rgb, vec3(0.299, 0.587, 0.114));
    float faceMask = 1.0 - smoothstep(0.16, 0.46, texLum);
    alpha = mix(alpha, max(alpha, 0.94), faceMask);

    // LAST alpha term: the face mask forces its marks to 0.94 with a max(),
    // so anything applied before it would be lost on the eyes and mouth.
    alpha *= spectralFade(fragPos, gubo.eyePos, ubo.mMat[3].xyz);

    alpha = clamp(alpha, 0.0, 1.0);
    if(alpha < ALPHA_CUTOFF) {
        discard;
    }

    // 3. Emission. Unlit: the tint, brightened at the rim, then pushed towards
    // the hunt colour by however far into a chase this ghost is.
    vec3 tint = mix(ubo.mS, HUNT_TINT, chase);
    float emission = mix(BODY_EMISSION, RIM_EMISSION, rim) * mix(1.0, HUNT_EMISSION, chase);

    // A cool core under the tint, strongest where the sheet is thinnest. Two
    // colours, not one, so the middle is a different temperature from the edge
    // -- what stops the ghost reading as flat.
    vec3 core = tint * 0.35 + vec3(0.05, 0.09, 0.14);
    vec3 color = mix(core, tint, rim) * emission;
    // The painted features, cut out of the light.
    color *= 1.0 - 0.93 * faceMask;

    // NOT premultiplied: the blend is srcAlpha * src + (1 - srcAlpha) * dst,
    // so alpha already scales this colour once. Multiplying it in here would
    // square that and leave the body black.
    outColor = vec4(color, alpha);
}
