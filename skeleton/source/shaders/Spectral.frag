// FRAGMENT SHADER for the ghosts (technique "Spectral" in scene.json, see the
// Pspectral pipeline in main.cpp). Everything else in this scene goes through
// CookTorrance.frag; the ghosts do not, and the reason is worth stating.
//
// A ghost is not a surface. It reflects nothing, it is not lit by the torches
// it drifts past, and -- since 46777d5 took them out of the shadow maps for
// cost reasons (see materials.json's "ghost" entry) -- it casts no shadow
// either. Shaded as an ordinary dielectric it read as exactly that: an opaque
// grey statue with a missing shadow, which is a bug you can see. Shaded as an
// emissive translucent volume the missing shadow stops being a bug and becomes
// the point: what has no body has nothing to block a torch with.
//
// So this shader computes no BRDF and reads no light. Three things build the
// apparition instead, in the order main() applies them:
//
//   1. a FRESNEL RIM. Density and emission both go up where the surface turns
//      away from the eye, which is the same grazing-angle term the specular in
//      CookTorrance.frag uses (F_Schlick), applied to ALPHA and to brightness
//      rather than to reflectance. It is what makes a hollow shell read as a
//      volume, and it is also the whole of the outline: the rim follows the
//      MESH, so the model's own ragged hem lights up because those tips are
//      thin geometry seen edge-on, not because anything here drew them.
//   2. NOISE. Object-space value noise mottling the body's density, faded out
//      at the rim so the silhouette stays clean. Same construction as
//      Flame.frag: no texture lookups, detail at more than one scale.
//   3. EMISSION. The colour is written straight out, unlit, peaking above 1.0
//      at the rim so the bloom chain's bright pass (BLOOM_THRESHOLD = 1.55 in
//      main.cpp) picks the silhouette up as a halo.
//
// WHAT IS DELIBERATELY NOT HERE, because it was and was wrong:
//
//   A vertical dissolve, and a procedural scalloped hem cut into it. Ghost.gltf
//   already ends in points -- measured around the body, its lowest vertex per
//   10 degrees swings between -1.80 and -1.57, so the hem is modelled with
//   roughly 0.23 of tooth. The dissolve faded everything from -1.75 up to
//   -0.55, which erased that hem completely and then drew a fake one in its
//   place, angled off atan() instead of off the geometry. Two hems, neither of
//   them the model's. The outline is the mesh's job; this shader's job is to
//   light it.
//
//   Any animation. No pulse, no scrolling noise. The ghost already moves, bobs
//   and turns, and a brightness pulse on top of that reads as a flickering lamp
//   rather than as a presence. The only thing that changes over time is the
//   chase tint, which is a gameplay tell that moves once and holds. ubo.time is
//   deliberately unread.
//
// On DENSITY. BODY_ALPHA is high, and that is a correction, not a default: at
// the 0.16 it started on you could read the dungeon's brick courses straight
// through a ghost's torso, and a recognisable pattern seen through something
// makes it a window, not a body. Real translucency lives at the edges here,
// which is also where every game ghost puts it. 0.58 killed the bricks but left
// the torso reading as painted plastic, so the value sits between: enough that
// the room behind is felt rather than read. If a lit wall starts showing its
// courses again, this is the one number to raise, and 0.58 is known safe.
//
// On BLENDING and DEPTH. The pipeline is alpha-blended (setTransparency(true)),
// and Starter.hpp hardcodes depthWriteEnable = VK_TRUE with no way to switch it
// off -- the same constraint Flame.hpp:266 works around by emitting its
// billboard layers back to front. There is no equivalent ordering to author
// here, since the ghost is a rigid mesh, so the pipeline keeps BACK-FACE
// CULLING instead: with one layer of translucency per pixel there is no
// self-blending order to get wrong. That is also why the discard below matters
// rather than being an optimization -- a fully transparent fragment that
// reaches the depth stage would still write depth and punch a ghost-shaped hole
// through the flames and the exit glow, both of which draw after the scene.

#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Must match the block in PosNormUV.vert field for field: one buffer at one
// binding, shared by the two stages, and shared again with CookTorrance.frag
// (the ghosts ride the same per-instance C++ struct as every other prop --
// see updateUniformBuffer()).
//
// Most of it goes unread here. Three fields are used:
//   mMat       the instance's world matrix, transposed below to get object
//              space. Column 3 is its origin.
//   mS         the spectral tint, from materials.json's "ghost" entry. It is
//              named specularColor because that is what it means to
//              CookTorrance.frag; on this technique nothing is specular, and
//              it is the colour the apparition emits.
//   F0         0..1, how far this ghost is into a CHASE. Reflectance-head-on
//              is meaningless without a BRDF, so the field is free, and
//              main.cpp overwrites it per ghost instance with Ghost::chaseBlend
//              after the material copy. Same piggybacking `glow` already does
//              on the other technique, and for the same reason: the struct's
//              spare room is spent, and adding a field costs a change in four
//              files that agree on the layout by hand.
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

// Flat mid-grey with a face painted on it in black -- two eyes and a mouth,
// nothing else. Read below as a MASK for those marks rather than as a colour:
// see faceMask in main().
layout(binding = 1, set = 1) uniform sampler2D albedoMap;

// Truncated on purpose, stopping at the last field this shader reads. std140
// offsets are positional, so a shader may stop declaring early but may never
// skip or reorder anything before what it uses -- Flame.frag declares the same
// prefix for the same reason. eyePos is the only one that matters here (the
// view vector the rim is built from); the rest is padding to reach it and to
// reach nothing after it.
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

// Density of the sheet seen face-on, and at a grazing angle. See the header on
// why the face-on figure is not the small number it looks like it should be.
const float BODY_ALPHA = 0.46;
const float RIM_ALPHA  = 0.95;
// Exponent of the grazing-angle ramp, and the single most important number
// here now that the rim is the whole outline. Higher is a thinner, sharper
// edge -- raise it if the ghost reads as a solid blob, lower it if the glow
// creeps too far up the body and swallows the shape.
const float RIM_POWER  = 2.6;

// Below this, the fragment is not drawn at all rather than drawn invisibly --
// see the depth-write note in the header.
const float ALPHA_CUTOFF = 0.015;

// Emission of the body and of the rim, before the tint. The rim clears
// BLOOM_THRESHOLD once alpha has scaled it, which is what puts a halo on the
// silhouette -- the model's tips included, they are nothing but rim; the body
// deliberately does not, so a ghost across the room glows at its edge instead
// of turning into a lamp.
//
// The rim figure is bounded from above by COLOUR, not by taste: pushed far
// enough it saturates all three channels and the edge comes out white, which
// throws the tint away exactly where the ghost is brightest. 2.35 keeps the
// blue through the tone map.
const float BODY_EMISSION = 0.60;
const float RIM_EMISSION  = 2.35;

// Where the tint goes during a chase, and how much brighter. Kept clear of the
// focus glow's gold and of the flames' hunt violet so the three cues stay
// distinguishable -- this one is the only one that means "it has seen you".
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

// 3 octaves rather than Flame.frag's 4: this field is stretched over a whole
// body instead of a hand-sized flame, so the finest octave lands below a pixel
// at any distance the ghost is actually seen from and only costs.
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
    // abs(), not max(..., 0): a sheet has no inside, and a fold whose normal
    // points away from the eye should thicken exactly as much as one that
    // points towards it. Without the abs those folds go to zero density and
    // punch holes in the silhouette.
    float facing = abs(dot(N, V));

    // 1. Fresnel rim. The outline of the ghost, and nothing else decides it.
    float rim = pow(1.0 - facing, RIM_POWER);
    float alpha = mix(BODY_ALPHA, RIM_ALPHA, rim);

    float chase = clamp(ubo.F0, 0.0, 1.0);

    // OBJECT SPACE, by transposing the world matrix's rotation rather than
    // inverting it. Valid because a ghost's Wm is translate * rotateY and
    // nothing else -- no scale anywhere on these instances in scene.json (see
    // the Wm assignment in main.cpp's ghost loop), so mat3(mMat) is orthonormal
    // and its inverse is its transpose. A general inverse() per fragment would
    // also work and would cost far more for a matrix that is never general.
    //
    // Only the noise needs it, and it needs it so the mottling is glued to the
    // ghost: walking and turning has to carry the field along instead of
    // sliding the model through something standing still in the room.
    vec3 rel = fragPos - ubo.mMat[3].xyz;
    vec3 objPos = vec3(dot(rel, ubo.mMat[0].xyz),
                       dot(rel, ubo.mMat[1].xyz),
                       dot(rel, ubo.mMat[2].xyz));

    // 2. Noise, on the BODY only. The (1 - rim) weight is what protects the
    // silhouette: at the edge the mottling is fully faded out, so no amount of
    // noise can eat a bite out of the outline or out of the model's tips.
    // Static -- see the header.
    float flow = fbm(vec2(objPos.x * 1.7 + objPos.z * 1.7, objPos.y * 1.15));
    alpha *= mix(1.0, mix(0.72, 1.20, flow), (1.0 - rim) * 0.7);

    // The face. The texture is flat grey with black eyes and a mouth painted on
    // it, so its DARK texels are the only information in it, and they mean the
    // opposite of what a density map would: a painted mark is the one part of a
    // ghost that should read as solid, not as a thin spot. So the mask makes
    // those marks the densest thing on the model and, further down, stops them
    // emitting -- black features floating in the glow, which is what the mesh's
    // author drew.
    float texLum = dot(texture(albedoMap, fragUV).rgb, vec3(0.299, 0.587, 0.114));
    float faceMask = 1.0 - smoothstep(0.16, 0.46, texLum);
    alpha = mix(alpha, max(alpha, 0.94), faceMask);

    alpha = clamp(alpha, 0.0, 1.0);
    if(alpha < ALPHA_CUTOFF) {
        discard;
    }

    // 3. Emission. Unlit: the tint, brightened at the rim, then pushed towards
    // the hunt colour by however far into a chase this ghost is.
    vec3 tint = mix(ubo.mS, HUNT_TINT, chase);
    float emission = mix(BODY_EMISSION, RIM_EMISSION, rim) * mix(1.0, HUNT_EMISSION, chase);

    // A cool core under the tint, strongest where the sheet is thinnest. Two
    // colours rather than one is what stops the ghost reading as flat: the
    // middle is a different temperature from the edge, the way anything
    // genuinely translucent is.
    vec3 core = tint * 0.35 + vec3(0.05, 0.09, 0.14);
    vec3 color = mix(core, tint, rim) * emission;
    // The painted features, cut out of the light.
    color *= 1.0 - 0.93 * faceMask;

    // NOT premultiplied: the blend is srcAlpha * src + (1 - srcAlpha) * dst
    // (Starter.hpp's transparent path), so the alpha above already scales this
    // colour once. Multiplying it in here would square that and leave the body
    // black.
    outColor = vec4(color, alpha);
}
