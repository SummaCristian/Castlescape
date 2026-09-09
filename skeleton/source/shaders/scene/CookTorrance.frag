// FRAGMENT SHADER: runs once per pixel, decides that pixel's color.
//
//   in fragPos/fragNorm/fragUV  from PosNormUV.vert, interpolated per pixel
//   set 0 (gubo)                per frame: camera position and every light
//   set 1 (ubo)                 per object: its matrices and material
//   albedoMap                   the object's texture
//
// main(), in order: read the base color from the texture; for each light work
// out how much reaches this pixel and how much bounces to the camera (the
// BRDF); add the ambient term for light that arrived after bouncing off other
// surfaces; write it out raw (Composite.frag does the range compression).
//
// The theory is in notes.md; comments here only say why the code is shaped
// the way it is.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// MAX_LIGHTS and LIGHT_*, shared with SceneLights.hpp. Found via glslc -I.
#include "custom/LightConstants.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Light-model quantities from the slides (roughly rho, F0, N, L, V, h, D, G, F
// there) are named descriptively here instead: diffuseColor, specularColor,
// normal, lightDir, viewDir, halfVector, distributionTerm, geometryTerm,
// fresnelTerm. The BRDF math itself is unchanged, see notes.md for the theory.
// Must match the block in PosNormUV.vert field for field.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 specularColor;
    float roughness;    // rho on the slides. 0 = mirror, 1 = matte
    float F0;           // reflectance head-on
    float diffuseShare; // specular gets (1 - diffuseShare)
    int flatNormals;      // 1: ignore the vertex normal, use the face's own
    int interiorAmbient;  // 1: ambient as if the surface were vertical
    float time;           // unused here; declared to match the Flame block (shared DSLlocal)
    // This model's share of ambient, overriding gubo.ambientWeight. Negative
    // means "no override", the default. See ambientShare() below.
    float ambientWeight;
    // 0..1 focus-glow strength, set per-instance in main.cpp when this is the
    // object the crosshair is aimed at. Used near the end of main().
    float glow;
    // 1: shade as a METAL. In main(): the diffuse term goes (diffuseShare forced to 0),
    // and the indirect term becomes metalAmbient() -- a reflection of the room
    // -- instead of the hemisphere times albedo. See Material::metallic.
    int metallic;
} ubo;

layout(binding = 1, set = 1) uniform sampler2D albedoMap;

struct Light {
    vec3 pos;       // point/spot only
    float g;        // distance at which the light is exactly `color`
    vec3 dir;       // direct: travel direction. spot: aim direction
    float beta;     // decay exponent: 0 constant, 1 linear, 2 quadratic
    vec3 color;
    float cosIn;    // spot: cosine of the half inner angle
    float cosOut;   // spot: cosine of the half outer angle
    int type;
    // -1: unshadowed, or a light past NUM_SHADOW_CUBES slots. Else the cube
    // array slot (a point light only, see shadowFactor() below) holding this
    // light's shadow map. Set by SceneLights from lights.json's
    // "castsShadow", see the struct comment there.
    int shadowIndex;
};

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    int debugFlags;      // LIGHT_DEBUG_* bits, set by the cheat menu
    float time;          // seconds since startup, unused here
    // All four ride the padding before lights[]:
    float ambientWeight;  // scene default ambient share, 0..1; see ambientShare()
    float ambientBounce;  // share of a point/spot's radiance that returns as indirect light
    float fogDensity;     // distance fog, from GEOM_CULL_CONE_DIST in main.cpp; used at the end of main()
    Light lights[MAX_LIGHTS];
} gubo;

// Shadow sampling, set 2: its own descriptor set because it belongs to
// neither "once a frame" (set 0) nor "once an object" (set 1) -- it's once
// per SHADOW-CASTING LIGHT. NUM_SHADOW_CUBES slots, one real 6-face cube per
// point light (the torches, see LightConstants.glsl). No matrix needed here:
// a samplerCube lookup is by DIRECTION, and the light's own position
// (gubo.lights[i].pos) is already available where shadowFactor() is called.
//
// SEPARATE sampler bindings rather than one binding declared as an array:
// Scene::init's descriptor-pool accounting (Scene.hpp, the loop that
// does `texturesInPool += 1` per binding) counts bindings, not the
// descriptors an array binding actually needs, and every existing binding in
// this project has count 1. An array binding would silently under-reserve
// the pool. Ordinary one-per-map bindings sidestep that instead of relying on
// a path nothing else here exercises.
//
// shadowCube0..18: the dynamic pool (main.cpp's
// dynamicShadowSlotBase..HAND_TORCH_SHADOW_INDEX, currently the whole 0..18
// range) -- whichever wall/dl/candle point light is currently nearest the
// player, reassigned at runtime by updateDynamicShadowSlots(). None of these
// belongs to a particular torch; which torch's cube map lands in which
// binding changes as the player moves.
layout(binding = 0, set = 2) uniform samplerCube shadowCube0;
layout(binding = 1, set = 2) uniform samplerCube shadowCube1;
layout(binding = 2, set = 2) uniform samplerCube shadowCube2;
layout(binding = 3, set = 2) uniform samplerCube shadowCube3;
layout(binding = 4, set = 2) uniform samplerCube shadowCube4;
layout(binding = 5, set = 2) uniform samplerCube shadowCube5;
layout(binding = 6, set = 2) uniform samplerCube shadowCube6;
layout(binding = 7, set = 2) uniform samplerCube shadowCube7;
layout(binding = 8, set = 2) uniform samplerCube shadowCube8;
layout(binding = 9, set = 2) uniform samplerCube shadowCube9;
layout(binding = 10, set = 2) uniform samplerCube shadowCube10;
layout(binding = 11, set = 2) uniform samplerCube shadowCube11;
layout(binding = 12, set = 2) uniform samplerCube shadowCube12;
layout(binding = 13, set = 2) uniform samplerCube shadowCube13;
layout(binding = 14, set = 2) uniform samplerCube shadowCube14;
layout(binding = 15, set = 2) uniform samplerCube shadowCube15;
layout(binding = 16, set = 2) uniform samplerCube shadowCube16;
layout(binding = 17, set = 2) uniform samplerCube shadowCube17;
layout(binding = 18, set = 2) uniform samplerCube shadowCube18;
layout(binding = 19, set = 2) uniform samplerCube shadowCube19;
layout(binding = 20, set = 2) uniform samplerCube shadowCube20;
layout(binding = 21, set = 2) uniform samplerCube shadowCube21;
layout(binding = 22, set = 2) uniform samplerCube shadowCube22;
layout(binding = 23, set = 2) uniform samplerCube shadowCube23;
layout(binding = 24, set = 2) uniform samplerCube shadowCube24;
layout(binding = 25, set = 2) uniform samplerCube shadowCube25;
layout(binding = 26, set = 2) uniform samplerCube shadowCube26;
layout(binding = 27, set = 2) uniform samplerCube shadowCube27;
layout(binding = 28, set = 2) uniform samplerCube shadowCube28;
layout(binding = 29, set = 2) uniform samplerCube shadowCube29;
layout(binding = 30, set = 2) uniform samplerCube shadowCube30;
layout(binding = 31, set = 2) uniform samplerCube shadowCube31;	// the held torch, fixed

// Same for the cube maps, sampled by direction, with one difference: this
// returns FOUR taps, not one. The chain below is a linear walk of up to 32
// comparisons to pick a binding, and it's the walk, not the fetch, that costs
// -- so the PCF this feeds takes all four taps INSIDE the resolved branch,
// paying for the walk once. NUM_SHADOW_CUBES is 32; add/remove a case by hand.
#define CUBE_TAP4(s) vec4(texture(s, d0).r, texture(s, d1).r, \
                          texture(s, d2).r, texture(s, d3).r)
vec4 sampleShadowCube4(int idx, vec3 d0, vec3 d1, vec3 d2, vec3 d3) {
    if(idx == 0) return CUBE_TAP4(shadowCube0);
    if(idx == 1) return CUBE_TAP4(shadowCube1);
    if(idx == 2) return CUBE_TAP4(shadowCube2);
    if(idx == 3) return CUBE_TAP4(shadowCube3);
    if(idx == 4) return CUBE_TAP4(shadowCube4);
    if(idx == 5) return CUBE_TAP4(shadowCube5);
    if(idx == 6) return CUBE_TAP4(shadowCube6);
    if(idx == 7) return CUBE_TAP4(shadowCube7);
    if(idx == 8) return CUBE_TAP4(shadowCube8);
    if(idx == 9) return CUBE_TAP4(shadowCube9);
    if(idx == 10) return CUBE_TAP4(shadowCube10);
    if(idx == 11) return CUBE_TAP4(shadowCube11);
    if(idx == 12) return CUBE_TAP4(shadowCube12);
    if(idx == 13) return CUBE_TAP4(shadowCube13);
    if(idx == 14) return CUBE_TAP4(shadowCube14);
    if(idx == 15) return CUBE_TAP4(shadowCube15);
    if(idx == 16) return CUBE_TAP4(shadowCube16);
    if(idx == 17) return CUBE_TAP4(shadowCube17);
    if(idx == 18) return CUBE_TAP4(shadowCube18);
    if(idx == 19) return CUBE_TAP4(shadowCube19);
    if(idx == 20) return CUBE_TAP4(shadowCube20);
    if(idx == 21) return CUBE_TAP4(shadowCube21);
    if(idx == 22) return CUBE_TAP4(shadowCube22);
    if(idx == 23) return CUBE_TAP4(shadowCube23);
    if(idx == 24) return CUBE_TAP4(shadowCube24);
    if(idx == 25) return CUBE_TAP4(shadowCube25);
    if(idx == 26) return CUBE_TAP4(shadowCube26);
    if(idx == 27) return CUBE_TAP4(shadowCube27);
    if(idx == 28) return CUBE_TAP4(shadowCube28);
    if(idx == 29) return CUBE_TAP4(shadowCube29);
    if(idx == 30) return CUBE_TAP4(shadowCube30);
    return CUBE_TAP4(shadowCube31);
}

// The torch path: one samplerCube lookup by direction, compared against the
// LINEAR distance to the light (ShadowCube.frag stores distance, not
// projective depth). A cube map covers every direction, so this always has
// an answer -- no "covered" out-parameter, no fallback slot.
//
// A plain lit/unlit test reads as a razor-sharp edge on a wall. The softening
// is four-tap PCF: sampleShadowCube4() reads the map at four directions
// around the lookup and this averages the verdicts. The kernel is a few
// texels wide and rotated, so its taps straddle the true silhouette and
// widening it costs no contact -- unlike softening from a single sample,
// which can only spill to the LIT side and left a bright strip between an
// object and its own shadow.
//
// The bias is computed HERE, not handed in, because sizing it honestly needs
// `dist`. It only covers floating-point noise: the across-texel slope error
// that a bias would otherwise fight is charged where it belongs, baked into
// the stored distance by ShadowCube.frag from the occluder's own tilt (see
// its header). A large bias, or a normal offset instead, each buys its own
// artefact -- light leaking through a closed door, or peter-panning where a
// shadow detaches from its base.
//
float shadowFromCube(int idx, vec3 pos, vec3 normal, vec3 lightPos, float NdotL) {
    // Depth slack: floating-point noise only. Front-face culling in the
    // capture keeps this surface out of its own map, so there is no acne left
    // to cover -- just a few ULPs between the distance computed here and the
    // one ShadowCube.frag computed from an interpolated position.
    const float CUBE_BIAS_MIN = 0.0015;
    const float CUBE_BIAS_MAX = 0.004;
    const float CUBE_SOFT_MIN = 0.002;
    // No normal offset: it only mattered when the surface was in its own map,
    // and it cost a lateral shift of the shadow edge.
    const float NORMAL_OFFSET_TEXELS = 0.0;
    const float NORMAL_OFFSET_MAX = 0.0;
    // PCF kernel radius in cube-face texels. The only thing that sets how wide
    // a shadow edge reads, and the one place where widening it doesn't eat the
    // contact.
    const float PCF_KERNEL_TEXELS = 3.0;

    float rawDist = length(pos - lightPos);
    float texelWorld = 2.0 * rawDist / float(SHADOW_CUBE_RES);

    // sin of the incidence angle. cosI is floored: a near edge-on fragment
    // gets almost nothing from this light anyway (the BRDF's NdotL), so the
    // floor just keeps sinI off 1.
    float cosI = max(NdotL, 0.15);
    float sinI = sqrt(1.0 - cosI * cosI);

    float offset = min(texelWorld * NORMAL_OFFSET_TEXELS * sinI, NORMAL_OFFSET_MAX);
    vec3 samplePos = pos + normal * offset;

    vec3 toFrag = samplePos - lightPos;
    float dist = length(toFrag);

    float bias = clamp(0.5 * texelWorld, CUBE_BIAS_MIN, CUBE_BIAS_MAX);
    // Per-tap band, not the edge softness (the kernel's job): just keeps one
    // tap from being a hard step, so the four average continuously.
    float softEdge = CUBE_SOFT_MIN;

    // The four tap directions: the lookup direction pushed sideways in the
    // plane perpendicular to it. An offset of `r` world units at right angles
    // to a length-dist vector turns it by r/dist, and one texel subtends
    // texelWorld/dist, so measuring the offset in texelWorld measures it in
    // face texels -- the bias's currency, independent of light distance.
    vec3 axis = abs(toFrag.y) < 0.99 * dist ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(toFrag, axis));
    vec3 B = normalize(cross(toFrag, T));
    float r = PCF_KERNEL_TEXELS * texelWorld;

    // Rotated grid, not a 2x2 box: a box's four points share two x and two y,
    // so a straight edge steps in thirds. Rotated, all four cross it apart.
    vec3 d0 = toFrag + r * ( 0.33 * T + 1.00 * B);
    vec3 d1 = toFrag + r * ( 1.00 * T - 0.33 * B);
    vec3 d2 = toFrag + r * (-0.33 * T - 1.00 * B);
    vec3 d3 = toFrag + r * (-1.00 * T + 0.33 * B);

    // One `dist` for all four: each tap's direction is longer than toFrag by
    // r^2/(2*dist), under a micrometre at three texels, far below the bias.
    vec4 gaps = vec4(dist) - sampleShadowCube4(idx, d0, d1, d2, d3);
    vec4 lit = vec4(1.0) - clamp((gaps - vec4(bias)) / softEdge, 0.0, 1.0);

    return dot(lit, vec4(0.25));
}

// 1.0: fully lit. 0.0: this light's shadow map says something else is closer
// to the light than `pos` is, i.e. `pos` is in shadow. shadowIndex < 0 skips
// the lookup and lights unconditionally, which is why a light without a slot
// leaks through every wall it reaches. Only LIGHT_POINT ever has a real
// shadowIndex (see LightData::shadowIndex in SceneLights.hpp) -- a direct or
// spot light always lights unconditionally through this same early-out.
//
// NdotL scales the cube path's normal offset, which is why normal has to come
// along too.
float shadowFactor(int shadowIndex, int type, vec3 pos, vec3 normal, vec3 lightPos, float NdotL) {
    // Shadows off (cheat menu). Reads gubo.debugFlags directly, since
    // debugOn() is declared further down.
    if(shadowIndex < 0 || (gubo.debugFlags & LIGHT_DEBUG_NO_SHADOWS) != 0) {
        return 1.0;
    }

    // type is always LIGHT_POINT here: a direct or spot light never gets a
    // shadowIndex >= 0 (SceneLights::init), so the check above already
    // caught it. Bias, softening band and normal offset all live inside
    // shadowFromCube() now: they are derived from the distance to the
    // light, which is the one thing this function doesn't have and that
    // one computes anyway. normal and NdotL are what scale them, see there.
    return shadowFromCube(shadowIndex, pos, normal, lightPos, NdotL);
}

// Whether a LightConstants.glsl debug view is on. Off in a normal frame, so
// this is a uniform branch -- the cheap kind on a GPU.
bool debugOn(int flag) {
    return (gubo.debugFlags & flag) != 0;
}

// LIGHT_DEBUG_HEATMAP's ramp: black through blue, green, yellow, to red. Log
// compression first, since the input is unclamped HDR radiance -- a linear
// ramp would read as solid red across most of a lit room.
vec3 heatmapRamp(float intensity) {
    float x = clamp(log(1.0 + intensity) / log(4.0), 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 0.0);
    vec3 c1 = vec3(0.0, 0.0, 1.0);
    vec3 c2 = vec3(0.0, 1.0, 0.0);
    vec3 c3 = vec3(1.0, 1.0, 0.0);
    vec3 c4 = vec3(1.0, 0.0, 0.0);
    if(x < 0.25) return mix(c0, c1, x / 0.25);
    if(x < 0.5)  return mix(c1, c2, (x - 0.25) / 0.25);
    if(x < 0.75) return mix(c2, c3, (x - 0.5) / 0.25);
    return mix(c3, c4, (x - 0.75) / 0.25);
}

const float PI = 3.14159265359;

// The indirect term for a METAL. A metal has no diffuse component, so nearly
// everything the eye gets off a lock or chain is a REFLECTION of the room --
// which left the chains near-black and the padlock reading as orange plastic
// before this existed. There is no environment to sample (no sun, no sky, no
// probe): a metal reflects the SAME torchlight bouncing off the walls around
// it, tinted through Fresnel instead of a diffuse wrap. `indirect` is that
// shared bounce value -- already summed over every point/spot light and
// shadow-tested in main() -- not an independently authored color, so a metal
// can never show ambient light the actual torches/candles didn't put there.
vec3 metalAmbient(vec3 normal, vec3 viewDir, vec3 specularColor, float roughness, float F0, vec3 indirect) {
    // Schlick on normal.viewDir. Ceiling is max(1 - roughness, F0), not 1.0: a rough
    // metal shouldn't turn mirror at the horizon and put a hard bright rim
    // around a tube.
    float NdotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    float fresnelTerm = F0 + (max(1.0 - roughness, F0) - F0) * pow(1.0 - NdotV, 5.0);

    // specularColor, not diffuseColor: for a metal the specular color IS the material's color.
    return indirect * specularColor * fresnelTerm;
}

// How much of this fragment's light is indirect, 0..1. Per-model if the
// material set one, the scene's default otherwise. This is E17's
// gubo.ambientLight, made per-model so a corridor and a courtyard can differ
// in one frame. The cheat gate sits HERE, downstream of both places a share
// can be authored, so a model with its own override still respects it. See
// LIGHT_DEBUG_NO_AMBIENT.
float ambientShare() {
    if(debugOn(LIGHT_DEBUG_NO_AMBIENT)) {
        return 0.0;
    }
    return ubo.ambientWeight >= 0.0 ? ubo.ambientWeight : gubo.ambientWeight;
}

// lx: direction towards the light. Constant for a direct light, per-fragment
// for the others, which is why a point light wraps around an object.
vec3 lightDirection(Light lt, vec3 pos) {
    if(lt.type == LIGHT_DIRECT) {
        return -lt.dir;
    }
    return normalize(lt.pos - pos);
}

// Radiance arriving, after decay and the spot cone. L09 s.23 and s.33.
vec3 lightRadiance(Light lt, vec3 pos) {
    if(lt.type == LIGHT_DIRECT) {
        return lt.color;
    }

    float dist = length(lt.pos - pos);

    // A plain divide guard still lets g/dist blow up right next to the source,
    // so radiance stays flat over most of a room then spikes to white in the
    // last stretch -- what a wall shows as you walk the held torch up to it.
    // NEAR_RADIUS is a smooth floor on the effective distance (sqrt never
    // drops below r), about the flame's own size, so the curve flattens near
    // the light instead of diverging.
    const float NEAR_RADIUS = 0.4;
    float distSoft = sqrt(dist * dist + NEAR_RADIUS * NEAR_RADIUS);
    vec3 radiance = lt.color * pow(lt.g / distSoft, lt.beta);

    if(lt.type == LIGHT_SPOT) {
        // lt.dir is where the lamp POINTS; the slides write this against lx,
        // hence the negation.
        float cosAngle = dot(-lightDirection(lt, pos), lt.dir);
        radiance *= clamp((cosAngle - lt.cosOut) / (lt.cosIn - lt.cosOut), 0.0, 1.0);
    }

    return radiance;
}

// distributionTerm: fraction of microfacets oriented along halfVector. GGX, E06 s.45.
float distributionGGX(vec3 normal, vec3 halfVector, float roughness) {
    float a2 = roughness * roughness;
    float NdotH = clamp(dot(normal, halfVector), 0.0, 1.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// geometryTerm: microfacets shadowing each other. Parameter-free form, E06 s.47.
// Without it rough surfaces blow out at grazing angles.
float geometricTerm(vec3 normal, vec3 halfVector, vec3 lightDir, vec3 viewDir) {
    float NdotH = clamp(dot(normal, halfVector), 0.0, 1.0);
    float NdotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    float NdotL = clamp(dot(normal, lightDir), 0.0, 1.0);
    float VdotH = max(dot(viewDir, halfVector), 0.0001);

    return min(1.0, min(2.0 * NdotH * NdotV / VdotH,
                        2.0 * NdotH * NdotL / VdotH));
}

// fresnelTerm: fraction reflected instead of transmitted, F0 head-on rising to 1 at the
// horizon. Schlick, E06 s.46. The 5 is his fit, not a derivation.
float fresnelSchlick(vec3 viewDir, vec3 halfVector, float F0) {
    float VdotH = clamp(dot(viewDir, halfVector), 0.0, 1.0);
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// Cook-Torrance, E06 s.38-39:
//   fr = clamp(normal.lightDir) * (diffuseShare * diffuseColor + (1-diffuseShare) * specularColor * distributionTerm*fresnelTerm*geometryTerm / (4 * clamp(normal.lightDir) * clamp(normal.viewDir)))
// Diffuse and specular are interpolated by diffuseShare, not added -- adding both at full
// strength returns more light than came in. The 4*(normal.lightDir)*(normal.viewDir) is the
// microfacet-to-surface-area normalization.
vec3 BRDF(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 diffuseColor, vec3 specularColor, float roughness, float F0, float diffuseShare) {
    float NdotL = clamp(dot(normal, lightDir), 0.0, 1.0);
    float NdotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    vec3 halfVector = normalize(lightDir + viewDir);

    float distributionTerm = distributionGGX(normal, halfVector, roughness);
    float geometryTerm = geometricTerm(normal, halfVector, lightDir, viewDir);
    float fresnelTerm = fresnelSchlick(viewDir, halfVector, F0);

    // Denominator hits 0 at the silhouette; NdotL zeroes the result there
    // anyway, the guard just avoids an inf (inf * 0 is NaN).
    vec3 specular = specularColor * (distributionTerm * fresnelTerm * geometryTerm) / max(4.0 * NdotL * NdotV, 0.0001);

    // Lambert diffuse (E06 s.38). Clamped NdotL, so a face turned away
    // contributes 0 rather than eating into what another light put there.
    return NdotL * (diffuseShare * diffuseColor + (1.0 - diffuseShare) * specular);
}

// The tone map (L09 s.45) MOVED to Composite.frag, the last pass of the HDR
// chain: it squashes everything into [0,1], and the bloom pass works by
// finding pixels above 1, so compressing here would leave the flame no glow.
// This shader writes raw radiance into a float attachment; the Tone Mapping
// cheat still works, one pass later.

// ---------------------------------------------------------------------------
// Procedural grime for the interior metals (chains, padlock, key). They have
// tarnished underground and none of it is in the flat MGCG albedo. With no
// second UV set or dirt map, it comes from world position: a few octaves of
// value noise, so neighbouring chain links weather differently.
//
// main() uses it for roughness UP (grime scatters) and reflection tint DOWN
// (a filmed surface reflects less). Albedo is left alone -- a metal has no
// diffuse term to darken. Gated to metal && interiorAmbient, so it skips the
// outdoor lanterns.
float grimeHash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

// Value noise: hash the 8 corners of p's cell, smoothstep the fraction,
// trilinearly blend. Flame.frag's noise2, one dimension up.
float grimeNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = p - i;
    f = f * f * (3.0 - 2.0 * f);
    float n000 = grimeHash(i + vec3(0, 0, 0)), n100 = grimeHash(i + vec3(1, 0, 0));
    float n010 = grimeHash(i + vec3(0, 1, 0)), n110 = grimeHash(i + vec3(1, 1, 0));
    float n001 = grimeHash(i + vec3(0, 0, 1)), n101 = grimeHash(i + vec3(1, 0, 1));
    float n011 = grimeHash(i + vec3(0, 1, 1)), n111 = grimeHash(i + vec3(1, 1, 1));
    return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
               mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}

// 0 clean .. 1 filthy. The objects are ~1-3 units across, so 6/18/50 per unit
// put the coarse blotches at a few cm and the grain below. smoothstep keeps
// clean metal clean instead of laying a grey veil over everything.
float grime(vec3 worldPos) {
    float g = grimeNoise(worldPos *  6.0) * 0.6
            + grimeNoise(worldPos * 18.0) * 0.3
            + grimeNoise(worldPos * 50.0) * 0.1;
    return smoothstep(0.35, 0.80, g);
}
// ---------------------------------------------------------------------------

void main() {
    // Interpolation shortens the normal wherever the corner normals diverge.
    vec3 normal = normalize(fragNorm);

    // The MGCG models average vertex normals across hard edges, so a flat face
    // gets a gradient instead of one value. For those, derive the face normal
    // here: dFdx/dFdy of the world position are two vectors in the triangle's
    // plane, so their cross product is perpendicular to it. Its sign depends
    // on winding, so orient it against the vertex normal (unreliable in
    // magnitude, fine for which side is out).
    if(ubo.flatNormals == 1) {
        vec3 faceN = normalize(cross(dFdx(fragPos), dFdy(fragPos)));
        normal = dot(faceN, normal) < 0.0 ? -faceN : faceN;
    }

    // No sRGB conversion: the image view is R8G8B8A8_SRGB, so the sampler
    // already returns linear values.
    vec3 diffuseColor = texture(albedoMap, fragUV).rgb;

    // Debug view: the shading normal remapped to [0,1] (+X red, +Y green,
    // +Z blue). After the flatNormals block, so it shows the normal the
    // lighting actually used.
    if(debugOn(LIGHT_DEBUG_NORMALS)) {
        outColor = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }

    // Debug view: the texture alone, no lighting. Tells "no light reached"
    // apart from "the texture is black here".
    if(debugOn(LIGHT_DEBUG_UNLIT)) {
        outColor = vec4(diffuseColor, 1.0);
        return;
    }

    vec3 viewDir = normalize(gubo.eyePos - fragPos);

    // Debug view: incoming light intensity, ignoring albedo. Forcing diffuseColor/specularColor/diffuseShare
    // reuses the Lo loop below unchanged; only the result handling differs.
    bool heatmap = debugOn(LIGHT_DEBUG_HEATMAP);
    if(heatmap) {
        diffuseColor = vec3(1.0);
    }

    // Both views want the specular gone. Named once because the metal path
    // must stand down too: a "no specular" view of an all-specular surface has
    // to show the diffuse fallback.
    bool specularOff = debugOn(LIGHT_DEBUG_NO_SPECULAR) || heatmap;
    bool metal = ubo.metallic == 1 && !specularOff;

    // Forcing diffuseShare to 1 zeroes the specular term (the
    // highlights go, nothing else changes). A metal goes to 0 -- its free
    // electrons absorb the subsurface scatter a diffuse lobe comes from.
    // Forced here so the flag alone defines "this is a metal".
    float diffuseShare = specularOff ? 1.0 : (metal ? 0.0 : ubo.diffuseShare);

    // Grime, interior metals only. Everyone else has g == 0, so both mixes
    // are the identity. Where it applies, a dirty patch roughens the surface
    // and mutes the reflection tint.
    //
    // The wrought-iron chains wear it fully, the brass padlock and key much
    // less. No per-model flag: the specular color is the tell -- brass is warm
    // (specularColor.b well under specularColor.r), steel is near neutral.
    float warmth = ubo.specularColor.b / max(ubo.specularColor.r, 1e-4);          // ~0.46 brass, ~1.04 steel
    float brassness = 1.0 - smoothstep(0.6, 0.95, warmth);  // 1 brass, 0 steel
    // Two knobs: grimeScale drops the overall bite; the pow() above 1 for
    // brass crushes the mid-grey coverage so only the hotspots survive, which
    // breaks up the soft blob instead of just fading it.
    float grimeScale = mix(1.0, 0.30, brassness);
    float g = grime(fragPos);
    g = pow(g, mix(1.0, 2.5, brassness)) * grimeScale;
    g = (metal && ubo.interiorAmbient == 1) ? g : 0.0;
    float roughG = mix(ubo.roughness, min(ubo.roughness * 2.0 + 0.20, 0.95), g);
    vec3  mSG    = ubo.specularColor * mix(1.0, 0.40, g);

    // Rendering equation: sum of radiance times BRDF over the sources, each
    // term zeroed by shadowFactor() where that light doesn't reach. The
    // hemispheric ambient below is exempt on purpose -- shadow mapping blocks
    // only a light's DIRECT contribution, or a shadow would be a hole into
    // black. The per-light bounce in the loop IS shadowed, for the reason at
    // its own site.
    //
    // LIGHT_ATTEN_EPS: below this a light's radiance is under 1 LSB of the
    // final 8-bit image, before the BRDF even runs. Point/spot lights are only
    // CPU-culled by distance to the CAMERA, so one that passed that cull can
    // still be near-zero at a far fragment; this catches that per pixel.
    const float LIGHT_ATTEN_EPS = 1e-3;

    vec3 Lo = vec3(0.0);
    // The point/spot lights' INDIRECT contribution, spent below out of the
    // ambient share. In this loop, not a function of its own, so it can reuse
    // `radiance`, `lightDir` and the visibility -- recomputing those would double the
    // length()/pow() and shadow fetch in the hottest loop. It skips the BRDF,
    // which bounced light has no lobe for.
    vec3 bounce = vec3(0.0);
    for(int i = 0; i < gubo.lightCount; i++) {
        vec3 radiance = lightRadiance(gubo.lights[i], fragPos);

        // Cheap reject before the BRDF's pow()s and shadowFactor()'s dependent
        // texture fetch. A direct light's radiance is constant, so this only
        // ever fires for decaying point/spot lights.
        if(max(radiance.r, max(radiance.g, radiance.b)) < LIGHT_ATTEN_EPS) {
            continue;
        }

        vec3 lightDir = lightDirection(gubo.lights[i], fragPos);
        // Same clamped dot the BRDF uses; shadowFactor scales its bias by it.
        float NdotL = clamp(dot(normal, lightDir), 0.0, 1.0);
        float vis = shadowFactor(gubo.lights[i].shadowIndex, gubo.lights[i].type, fragPos, normal, gubo.lights[i].pos, NdotL);
        Lo += radiance
            * BRDF(normal, lightDir, viewDir, diffuseColor, mSG, roughG, ubo.F0, diffuseShare)
            * vis;

        // Wrap-around diffuse, NOT the BRDF's clamped cosine: (dot + 1) / 2.
        // Bounced light arrives from most of the hemisphere, so it has no
        // terminator -- a face turned away from a torch is dimmer, not black.
        //
        // Direct lights are excluded on principle (a directional light has no
        // position for this wrap term to be relative to) even though none are
        // authored in this scene today -- see lights.json, the sun is gone.
        //
        // `vis` applies here too. It didn't originally, and a shadow near a
        // torch filled brightest at the torch end and faded -- the glow at the
        // start of a barrel's shadow, because the term never read the map. A
        // fragment the flame can't see now collects no indirect light at all
        // on a diffuse surface: there is no hemispheric floor left to fall
        // back on (CookTorrance.frag's ambient blend), which is the honest
        // look for a dungeon corridor with no torch in it.
        if(gubo.lights[i].type != LIGHT_DIRECT) {
            bounce += radiance * (dot(normal, lightDir) * 0.5 + 0.5) * vis;
        }
    }

    // E17's blend, not a sum: ambient is a SHARE of the light at this fragment
    // and the direct term gives up exactly what ambient takes. Summed (as
    // before) the hemispheric term was a brightness floor under every pixel --
    // no visibility term, so a sealed room collected as much indirect light as
    // the courtyard. Blended, the ambient never exceeds `aw` of the frame.
    //
    // Still not occlusion: a per-model authored guess at how enclosed a
    // surface is, until there's an AO map to bake (the MGCG pack ships albedo
    // only).
    //
    // The ONLY indirect light in this scene is torch/candle bounce -- no
    // sun, no sky, no separately authored "ambient" standing in for one.
    // `indirect` reuses each point/spot light's own radiance and shadow
    // visibility from the loop above, so a corridor with no torch in it
    // collects nothing, the way an unlit dungeon corridor actually looks.
    // Diffuse surfaces take it times albedo (diffuseColor); metals take the same value
    // through metalAmbient()'s Fresnel instead, so a metal can never show a
    // reflection color the actual torches/candles didn't put there.
    float aw = ambientShare();
    vec3 indirect = debugOn(LIGHT_DEBUG_NO_BOUNCE) ? vec3(0.0) : bounce * gubo.ambientBounce;
    vec3 ambient = metal ? metalAmbient(normal, viewDir, mSG, roughG, ubo.F0, indirect)
                         : indirect * diffuseColor;
    vec3 color = Lo * (1.0 - aw) + ambient * aw;

    if(heatmap) {
        float intensity = dot(color, vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(heatmapRamp(intensity), 1.0);
        return;
    }

    // Focus glow: a gold "magic field" aura on the silhouette of whichever
    // door/pickup/candle/torch instance the crosshair is aimed at (ubo.glow,
    // set per-instance in main.cpp). Purely a per-fragment color add on this
    // surface -- it casts no light on anything else. A Fresnel rim term
    // concentrates it at the silhouette, and `flow` rides a sine over world
    // position so it reads as travelling around the object rather than
    // pulsing in place.
    if(ubo.glow != 0.0) {
        // ubo.glow packs two things: rounded magnitude picks the category
        // color (1 Door gold, 2 Pickup blue/purple, 3 Candle orange,
        // 4 WallTorch yellow), a negative sign overrides it with red for
        // something disabled (a locked door with no key, a candle with
        // nothing to light it). Same aura, different color.
        const vec3 GLOW_COLOR_DOOR = vec3(1.0, 0.78, 0.25);
        const vec3 GLOW_COLOR_PICKUP = vec3(0.55, 0.35, 1.0);
        // Hotter and redder than the door's gold: a candle promises FIRE.
        const vec3 GLOW_COLOR_CANDLE = vec3(1.0, 0.45, 0.10);
        // Brighter and yellower still: a burning wall torch is what you light
        // your own torch from.
        const vec3 GLOW_COLOR_TORCH = vec3(1.0, 0.66, 0.22);
        const vec3 GLOW_COLOR_DISABLED = vec3(1.0, 0.15, 0.1);
        vec3 GLOW_COLOR;
        if(ubo.glow < 0.0) {
            GLOW_COLOR = GLOW_COLOR_DISABLED;
        } else if(ubo.glow > 3.5) {
            GLOW_COLOR = GLOW_COLOR_TORCH;
        } else if(ubo.glow > 2.5) {
            GLOW_COLOR = GLOW_COLOR_CANDLE;
        } else if(ubo.glow > 1.5) {
            GLOW_COLOR = GLOW_COLOR_PICKUP;
        } else {
            GLOW_COLOR = GLOW_COLOR_DOOR;
        }
        float glowStrength = (ubo.glow != 0.0) ? 1.0 : 0.0;
        float ndotv = clamp(dot(normal, viewDir), 0.0, 1.0);
        float edgeTerm = 1.0 - ndotv;

        // Wider than a plain pow(edgeTerm, 3.0): still not enough contrast on
        // small shiny props like the key, whose own highlights fight the rim.
        float rim = pow(edgeTerm, 2.2);

        // A thin near-black separator right at the silhouette (much sharper
        // power), darkening the surface BEFORE the gold is added. Without it a
        // reflective surface bounces a highlight through the gold band and the
        // two wash together; the darker edge separates the aura from the model
        // like an outline round a sticker.
        float edgeOutline = pow(edgeTerm, 12.0);
        color = mix(color, color * 0.15, edgeOutline * glowStrength);

        float flow = 0.5 + 0.5 * sin(dot(fragPos, vec3(1.3, 0.9, 1.1)) * 2.2 - ubo.time * 2.0);
        vec3 glowRaw = GLOW_COLOR * glowStrength * rim * flow * 0.9;

        // Composite.frag's tone map divides each pixel by its own luminance,
        // so a plain additive glow gets crushed where the surface is bright
        // and shows undimmed where it's dark -- backwards. Scaling the add by
        // (1 + pre-glow luminance) cancels that division to first order, so
        // the glow looks about equally bright in torchlight or pitch dark.
        // Unclamped HDR, so it still blooms through the post chain.
        float baseLuminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color += glowRaw * (1.0 + baseLuminance);
    }

    // Distance fog: fades toward black (the background clear color) with
    // distance from the eye, so the geometry cull (GEOM_CULL_* in main.cpp)
    // reads as things dissolving into haze rather than popping out mid-screen.
    // Exponential-SQUARED falloff stays near 1 close to the camera, where
    // clarity still matters, and bites in the back half where the cull needs
    // it. In HDR linear space, before the tone map, so it blends like a
    // physical haze.
    float fogDist = length(fragPos - gubo.eyePos);
    float fogFactor = exp(-pow(gubo.fogDensity * fogDist, 2.0));
    color = mix(vec3(0.0), color, fogFactor);

    // Linear, unclamped, into an R16G16B16A16_SFLOAT attachment: a surface lit
    // past 1.0 keeps saying so. Composite.frag tone maps it down at the end;
    // the bloom passes in between read the excess.
    outColor = vec4(color, 1.0);
}
