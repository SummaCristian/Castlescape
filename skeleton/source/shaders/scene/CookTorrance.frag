// FRAGMENT SHADER: per-pixel Cook-Torrance shading.
// main(): sample albedo; per light, BRDF + shadow visibility; add ambient
// bounce term; write raw HDR (Composite.frag range-compresses).
// Theory in notes.md; comments here explain code shape only.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// MAX_LIGHTS and LIGHT_*, shared with SceneLights.hpp.
#include "custom/LightConstants.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Slide names (rho, F0, N, L, V, h, D, G, F) mapped to descriptive names:
// diffuseColor, specularColor, normal, lightDir, viewDir, halfVector,
// distributionTerm, geometryTerm, fresnelTerm. Must match PosNormUV.vert.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 specularColor;
    float roughness;    // 0 = mirror, 1 = matte
    float F0;           // reflectance head-on
    float diffuseShare; // specular gets (1 - diffuseShare)
    int flatNormals;      // 1: use face normal instead of vertex normal
    int interiorAmbient;  // 1: ambient as if surface were vertical
    float time;           // unused; kept to match Flame block layout
    float ambientWeight;  // per-model ambient override; negative = no override
    float glow;            // 0..1 focus-glow strength (crosshair target)
    int metallic;          // 1: shade as metal, see metalAmbient()
} ubo;

layout(binding = 1, set = 1) uniform sampler2D albedoMap;

struct Light {
    vec3 pos;       // point/spot only
    float g;        // distance at which the light is exactly `color`
    vec3 dir;       // direct: travel direction. spot: aim direction
    float beta;     // decay exponent: 0 constant, 1 linear, 2 quadratic
    vec3 color;
    float cosIn;    // spot: cosine of half inner angle
    float cosOut;   // spot: cosine of half outer angle
    int type;
    // -1: unshadowed / past NUM_SHADOW_CUBES. Else the cube array slot
    // holding this point light's shadow map (set by SceneLights).
    int shadowIndex;
};

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    int debugFlags;      // LIGHT_DEBUG_* bits, cheat menu
    float time;          // unused here
    float ambientWeight;  // scene default ambient share, 0..1
    float ambientBounce;  // share of point/spot radiance returned as indirect
    float fogDensity;     // distance fog
    Light lights[MAX_LIGHTS];
} gubo;

// Shadow sampling, set 2: once-per-shadow-casting-light, its own set.
// NUM_SHADOW_CUBES slots, one 6-face cube per point light (torches).
// samplerCube lookup is by direction, no matrix needed.
//
// Separate bindings, not an array: descriptor-pool accounting (Scene::init)
// counts bindings not array size, so an array binding would under-reserve.
//
// shadowCube0..18: dynamic pool, reassigned at runtime to whichever
// wall/dl/candle point light is nearest the player (updateDynamicShadowSlots()).
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
layout(binding = 31, set = 2) uniform samplerCube shadowCube31;	// held torch, fixed

// Returns 4 taps (for PCF) per call. The if-chain resolves the binding once;
// paying the walk cost a single time instead of per-tap. NUM_SHADOW_CUBES=32.
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

// Torch shadow: one samplerCube lookup by direction, compared against the
// LINEAR distance to the light (ShadowCube.frag stores distance, not
// projective depth). Always has an answer, no fallback slot needed.
//
// 4-tap PCF (sampleShadowCube4) softens the lit/unlit edge; rotated kernel
// straddles the true silhouette without eating contact shadows.
//
// Bias computed here (needs `dist` to size honestly): covers only
// floating-point noise, since front-face culling keeps a surface out of its
// own shadow map (no acne to fight).
float shadowFromCube(int idx, vec3 pos, vec3 normal, vec3 lightPos, float NdotL) {
    // Depth slack: floating-point noise only (see header).
    const float CUBE_BIAS_MIN = 0.0015;
    const float CUBE_BIAS_MAX = 0.004;
    const float CUBE_SOFT_MIN = 0.002;
    // No normal offset: only mattered for self-shadowing, not needed now.
    const float NORMAL_OFFSET_TEXELS = 0.0;
    const float NORMAL_OFFSET_MAX = 0.0;
    // PCF kernel radius in cube-face texels: sets shadow edge softness.
    const float PCF_KERNEL_TEXELS = 3.0;

    float rawDist = length(pos - lightPos);
    float texelWorld = 2.0 * rawDist / float(SHADOW_CUBE_RES);

    // sin of incidence angle; cosI floored since an edge-on fragment gets
    // almost nothing from this light anyway (BRDF's NdotL).
    float cosI = max(NdotL, 0.15);
    float sinI = sqrt(1.0 - cosI * cosI);

    float offset = min(texelWorld * NORMAL_OFFSET_TEXELS * sinI, NORMAL_OFFSET_MAX);
    vec3 samplePos = pos + normal * offset;

    vec3 toFrag = samplePos - lightPos;
    float dist = length(toFrag);

    float bias = clamp(0.5 * texelWorld, CUBE_BIAS_MIN, CUBE_BIAS_MAX);
    // Per-tap band (not edge softness): keeps one tap from being a hard step.
    float softEdge = CUBE_SOFT_MIN;

    // Four tap directions: lookup pushed sideways in the perpendicular plane.
    // Offset measured in texelWorld units = face texels, independent of light distance.
    vec3 axis = abs(toFrag.y) < 0.99 * dist ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(toFrag, axis));
    vec3 B = normalize(cross(toFrag, T));
    float r = PCF_KERNEL_TEXELS * texelWorld;

    // Rotated grid, not a 2x2 box: box taps share x/y pairs so a straight
    // edge steps in thirds; rotated, all four cross it apart.
    vec3 d0 = toFrag + r * ( 0.33 * T + 1.00 * B);
    vec3 d1 = toFrag + r * ( 1.00 * T - 0.33 * B);
    vec3 d2 = toFrag + r * (-0.33 * T - 1.00 * B);
    vec3 d3 = toFrag + r * (-1.00 * T + 0.33 * B);

    // One `dist` shared by all four taps: error is negligible vs. the bias.
    vec4 gaps = vec4(dist) - sampleShadowCube4(idx, d0, d1, d2, d3);
    vec4 lit = vec4(1.0) - clamp((gaps - vec4(bias)) / softEdge, 0.0, 1.0);

    return dot(lit, vec4(0.25));
}

// 1.0 fully lit, 0.0 in shadow. shadowIndex < 0 skips the lookup (lights
// unconditionally) -- only LIGHT_POINT ever has a real shadowIndex.
float shadowFactor(int shadowIndex, int type, vec3 pos, vec3 normal, vec3 lightPos, float NdotL) {
    // Shadows off (cheat menu).
    if(shadowIndex < 0 || (gubo.debugFlags & LIGHT_DEBUG_NO_SHADOWS) != 0) {
        return 1.0;
    }

    // type is always LIGHT_POINT here (see SceneLights::init).
    return shadowFromCube(shadowIndex, pos, normal, lightPos, NdotL);
}

// Uniform branch (cheap on GPU): off in a normal frame.
bool debugOn(int flag) {
    return (gubo.debugFlags & flag) != 0;
}

// LIGHT_DEBUG_HEATMAP ramp: black->blue->green->yellow->red. Log compression
// first since input is unclamped HDR radiance.
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

// Indirect term for a metal: no diffuse, so a metal reflects the same
// torchlight bounce as everything else, tinted through Fresnel instead of a
// diffuse wrap. `indirect` is the shared bounce value from main()'s loop, so
// a metal never shows ambient light the torches/candles didn't put there.
vec3 metalAmbient(vec3 normal, vec3 viewDir, vec3 specularColor, float roughness, float F0, vec3 indirect) {
    // Schlick on N.V; ceiling max(1-roughness, F0) so a rough metal doesn't
    // turn mirror-bright at the horizon.
    float NdotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    float fresnelTerm = F0 + (max(1.0 - roughness, F0) - F0) * pow(1.0 - NdotV, 5.0);

    // specularColor, not diffuseColor: for a metal, specular color IS the material color.
    return indirect * specularColor * fresnelTerm;
}

// Share of this fragment's light that's indirect, 0..1. Per-model override,
// else scene default. Cheat gate sits here so an override still respects it.
float ambientShare() {
    if(debugOn(LIGHT_DEBUG_NO_AMBIENT)) {
        return 0.0;
    }
    return ubo.ambientWeight >= 0.0 ? ubo.ambientWeight : gubo.ambientWeight;
}

// Direction towards the light: constant for direct, per-fragment otherwise
// (why point lights wrap around objects).
vec3 lightDirection(Light lt, vec3 pos) {
    if(lt.type == LIGHT_DIRECT) {
        return -lt.dir;
    }
    return normalize(lt.pos - pos);
}

// Radiance arriving after decay and spot cone. L09 s.23, s.33.
vec3 lightRadiance(Light lt, vec3 pos) {
    if(lt.type == LIGHT_DIRECT) {
        return lt.color;
    }

    float dist = length(lt.pos - pos);

    // NEAR_RADIUS: smooth floor on effective distance (~flame size), so
    // radiance doesn't spike to white right next to the source.
    const float NEAR_RADIUS = 0.4;
    float distSoft = sqrt(dist * dist + NEAR_RADIUS * NEAR_RADIUS);
    vec3 radiance = lt.color * pow(lt.g / distSoft, lt.beta);

    if(lt.type == LIGHT_SPOT) {
        // lt.dir points where the lamp aims; negated to compare against lx.
        float cosAngle = dot(-lightDirection(lt, pos), lt.dir);
        radiance *= clamp((cosAngle - lt.cosOut) / (lt.cosIn - lt.cosOut), 0.0, 1.0);
    }

    return radiance;
}

// GGX normal distribution: fraction of microfacets aligned with halfVector. E06 s.45.
float distributionGGX(vec3 normal, vec3 halfVector, float roughness) {
    float a2 = roughness * roughness;
    float NdotH = clamp(dot(normal, halfVector), 0.0, 1.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Geometry term: microfacet self-shadowing, parameter-free form. E06 s.47.
// Prevents rough surfaces blowing out at grazing angles.
float geometricTerm(vec3 normal, vec3 halfVector, vec3 lightDir, vec3 viewDir) {
    float NdotH = clamp(dot(normal, halfVector), 0.0, 1.0);
    float NdotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    float NdotL = clamp(dot(normal, lightDir), 0.0, 1.0);
    float VdotH = max(dot(viewDir, halfVector), 0.0001);

    return min(1.0, min(2.0 * NdotH * NdotV / VdotH,
                        2.0 * NdotH * NdotL / VdotH));
}

// Fresnel-Schlick approximation. E06 s.46 (5 is an empirical fit).
float fresnelSchlick(vec3 viewDir, vec3 halfVector, float F0) {
    float VdotH = clamp(dot(viewDir, halfVector), 0.0, 1.0);
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// Cook-Torrance BRDF, E06 s.38-39. Diffuse/specular interpolated by
// diffuseShare (not summed, else energy exceeds input).
vec3 BRDF(vec3 normal, vec3 lightDir, vec3 viewDir, vec3 diffuseColor, vec3 specularColor, float roughness, float F0, float diffuseShare) {
    float NdotL = clamp(dot(normal, lightDir), 0.0, 1.0);
    float NdotV = clamp(dot(normal, viewDir), 0.0, 1.0);
    vec3 halfVector = normalize(lightDir + viewDir);

    float distributionTerm = distributionGGX(normal, halfVector, roughness);
    float geometryTerm = geometricTerm(normal, halfVector, lightDir, viewDir);
    float fresnelTerm = fresnelSchlick(viewDir, halfVector, F0);

    // Guard avoids inf*0=NaN; NdotL already zeroes the result at the silhouette.
    vec3 specular = specularColor * (distributionTerm * fresnelTerm * geometryTerm) / max(4.0 * NdotL * NdotV, 0.0001);

    // Lambert diffuse (E06 s.38), clamped NdotL.
    return NdotL * (diffuseShare * diffuseColor + (1.0 - diffuseShare) * specular);
}

// Tone map moved to Composite.frag: bloom needs unclamped values >1 here.

// ---------------------------------------------------------------------------
// Procedural grime for interior metals (chains, padlock, key): no dirt map in
// the flat MGCG albedo, so it's derived from world position via value noise.
// Used for roughness up / reflection tint down. Gated to metal && interiorAmbient.
float grimeHash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

// Value noise, 3D version of Flame.frag's noise2: hash cell's 8 corners, trilerp.
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

// 0 clean .. 1 filthy. Multi-scale (6/18/50 per unit) for blotches + grain;
// smoothstep keeps clean metal clean instead of a grey veil everywhere.
float grime(vec3 worldPos) {
    float g = grimeNoise(worldPos *  6.0) * 0.6
            + grimeNoise(worldPos * 18.0) * 0.3
            + grimeNoise(worldPos * 50.0) * 0.1;
    return smoothstep(0.35, 0.80, g);
}
// ---------------------------------------------------------------------------

void main() {
    // Interpolation shortens the normal wherever corner normals diverge.
    vec3 normal = normalize(fragNorm);

    // MGCG models average normals across hard edges, so flat faces get a
    // gradient. Derive the true face normal from position derivatives
    // instead; orient against the vertex normal (sign only, from winding).
    if(ubo.flatNormals == 1) {
        vec3 faceN = normalize(cross(dFdx(fragPos), dFdy(fragPos)));
        normal = dot(faceN, normal) < 0.0 ? -faceN : faceN;
    }

    // No sRGB conversion: image view is R8G8B8A8_SRGB, sampler returns linear.
    vec3 diffuseColor = texture(albedoMap, fragUV).rgb;

    // Debug view: shading normal remapped to [0,1] (+X red, +Y green, +Z blue).
    if(debugOn(LIGHT_DEBUG_NORMALS)) {
        outColor = vec4(normal * 0.5 + 0.5, 1.0);
        return;
    }

    // Debug view: texture alone, no lighting.
    if(debugOn(LIGHT_DEBUG_UNLIT)) {
        outColor = vec4(diffuseColor, 1.0);
        return;
    }

    vec3 viewDir = normalize(gubo.eyePos - fragPos);

    // Debug view: incoming light intensity, ignoring albedo. Reuses the Lo
    // loop unchanged; only result handling differs.
    bool heatmap = debugOn(LIGHT_DEBUG_HEATMAP);
    if(heatmap) {
        diffuseColor = vec3(1.0);
    }

    // Metal path must also stand down for a "no specular" view.
    bool specularOff = debugOn(LIGHT_DEBUG_NO_SPECULAR) || heatmap;
    bool metal = ubo.metallic == 1 && !specularOff;

    // diffuseShare=1 zeroes specular; a metal forces 0 (no diffuse lobe).
    float diffuseShare = specularOff ? 1.0 : (metal ? 0.0 : ubo.diffuseShare);

    // Grime, interior metals only (g==0 elsewhere, mixes are identity).
    // Brass vs steel detected from specular color warmth, no per-model flag.
    float warmth = ubo.specularColor.b / max(ubo.specularColor.r, 1e-4);          // ~0.46 brass, ~1.04 steel
    float brassness = 1.0 - smoothstep(0.6, 0.95, warmth);  // 1 brass, 0 steel
    // grimeScale lowers overall bite; pow>1 for brass crushes mid-grey so
    // only hotspots survive.
    float grimeScale = mix(1.0, 0.30, brassness);
    float g = grime(fragPos);
    g = pow(g, mix(1.0, 2.5, brassness)) * grimeScale;
    g = (metal && ubo.interiorAmbient == 1) ? g : 0.0;
    float roughG = mix(ubo.roughness, min(ubo.roughness * 2.0 + 0.20, 0.95), g);
    vec3  mSG    = ubo.specularColor * mix(1.0, 0.40, g);

    // Rendering equation: sum of radiance*BRDF over sources, each zeroed by
    // shadowFactor(). Hemispheric ambient below is exempt on purpose (shadow
    // mapping blocks only a light's direct contribution).
    //
    // LIGHT_ATTEN_EPS: below this, radiance is under 1 LSB of the final
    // image; catches point/spot lights that passed CPU distance-cull but are
    // near-zero at this fragment.
    const float LIGHT_ATTEN_EPS = 1e-3;

    vec3 Lo = vec3(0.0);
    // Point/spot indirect contribution, spent below via ambient share. Kept
    // inline (not a function) to reuse `radiance`/`lightDir`/visibility from
    // this loop. Skips the BRDF, which bounced light has no lobe for.
    vec3 bounce = vec3(0.0);
    for(int i = 0; i < gubo.lightCount; i++) {
        vec3 radiance = lightRadiance(gubo.lights[i], fragPos);

        // Cheap reject before BRDF's pow()s and shadowFactor()'s texture fetch.
        if(max(radiance.r, max(radiance.g, radiance.b)) < LIGHT_ATTEN_EPS) {
            continue;
        }

        vec3 lightDir = lightDirection(gubo.lights[i], fragPos);
        float NdotL = clamp(dot(normal, lightDir), 0.0, 1.0);
        float vis = shadowFactor(gubo.lights[i].shadowIndex, gubo.lights[i].type, fragPos, normal, gubo.lights[i].pos, NdotL);
        Lo += radiance
            * BRDF(normal, lightDir, viewDir, diffuseColor, mSG, roughG, ubo.F0, diffuseShare)
            * vis;

        // Wrap-around diffuse (dot+1)/2, not BRDF's clamped cosine: bounced
        // light arrives from most of the hemisphere, no hard terminator.
        // Direct lights excluded (no position to wrap around).
        // `vis` applies here too, else a shadow near a torch glowed at its
        // own start; a fragment the flame can't see now gets no bounce either.
        if(gubo.lights[i].type != LIGHT_DIRECT) {
            bounce += radiance * (dot(normal, lightDir) * 0.5 + 0.5) * vis;
        }
    }

    // Blend, not sum: ambient is a SHARE of light at this fragment, direct
    // term gives up exactly what ambient takes (else a sealed room and a
    // courtyard got the same brightness floor).
    //
    // Not occlusion -- a per-model authored guess until an AO map exists.
    // Only indirect light in this scene is torch/candle bounce, reusing the
    // loop's radiance+visibility, so an unlit corridor collects nothing.
    // Diffuse takes it * albedo; metals take it through metalAmbient()'s Fresnel.
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

    // Focus glow: gold "magic field" aura on the crosshair-targeted instance
    // (ubo.glow, per-instance). Fresnel rim term concentrates it at the
    // silhouette; `flow` rides a sine over world position for travelling look.
    if(ubo.glow != 0.0) {
        // Rounded magnitude picks category color (1 Door, 2 Pickup, 3 Candle,
        // 4 WallTorch); negative sign overrides to red (disabled).
        const vec3 GLOW_COLOR_DOOR = vec3(1.0, 0.78, 0.25);
        const vec3 GLOW_COLOR_PICKUP = vec3(0.55, 0.35, 1.0);
        const vec3 GLOW_COLOR_CANDLE = vec3(1.0, 0.45, 0.10);
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

        // Wider than pow(edgeTerm,3): more contrast on small shiny props.
        float rim = pow(edgeTerm, 2.2);

        // Thin near-black separator at the silhouette (sharper power),
        // darkens the surface before gold is added so a reflective highlight
        // doesn't wash into the aura -- outline like round a sticker.
        float edgeOutline = pow(edgeTerm, 12.0);
        color = mix(color, color * 0.15, edgeOutline * glowStrength);

        float flow = 0.5 + 0.5 * sin(dot(fragPos, vec3(1.3, 0.9, 1.1)) * 2.2 - ubo.time * 2.0);
        vec3 glowRaw = GLOW_COLOR * glowStrength * rim * flow * 0.9;

        // Composite.frag's tone map divides by luminance, so a plain additive
        // glow would get crushed on bright surfaces. Scaling by
        // (1 + pre-glow luminance) cancels that to first order.
        float baseLuminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color += glowRaw * (1.0 + baseLuminance);
    }

    // Distance fog toward black, exp-squared falloff (stays near 1 close in,
    // bites in the back half where GEOM_CULL needs it). Linear HDR, pre-tonemap.
    float fogDist = length(fragPos - gubo.eyePos);
    float fogFactor = exp(-pow(gubo.fogDensity * fogDist, 2.0));
    color = mix(vec3(0.0), color, fogFactor);

    // Linear, unclamped: Composite.frag tone maps at the end, bloom reads the excess.
    outColor = vec4(color, 1.0);
}
