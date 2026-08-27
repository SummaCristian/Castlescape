// FRAGMENT SHADER. Runs on the GPU, once for every pixel of every triangle we
// draw, and its whole job is to decide that pixel's colour.
//
// Where its inputs come from:
//   in fragPos/fragNorm/fragUV  from PosNormUV.vert, the vertex shader. The GPU
//                               interpolates them across the triangle, so each
//                               pixel gets its own values.
//   set 0 (gubo)                written once per frame by main.cpp: the camera
//                               position and every light in the scene.
//   set 1 (ubo)                 written once per object by main.cpp: that
//                               object's matrices and its material.
//   albedoMap                   the object's texture.
//
// What main() does, in order:
//   1. read the base colour out of the texture
//   2. for each light: work out how much light reaches this pixel, and how much
//      of it bounces towards the camera (that second part is the BRDF)
//   3. add the ambient term, which stands in for light that arrived after
//      bouncing off other surfaces
//   4. squash the total into a displayable range and write it out
//
// The theory behind the formulas is in notes.md; the comments here only say why
// the code is shaped the way it is.

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

// MAX_LIGHTS and LIGHT_*, shared with SceneLights.hpp. Found via glslc -I.
#include "custom/LightConstants.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Light-model quantities keep the slides' names (mD, mS, N, L, V, h, D, G, F).
// Must match the block in PosNormUV.vert field for field.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 mS;          // specular color
    float roughness;  // rho on the slides. 0 = mirror, 1 = matte
    float F0;         // reflectance head-on
    float k;          // diffuse share, specular gets (1 - k)
    int flatNormals;      // 1: ignore the vertex normal, use the face's own
    int interiorAmbient;  // 1: ambient as if the surface were vertical
    // Unused here, declared to keep this block identical to the one Flame.vert
    // and Flame.frag see: both pipelines share DSLlocal and one C++ struct.
    float time;
    // This model's share of ambient, overriding gubo.ambientWeight. Negative
    // means "no override", which is the default: only the models that need a
    // different share from the scene's carry one. See ambientShare() below.
    float ambientWeight;
    // 0..1: this instance's focus-glow strength, set per-instance in
    // updateUniformBuffer() when it's the object the crosshair is aimed at.
    // See its use near the end of main() below.
    float glow;
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
    // -1: unshadowed. Every light in the current lights.json casts a shadow,
    // so nothing hits that path now, but it stays for lights added past
    // NUM_SHADOW_MAPS_2D/NUM_SHADOW_CUBES. Else the slot -- in the 2D array
    // for a direct/spot light, in the cube array for a point light, see
    // shadowFactor() below -- holding this light's shadow map. Set by
    // SceneLights from lights.json's "castsShadow", see the struct comment
    // there.
    int shadowIndex;
};

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    vec3 ambientUpper;   // indirect light from the sky
    vec3 ambientLower;   // indirect light bounced off the ground
    vec3 ambientDir;     // axis the two blend along, i.e. world up
    int debugFlags;      // LIGHT_DEBUG_* bits, set by the cheat menu
    float time;          // seconds since startup, unused here (see Flame.vert)
    // The scene's default share of ambient, 0..1. See ambientShare() below.
    // Rides in the padding before lights[], like debugFlags and time.
    float ambientWeight;
    Light lights[MAX_LIGHTS];
} gubo;

// Shadow sampling, set 2: its own descriptor set because it belongs to
// neither "once a frame" (set 0) nor "once an object" (set 1) -- it's once
// per SHADOW-CASTING LIGHT. Two separate families of slots now, one per
// projection kind a shadow can use (see LightConstants.glsl):
//   2D depth maps   NUM_SHADOW_MAPS_2D slots, direct/spot lights (the sun
//                   today). lightSpace is the view-projection matrix
//                   Shadow.vert rendered that map with.
//   cube maps       NUM_SHADOW_CUBES slots, one real 6-face cube per point
//                   light (the torches). No matrix needed here: a samplerCube
//                   lookup is by DIRECTION, and the light's own position
//                   (gubo.lights[i].pos) is already available where
//                   shadowFactor() is called.
//
// SEPARATE sampler bindings rather than one binding declared as an array:
// Scene::init's descriptor-pool accounting (Scene.hpp, the loop that
// does `texturesInPool += 1` per binding) counts bindings, not the
// descriptors an array binding actually needs, and every existing binding in
// this project has count 1. An array binding would silently under-reserve
// the pool. Ordinary one-per-map bindings sidestep that instead of relying on
// a path nothing else here exercises.
layout(binding = 0, set = 2) uniform ShadowUniformBufferObject {
    mat4 lightSpace[NUM_SHADOW_MAPS_2D];
} shadowUbo;

layout(binding = 1, set = 2) uniform sampler2D shadowMap2D_0;
layout(binding = 2, set = 2) uniform sampler2D shadowMap2D_1;

// shadowCube0..18: the dynamic pool (main.cpp's
// dynamicShadowSlotBase..HAND_TORCH_SHADOW_INDEX, currently the whole 0..18
// range) -- whichever wall/dl/candle point light is currently nearest the
// player, reassigned at runtime by updateDynamicShadowSlots(). None of these
// belongs to a particular torch; which torch's cube map lands in which
// binding changes as the player moves.
layout(binding = 3, set = 2) uniform samplerCube shadowCube0;
layout(binding = 4, set = 2) uniform samplerCube shadowCube1;
layout(binding = 5, set = 2) uniform samplerCube shadowCube2;
layout(binding = 6, set = 2) uniform samplerCube shadowCube3;
layout(binding = 7, set = 2) uniform samplerCube shadowCube4;
layout(binding = 8, set = 2) uniform samplerCube shadowCube5;
layout(binding = 9, set = 2) uniform samplerCube shadowCube6;
layout(binding = 10, set = 2) uniform samplerCube shadowCube7;
layout(binding = 11, set = 2) uniform samplerCube shadowCube8;
layout(binding = 12, set = 2) uniform samplerCube shadowCube9;
layout(binding = 13, set = 2) uniform samplerCube shadowCube10;
layout(binding = 14, set = 2) uniform samplerCube shadowCube11;
layout(binding = 15, set = 2) uniform samplerCube shadowCube12;
layout(binding = 16, set = 2) uniform samplerCube shadowCube13;
layout(binding = 17, set = 2) uniform samplerCube shadowCube14;
layout(binding = 18, set = 2) uniform samplerCube shadowCube15;
layout(binding = 19, set = 2) uniform samplerCube shadowCube16;
layout(binding = 20, set = 2) uniform samplerCube shadowCube17;
layout(binding = 21, set = 2) uniform samplerCube shadowCube18;
layout(binding = 22, set = 2) uniform samplerCube shadowCube19;
layout(binding = 23, set = 2) uniform samplerCube shadowCube20;
layout(binding = 24, set = 2) uniform samplerCube shadowCube21;
layout(binding = 25, set = 2) uniform samplerCube shadowCube22;
layout(binding = 26, set = 2) uniform samplerCube shadowCube23;
layout(binding = 27, set = 2) uniform samplerCube shadowCube24;
layout(binding = 28, set = 2) uniform samplerCube shadowCube25;
layout(binding = 29, set = 2) uniform samplerCube shadowCube26;
layout(binding = 30, set = 2) uniform samplerCube shadowCube27;
layout(binding = 31, set = 2) uniform samplerCube shadowCube28;
layout(binding = 32, set = 2) uniform samplerCube shadowCube29;
layout(binding = 33, set = 2) uniform samplerCube shadowCube30;
layout(binding = 34, set = 2) uniform samplerCube shadowCube31;	// the held torch, fixed

// Stands in for shadowMaps2D[idx], which the separate-bindings choice above
// rules out. NUM_SHADOW_MAPS_2D is 2 (LightConstants.glsl); if that ever
// changes, a case has to be added or removed here by hand.
float sampleShadowMap2D(int idx, vec2 uv) {
    if(idx == 0) return texture(shadowMap2D_0, uv).r;
    return texture(shadowMap2D_1, uv).r;
}

// Same idea for the cube maps, sampled by direction rather than by UV.
// NUM_SHADOW_CUBES is 20 (LightConstants.glsl: 19 dynamically-assigned slots
// shared by every wall/dl torch and candle, plus the held torch's own fixed
// last one); a case has to be added or removed here by hand if that changes.
float sampleShadowCube(int idx, vec3 dir) {
    if(idx == 0) return texture(shadowCube0, dir).r;
    if(idx == 1) return texture(shadowCube1, dir).r;
    if(idx == 2) return texture(shadowCube2, dir).r;
    if(idx == 3) return texture(shadowCube3, dir).r;
    if(idx == 4) return texture(shadowCube4, dir).r;
    if(idx == 5) return texture(shadowCube5, dir).r;
    if(idx == 6) return texture(shadowCube6, dir).r;
    if(idx == 7) return texture(shadowCube7, dir).r;
    if(idx == 8) return texture(shadowCube8, dir).r;
    if(idx == 9) return texture(shadowCube9, dir).r;
    if(idx == 10) return texture(shadowCube10, dir).r;
    if(idx == 11) return texture(shadowCube11, dir).r;
    if(idx == 12) return texture(shadowCube12, dir).r;
    if(idx == 13) return texture(shadowCube13, dir).r;
    if(idx == 14) return texture(shadowCube14, dir).r;
    if(idx == 15) return texture(shadowCube15, dir).r;
    if(idx == 16) return texture(shadowCube16, dir).r;
    if(idx == 17) return texture(shadowCube17, dir).r;
    if(idx == 18) return texture(shadowCube18, dir).r;
    if(idx == 19) return texture(shadowCube19, dir).r;
    if(idx == 20) return texture(shadowCube20, dir).r;
    if(idx == 21) return texture(shadowCube21, dir).r;
    if(idx == 22) return texture(shadowCube22, dir).r;
    if(idx == 23) return texture(shadowCube23, dir).r;
    if(idx == 24) return texture(shadowCube24, dir).r;
    if(idx == 25) return texture(shadowCube25, dir).r;
    if(idx == 26) return texture(shadowCube26, dir).r;
    if(idx == 27) return texture(shadowCube27, dir).r;
    if(idx == 28) return texture(shadowCube28, dir).r;
    if(idx == 29) return texture(shadowCube29, dir).r;
    if(idx == 30) return texture(shadowCube30, dir).r;
    return texture(shadowCube31, dir).r;
}

// The sun/spot path: unchanged from the single-perspective-map technique,
// just renamed now that it's not sharing a namespace with the torches'
// former (and now gone) second map.
float shadowFromMap2D(int idx, vec3 pos, float bias) {
    vec4 lightClip = shadowUbo.lightSpace[idx] * vec4(pos, 1.0);

    // Behind this map's camera. For a perspective matrix w is the view-space
    // distance in FRONT of the camera, so w <= 0 puts `pos` on the far side of
    // the plane through the light. The divide below would mirror such a point
    // back into the map's 0..1 range and sample a depth belonging to a
    // completely different direction, so it has to be caught here. The sun's
    // orthographic matrix always yields w = 1 and never trips this.
    if(lightClip.w <= 0.0) {
        return 1.0;
    }

    vec3 lightNDC = lightClip.xyz / lightClip.w;

    // GLM_FORCE_DEPTH_ZERO_TO_ONE (Starter.hpp) means lightNDC.z is already
    // Vulkan's 0..1 depth range, same as what's stored in the shadow map; only
    // XY need remapping from NDC's -1..1 to a texture's 0..1.
    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;

    // Outside the map (the sun's fixed ortho box doesn't reach here): nothing
    // to compare against. Missing this check would sample garbage at the
    // map's clamped edge instead.
    if(shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
       shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
       lightNDC.z < 0.0 || lightNDC.z > 1.0) {
        return 1.0;
    }

    float closestDepth = sampleShadowMap2D(idx, shadowUV);
    return (lightNDC.z - bias > closestDepth) ? 0.0 : 1.0;
}

// The torch path: one samplerCube lookup by direction, compared against the
// LINEAR distance to the light (see ShadowCube.frag's header for why the
// cube stores distance rather than projective depth). Unlike the old
// two-map dance this always has an answer -- a cube map covers every
// direction by construction -- so there is no "covered" out-parameter and no
// fallback to a second slot.
//
// A plain (dist - bias > closestDist) ? 0.0 : 1.0 comparison is a binary
// lit/unlit test, which reads as a razor-sharp edge on a wall. Multi-sample
// PCF (jittering the lookup direction and averaging several taps) was tried
// here and reverted: with only a few taps -- more wasn't affordable, since
// sampleShadowCube() below is a linear branch chain picking one of
// NUM_SHADOW_CUBES samplerCube bindings rather than a plain texture() call,
// so every tap repeats that whole chain -- the samples land far enough apart
// to show up as separate overlapping blobs instead of blending into one soft
// edge.
//
// This does the softening with the SAME single sample instead: rather than a
// hard step, it ramps from fully lit down to fully shadowed across a small
// band of world-space distance, starting exactly where the hard test's
// threshold used to sit.
//
// occluderGap is how much closer the stored occluder is than this fragment:
// ~0 (or negative, floating-point noise aside) when the map's closest hit
// IS this fragment's own surface -- i.e. nothing occludes it -- and growing
// positive as a real occluder sits further in front of it. Ramping the
// *lit* factor down starting at occluderGap == bias (equivalent to the old
// `dist - bias > closestDist` threshold) rather than at occluderGap == 0
// matters: with the old (closestDist - (dist - bias)) formulation, an
// unoccluded surface -- occluderGap == 0 -- landed only `bias` units into a
// softEdge-wide ramp rather than solidly on its plateau, so ordinary lit
// walls sat partway up the ramp and any texel-to-texel precision noise
// could tip the result either way -- which is exactly the "suddenly not
// lit" pop reported. This version has a flat lit==1.0 plateau for every
// occluderGap <= bias, matching the hard test's own unoccluded case.
float shadowFromCube(int idx, vec3 pos, vec3 lightPos, float bias, float softEdge) {
    vec3 toFrag = pos - lightPos;
    float dist = length(toFrag);
    float closestDist = sampleShadowCube(idx, toFrag);

    float occluderGap = dist - closestDist;
    return 1.0 - clamp((occluderGap - bias) / softEdge, 0.0, 1.0);
}

// 1.0: fully lit. 0.0: this light's shadow map says something else is closer
// to the light than `pos` is, i.e. `pos` is in shadow. shadowIndex < 0 skips
// the lookup and lights unconditionally, which is why a light without a slot
// leaks through every wall it reaches.
//
// NdotL is only used to pick the 2D-map bias, see below.
float shadowFactor(int shadowIndex, int type, vec3 pos, vec3 lightPos, float NdotL) {
    // Shadows off (cheat menu): light everything as if no map existed. Reads
    // gubo.debugFlags directly rather than through debugOn(), which is
    // declared further down the file.
    if(shadowIndex < 0 || (gubo.debugFlags & LIGHT_DEBUG_NO_SHADOWS) != 0) {
        return 1.0;
    }

    if(type == LIGHT_POINT) {
        // Slope-scaled, same reason the 2D path below does it: a flat bias
        // was tuned for a surface facing roughly toward the light, but at a
        // grazing angle -- exactly what you get brushing past a curved
        // pillar at close range, held torch just centimetres from its
        // surface -- one shadow-map texel covers a much larger stretch of
        // that surface's true depth, so the stored "closest occluder"
        // distance can be meaningfully off from any single fragment's real
        // distance even with nothing actually occluding it. A flat 0.03 was
        // nowhere near enough slack for that case: it read as a large,
        // unstable black patch swimming across the pillar as the player
        // walked past, not a fine-grained self-shadow flicker. Both the
        // bias and the softening band widen together at grazing incidence,
        // since the same growing depth-quantization error is what both are
        // there to absorb.
        const float CUBE_BIAS_MIN = 0.03;   // head-on
        const float CUBE_BIAS_MAX = 0.35;   // edge-on
        const float CUBE_SOFT_MIN = 0.15;
        const float CUBE_SOFT_MAX = 0.6;
        float grazing = clamp(1.0 - NdotL, 0.0, 1.0);
        float bias = mix(CUBE_BIAS_MIN, CUBE_BIAS_MAX, grazing);
        float softEdge = mix(CUBE_SOFT_MIN, CUBE_SOFT_MAX, grazing);
        return shadowFromCube(shadowIndex, pos, lightPos, bias, softEdge);
    }

    // The bias is per PROJECTION KIND, because one number cannot serve both.
    // These are offsets in the map's 0..1 depth, and how many centimetres that
    // buys depends entirely on how the projection distributes depth:
    //
    //   the sun's orthographic box spreads 1..200 linearly, so a fixed 0.0015
    //   is a fixed ~30cm everywhere. Left exactly as it was, since it works.
    //
    //   a hypothetical shadow-casting spot would crowd most of its range into
    //   the first metre the way the torches' old perspective maps did, hence
    //   the slope-scaled pair kept below for that branch.
    float bias;
    if(type == LIGHT_DIRECT) {
        bias = 0.0015;
    } else {
        const float BIAS_MIN = 0.0004;   // head-on
        const float BIAS_MAX = 0.0030;   // edge-on
        bias = mix(BIAS_MIN, BIAS_MAX, clamp(1.0 - NdotL, 0.0, 1.0));
    }

    return shadowFromMap2D(shadowIndex, pos, bias);
}

// Whether one of the debug views from LightConstants.glsl is on. All of them
// are off in a normal frame, so this is a uniform branch: every pixel of every
// draw takes the same side of it, which is the cheap kind on a GPU.
bool debugOn(int flag) {
    return (gubo.debugFlags & flag) != 0;
}

// LIGHT_DEBUG_HEATMAP's color ramp: black (no light) through blue, green,
// yellow, to red (overbright). A log compression goes first because the
// input is unclamped HDR radiance -- torches sit well above 1.0 close up --
// so a plain linear ramp would just read as solid red across most of a lit
// room instead of showing the falloff the cheat exists to visualize.
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

// Hemispheric ambient, E07 s.47-54. Indirect light, blended by which way the
// surface faces: aligned with ambientDir is all sky, opposite is all ground.
//
// interiorAmbient pins the blend at 0.5, the weight a vertical surface gets,
// instead of deriving it from the normal. Indoors the two ends of the model
// don't exist -- a dungeon ceiling has no sky above it and no courtyard below
// -- and taking the ground end literally made it collect ambientLower alone:
// 1.80x darker than the walls it meets, and brown where they are cool. Only
// the models that ask for it in materials.json; outdoors the real blend is
// what puts the sky on the tower tops.
vec3 hemisphericAmbient(vec3 N, vec3 mD) {
    float w = (dot(N, gubo.ambientDir) + 1.0) / 2.0;   // dot is -1..1, w is 0..1
    if(ubo.interiorAmbient == 1) w = 0.5;
    return mix(gubo.ambientLower, gubo.ambientUpper, w) * mD;
}

// How much of this fragment's light is indirect, 0..1. Per-model if the
// material set one, the scene's default otherwise.
//
// This is E17's gubo.ambientLight (LambertBlinnTexture.frag), the maze lab's
// one knob for an enclosed space, made per-model so a dungeon corridor and an
// open courtyard can hold different values in the same frame. The maze runs at
// 0.05; lights.json documents what the two ends of this scene use and why.
float ambientShare() {
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

    // max(dist, 0.0001) alone only guards the divide -- g/dist still grows
    // essentially unbounded as dist shrinks, so radiance stays fairly flat
    // over most of a room then rockets upward in the last stretch right
    // next to the source, which the tonemap then crushes to white almost
    // immediately after. Continuous on paper, but it reads as "barely
    // brightening, then suddenly maxed out" over the final approach to any
    // surface right in front of the light -- which is what a wall directly
    // ahead of the held torch shows as you walk up to it.
    //
    // NEAR_RADIUS softens that: it's a smooth floor on how close `dist` can
    // effectively get (sqrt(dist^2 + r^2) never drops below r), roughly the
    // torch flame's own physical size, so the curve flattens out near the
    // light instead of diverging. Spreads the same total brightness change
    // over more distance instead of dumping most of it into the last few
    // centimetres.
    const float NEAR_RADIUS = 0.4;
    float distSoft = sqrt(dist * dist + NEAR_RADIUS * NEAR_RADIUS);
    vec3 radiance = lt.color * pow(lt.g / distSoft, lt.beta);

    if(lt.type == LIGHT_SPOT) {
        // lt.dir is where the lamp POINTS, so a lamp aimed down is [0,-1,0].
        // The slides write this against lx, hence the negation here.
        float cosAngle = dot(-lightDirection(lt, pos), lt.dir);
        radiance *= clamp((cosAngle - lt.cosOut) / (lt.cosIn - lt.cosOut), 0.0, 1.0);
    }

    return radiance;
}

// D: fraction of microfacets oriented along h. GGX, E06 s.45.
float distributionGGX(vec3 N, vec3 h, float roughness) {
    float a2 = roughness * roughness;
    float NdotH = clamp(dot(N, h), 0.0, 1.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// G: microfacets shadowing each other. Parameter-free form, E06 s.47.
// Without it rough surfaces blow out at grazing angles.
float geometricTerm(vec3 N, vec3 h, vec3 L, vec3 V) {
    float NdotH = clamp(dot(N, h), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float VdotH = max(dot(V, h), 0.0001);

    return min(1.0, min(2.0 * NdotH * NdotV / VdotH,
                        2.0 * NdotH * NdotL / VdotH));
}

// F: fraction reflected instead of transmitted, F0 head-on rising to 1 at the
// horizon. Schlick, E06 s.46. The 5 is his fit, not a derivation.
float fresnelSchlick(vec3 V, vec3 h, float F0) {
    float VdotH = clamp(dot(V, h), 0.0, 1.0);
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// Cook-Torrance, E06 s.38-39:
//   fr = clamp(N.L) * (k * mD + (1-k) * mS * D*F*G / (4 * clamp(N.L) * clamp(N.V)))
// Diffuse and specular are interpolated by k, not added: adding both at full
// strength would return more light than came in. The 4*(N.L)*(N.V) is the
// microfacet-area to surface-area normalization, not a fudge factor.
vec3 BRDF(vec3 N, vec3 L, vec3 V, vec3 mD, vec3 mS, float roughness, float F0, float k) {
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    vec3 h = normalize(L + V);

    float D = distributionGGX(N, h, roughness);
    float G = geometricTerm(N, h, L, V);
    float F = fresnelSchlick(V, h, F0);

    // Denominator hits 0 at the silhouette. The NdotL below zeroes the result
    // there anyway; the guard just avoids an inf, since inf * 0 is a NaN.
    vec3 specular = mS * (D * F * G) / max(4.0 * NdotL * NdotV, 0.0001);

    // Lambert diffuse, as E06 s.38 prescribes for this model.
    // Clamped NdotL, so a face turned away contributes 0 rather than a negative
    // amount that would eat into what another light put there.
    return NdotL * (k * mD + (1.0 - k) * specular);
}

// The tone map (L09 s.45) used to be applied at the end of this shader. It has
// MOVED to Composite.frag, the last pass of the HDR chain, and the reason is
// worth stating: tone mapping squashes everything into [0,1], and a bloom pass
// works by finding the pixels that came out ABOVE 1. Compressing the range here
// would throw away the only thing the bright pass is looking for, and the flame
// would end up with no glow around it at all.
//
// So this shader now writes raw, un-clamped radiance into a floating-point
// attachment, and the range compression happens once, at the very end, after
// the bloom has been extracted from it. The Tone Mapping cheat still works; it
// just takes effect one pass later (gubo.debugFlags is forwarded to the
// composite's own uniform block by updateUniformBuffer()).

void main() {
    // Interpolation shortens the normal wherever the corner normals diverge.
    vec3 N = normalize(fragNorm);

    // The MGCG models average their vertex normals across hard edges, so a flat
    // face comes out with a gradient across it instead of one constant value
    // (E06 s.3-16: a hard-edged solid needs its vertices duplicated per face,
    // and these are not). For those models the face's own normal is derived
    // here instead: the derivatives of the world position across the triangle
    // are two vectors lying in its plane, so their cross product is exactly
    // perpendicular to it.
    //
    // The sign of that cross product depends on winding, so it is oriented
    // against the vertex normal, which is unreliable in magnitude but perfectly
    // good at saying which side is out.
    if(ubo.flatNormals == 1) {
        vec3 faceN = normalize(cross(dFdx(fragPos), dFdy(fragPos)));
        N = dot(faceN, N) < 0.0 ? -faceN : faceN;
    }

    // No sRGB conversion here: the texture's image view is VK_FORMAT_R8G8B8A8_SRGB
    // (Starter.hpp's default), so the sampler already returns linear values.
    vec3 mD = texture(albedoMap, fragUV).rgb;

    // Debug view: the shading normal, remapped from [-1,1] to [0,1], so +X is
    // red, +Y green, +Z blue. Placed after the flatNormals block above so what
    // it shows is the normal the lighting actually used, faceted faces
    // included, which is the point of looking at it. Not a color in any real
    // sense, so the sRGB encode the swapchain applies to it is meaningless
    // here; the directions are still perfectly readable.
    if(debugOn(LIGHT_DEBUG_NORMALS)) {
        outColor = vec4(N * 0.5 + 0.5, 1.0);
        return;
    }

    // Debug view: the texture alone, no lighting and no ambient. Tells a black
    // pixel that no light reached apart from a black pixel in the texture.
    if(debugOn(LIGHT_DEBUG_UNLIT)) {
        outColor = vec4(mD, 1.0);
        return;
    }

    vec3 V = normalize(gubo.eyePos - fragPos);

    // Debug view: incoming light intensity, ignoring the surface's own
    // albedo. Forcing mD/mS/k the same way LIGHT_DEBUG_NO_SPECULAR does
    // reuses the normal Lo loop below unchanged; only what happens to the
    // result (the ramp instead of a straight write) differs, further down.
    bool heatmap = debugOn(LIGHT_DEBUG_HEATMAP);
    if(heatmap) {
        mD = vec3(1.0);
    }

    // k is the diffuse share, so forcing it to 1 leaves the specular term
    // multiplied by 0: the highlights go, everything else stays exactly as it
    // was. Done here rather than inside BRDF so that function keeps taking all
    // its inputs as arguments.
    float k = (debugOn(LIGHT_DEBUG_NO_SPECULAR) || heatmap) ? 1.0 : ubo.k;

    // Rendering equation: sum over the sources of radiance times BRDF, each
    // term zeroed by shadowFactor() wherever that one light doesn't reach
    // this point. Ambient below is untouched by it on purpose: shadow mapping
    // only ever blocks a light's DIRECT contribution, never the indirect
    // bounce hemisphericAmbient() stands in for -- otherwise a shadow would
    // read as a hole into pure black instead of the dim, indirectly-lit area
    // a real one is.
    // Below this, a light's radiance at this fragment is darker than the
    // final image can show even before the BRDF and shadow map get
    // involved -- well under 1 LSB of an 8-bit display once exposure and
    // the sRGB curve in Composite.frag are through with it. Point/spot
    // lights are only CPU-culled by distance to the CAMERA (see
    // TORCH_LIGHT_CULL_DIST in main.cpp), not to the fragment being shaded,
    // so a torch that passed that cull can still be near-zero at a
    // fragment on the far side of a large room; this catches that case per
    // pixel instead.
    const float LIGHT_ATTEN_EPS = 1e-3;

    vec3 Lo = vec3(0.0);
    for(int i = 0; i < gubo.lightCount; i++) {
        vec3 radiance = lightRadiance(gubo.lights[i], fragPos);

        // Cheap reject before the expensive part: BRDF's handful of pow()s
        // and, more importantly, shadowFactor()'s dependent shadow-map
        // texture fetch. A direct light's radiance is a constant scene
        // color, never near-zero while it's enabled, so this never fires
        // for the sun -- only point/spot lights actually decay with
        // distance.
        if(max(radiance.r, max(radiance.g, radiance.b)) < LIGHT_ATTEN_EPS) {
            continue;
        }

        vec3 L = lightDirection(gubo.lights[i], fragPos);
        // Same clamped dot the BRDF uses, computed once here because
        // shadowFactor scales its depth bias by it too.
        float NdotL = clamp(dot(N, L), 0.0, 1.0);
        Lo += radiance
            * BRDF(N, L, V, mD, ubo.mS, ubo.roughness, ubo.F0, k)
            * shadowFactor(gubo.lights[i].shadowIndex, gubo.lights[i].type, fragPos, gubo.lights[i].pos, NdotL);
    }

    // E17's blend (LambertBlinnTexture.frag:51-52), not a sum: ambient is a
    // SHARE of the light at this fragment, and the direct term gives up
    // exactly what ambient takes. Summing the two, as this did before, made
    // hemisphericAmbient() a brightness floor under every pixel in the scene
    // -- it has no visibility term, so a sealed room collected the same
    // indirect light as the open courtyard, and no ceiling could stop it. The
    // blend can't do that: at w the ambient never contributes more than w of
    // the frame, and the total can never exceed what the direct lights alone
    // would have given.
    //
    // Note this is still not occlusion. It is a per-model authored guess at
    // how enclosed a surface is, which is what E17 does and what the assets
    // allow -- the MGCG pack ships albedo only, so the AO map E14/E15 sample
    // (aoMap, [TODO 5b]) has nothing to read. Baking one is the honest fix.
    float aw = ambientShare();
    vec3 color = Lo * (1.0 - aw) + hemisphericAmbient(N, mD) * aw;

    if(heatmap) {
        float intensity = dot(color, vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(heatmapRamp(intensity), 1.0);
        return;
    }

    // Focus glow: a whimsical gold "magic field" aura on the silhouette of
    // whichever single door/pickup instance the player's crosshair is
    // currently aimed at (ubo.glow, set per-instance in main.cpp's
    // updateUniformBuffer loop -- see gazedInstance), chosen to stay clear
    // of the ghost attack mode's own color cue. Purely a per-fragment color
    // addition on this instance's own surface -- it doesn't cast any light
    // onto anything else nearby (an earlier version injected a point light
    // for that; removed, so the effect stays exactly on the model).
    //
    // A Fresnel/rim term (grazing angles between the surface normal and the
    // view direction) concentrates this at the model's silhouette rather
    // than washing the whole surface, like a field clinging to its edges.
    // `flow` rides a sine wave over world position instead of just fragment
    // time, so as the term sweeps 0..1 it reads as travelling around the
    // object's surface rather than the whole thing pulsing in place
    // together.
    if(ubo.glow > 0.0) {
        const vec3 GLOW_COLOR = vec3(1.0, 0.78, 0.25);
        float ndotv = clamp(dot(N, V), 0.0, 1.0);
        float edgeTerm = 1.0 - ndotv;

        // A bit wider than the original pow(edgeTerm, 3.0): on its own this
        // still wasn't enough contrast on small, shiny props like the key,
        // whose own specular highlights compete with a thin gold rim for
        // attention -- see the dark outline below for the rest of the fix.
        float rim = pow(edgeTerm, 2.2);

        // A thin, near-black separator right at the true geometric
        // silhouette -- a much sharper power than the gold rim, so it only
        // shows in the last couple of degrees -- darkening the surface
        // there BEFORE the gold is added. A reflective surface like the
        // key's can otherwise bounce a bright highlight straight through
        // right where the gold band sits, washing the two together into one
        // gold-on-gold blur; giving the gold something duller immediately
        // underneath it at the very edge is what actually separates it from
        // the model, the way an outline separates a sticker from its
        // background.
        float edgeOutline = pow(edgeTerm, 12.0);
        color = mix(color, color * 0.15, edgeOutline * ubo.glow);

        float flow = 0.5 + 0.5 * sin(dot(fragPos, vec3(1.3, 0.9, 1.1)) * 2.2 - ubo.time * 2.0);
        vec3 glowRaw = GLOW_COLOR * ubo.glow * rim * flow * 0.9;

        // Composite.frag's tone map (toneMap() there) divides everything at
        // a pixel by that SAME pixel's own total luminance, so a plain
        // additive glow gets proportionally crushed wherever the surface
        // underneath is already bright, and shows almost undimmed wherever
        // it's dark -- exactly backwards from "always visible no matter
        // what". Scaling the addition by (1 + this fragment's own
        // pre-glow luminance) cancels that division back out to first
        // order (the algebra: output = (c + k*(Y+1)) / (Y + k*(Y+1) + 1)
        // -> k/(1+k) as Y grows, a constant, instead of shrinking towards
        // 0), so the glow's apparent brightness ends up roughly the same
        // whether the object sits in full torchlight or pitch dark.
        // Unclamped HDR like the rest of `color`, so it still blooms
        // through the same post chain the torch flames ride.
        float baseLuminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
        color += glowRaw * (1.0 + baseLuminance);
    }

    // Written linear and unclamped into an R16G16B16A16_SFLOAT attachment, so
    // a surface that receives more than a unit of light keeps saying so rather
    // than being cut off at white. Composite.frag tone maps it down at the end
    // of the chain; the bloom passes in between are what read the excess.
    outColor = vec4(color, 1.0);
}
