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
    // 1: shade this model as a METAL. Two things follow from it, both in
    // main(): the diffuse term goes away entirely (k is forced to 0, a metal
    // has no subsurface scattering to produce one), and the indirect term
    // becomes metalAmbient() -- a reflection of the room -- instead of the
    // hemisphere times the albedo. See Material::metallic in SceneMaterials.hpp.
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
    vec3 ambientUpper;   // indirect light from the sky
    vec3 ambientLower;   // indirect light bounced off the ground
    vec3 ambientDir;     // axis the two blend along, i.e. world up
    int debugFlags;      // LIGHT_DEBUG_* bits, set by the cheat menu
    float time;          // seconds since startup, unused here (see Flame.vert)
    // The scene's default share of ambient, 0..1. See ambientShare() below.
    // Rides in the padding before lights[], like debugFlags and time.
    float ambientWeight;
    // Share of a point/spot light's radiance that comes back as INDIRECT
    // light. See AmbientLight::bounce in SceneLights.hpp and pointBounce()
    // below. Rides in the same padding.
    float ambientBounce;
    // Distance fog density, set from GEOM_CULL_CONE_DIST in
    // updateUniformBuffer() -- see its comment there. Used at the very end
    // of main() below, in the same padding as everything above it.
    float fogDensity;
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

// Same idea for the cube maps, sampled by direction rather than by UV, with
// one difference that matters: this returns FOUR taps, not one.
//
// The PCF this feeds was tried once before as four separate one-tap calls and
// reverted as unaffordable, correctly -- the chain below is a linear walk of
// up to 32 comparisons to pick a binding, so calling it four times walks it
// four times, and it is the walk, not the fetch, that costs. Taking all four
// taps INSIDE the branch that already resolved pays for the walk once and
// adds three texture reads to it, on texels adjacent to the first, which is
// the cheapest thing a sampler can be asked to do.
//
// NUM_SHADOW_CUBES is 32 (LightConstants.glsl: 31 dynamically-assigned slots
// shared by every wall/dl torch and candle, plus the held torch's own fixed
// last one); a case has to be added or removed here by hand if that changes.
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
// LINEAR distance to the light (see ShadowCube.frag's header for why the
// cube stores distance rather than projective depth). Unlike the old
// two-map dance this always has an answer -- a cube map covers every
// direction by construction -- so there is no "covered" out-parameter and no
// fallback to a second slot.
//
// A plain (dist - bias > closestDist) ? 0.0 : 1.0 comparison is a binary
// lit/unlit test, which reads as a razor-sharp edge on a wall. The softening
// is four-tap PCF: sampleShadowCube4() reads the map at four directions
// around the lookup and this averages the four verdicts.
//
// Two earlier attempts at that softness are worth keeping straight, because
// the shape of this one is a reaction to both.
//
// PCF was tried first as four separate one-tap calls and reverted as
// unaffordable -- correctly, for the reason sampleShadowCube4()'s header
// gives, and the fix was to restructure the fetch rather than to give up on
// the technique. It was also reverted as ugly, the taps reading as separate
// overlapping blobs; that was the kernel, jittered far enough apart to alias
// with four samples. The kernel here is a few texels wide and rotated.
//
// What replaced it was softening from the SAME single sample: ramp the lit
// factor down across a band of occluderGap instead of stepping it. That is
// the version this one replaces, and its failure is worth stating because it
// is not obvious. occluderGap measures how far INTO a shadow a fragment is,
// not how far the occluder is from it, and the two come apart badly wherever
// a shape overhangs its own base. Around the foot of a barrel -- widest at
// its waist, so the floor by its base sits under the bulge -- the occluder
// stays barely in front of the floor for some way out, the gap crawls, and a
// band 0.03 wide in gap spread over roughly 0.1 of floor. The band could only
// ever spill to the LIT side of the silhouette, so what it produced there was
// a bright strip between the barrel and its own shadow, and every value that
// made the strip acceptable made the shadow edge hard. A kernel has no such
// bind: its taps straddle the silhouette, so the transition is centred on the
// true edge and widening it costs no contact.
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
//
// The bias is computed HERE rather than handed in, because the only honest
// way to size it needs `dist`, which only this function has. It used to be a
// pair of world-space constants interpolated by grazing angle (0.03 head-on to
// 0.35 edge-on, with a matching 0.15..0.6 ramp), and that number was far too
// large for what a bias is actually for: a door panel sits ~0.6 units in front
// of the chains bolted to it, so a torch on the far side of a CLOSED door
// still lit them through it wherever the surface faced the light obliquely --
// which on a round chain link is most of what you see, and on a metal (no
// diffuse term, all specular) it reads as the torch's colour smeared over the
// links. The oversized bias was itself compensating for the bilinear cube
// sampler, see createCubeShadowMaps(); with NEAREST filtering the bias only
// has to cover the error that is genuinely there:
//
//   one texel of a cube face covers 2*dist/SHADOW_CUBE_RES of world space at
//   distance dist (a face spans 90 degrees, so its width at dist is 2*dist),
//
//   and across that texel the recorded surface's own distance varies by that
//   width times the slope of the surface as seen from the light, i.e. tan of
//   the incidence angle -- which is what makes a grazing surface need more
//   slack than a head-on one, the effect the old constants were reaching for
//   but expressed in the units it actually happens in.
//
// That error, though, must NOT be paid for out of the depth bias, which is
// the mistake the first attempt at this repeated in smaller units. Whatever
// the bias is, it is a distance a real occluder is allowed to sit in front of
// a surface without stopping the light -- so the moment it approaches the
// ~0.65 units between a chain link and the far face of the door leaf it hangs
// on, the door stops being a door. It grows with distance (the texel does),
// the gap doesn't, so no cap on it is both large enough to cover a far wall
// and small enough to respect a near door: at the chains' 5 units the first
// version held, at the blue and purple torches' 10 and 12 it hit its cap and
// let them through, which is exactly the two colours that survived.
//
// The version after that spent it on a NORMAL OFFSET instead -- move the
// lookup along the surface's own normal, off the surface and towards the
// light, so it lands in a texel whose recorded distance belongs to this
// surface rather than to the stretch of it half a texel away -- sized
// texelWorld*(1 + 2*tan) against a flat 0.12 cap. That does not leak through
// the occluder, but it buys the offset's own failure mode instead:
// PETER-PANNING. tan diverges at grazing incidence, so a floor lit obliquely
// by a wall torch always paid the cap, ~12 texels at 5 units where a normal
// offset wants one or two; lifting the sample point 0.12 off the floor shifts
// the shadow's edge sideways by 0.12/tan(the torch's elevation over that
// floor), and a barrel's shadow detached from its base by a third of a unit.
//
// BOTH of those are the same misattribution, and the fix is neither: the
// across-texel error belongs to the surface being SAMPLED, so it is charged
// there, baked into the stored distance by ShadowCube.frag using that
// surface's own tilt. See its header. What is left here is a token constant
// for floating-point noise -- the cube sampler is NEAREST, so there is no
// filtering error on top -- and one texel of normal offset, sin-scaled so it
// stays inside that budget at every angle, against the cube's own
// quantisation of the receiver's position. At 5 units from a torch the two
// together come to under two centimetres of world space, which is where the
// lit strip between a barrel and its shadow went.
// LIGHT_DEBUG_SHADOW_GAP's working state, written by shadowFromCube() and
// read once at the end of main(). Globals rather than out-parameters because
// the call sits inside the light loop's expression and there is exactly one
// invocation's worth of them.
//
// A shadow belongs to a specific light, so this reports ONE of them: the one
// whose radiance dominates this fragment, picked in the light loop below.
//
// Two earlier choices were both wrong in the same way -- they picked the
// light by something other than which one is actually lighting the fragment.
// Keeping the smallest gap, i.e. the light most willing to call it lit,
// paints the frame green: with several torches burning almost every fragment
// has at least one flame with a clear line to it. Pinning it to the held
// torch instead answers honestly but about the wrong light -- a floor
// shadowed by a WALL torch is genuinely unoccluded as far as the torch in
// your hand is concerned, so that reads green too, and says nothing about
// the shadow being looked at.
//
// dbgLast* is scratch: shadowFromCube() fills it for whichever light it was
// just called for, and the loop promotes it to dbg* if that light is the
// brightest seen so far.
bool  dbgLastValid = false;
float dbgLastGap = 0.0;
float dbgLastBias = 0.0;
float dbgLastSoft = 0.0;

bool  dbgCube = false;
float dbgGap = 0.0;
float dbgBias = 0.0;
float dbgSoft = 0.0;
float dbgBestLum = -1.0;

float shadowFromCube(int idx, vec3 pos, vec3 N, vec3 lightPos, float NdotL) {
    // Depth slack: floating-point noise only. There is no acne left for it to
    // cover -- the capture culls front faces, so this surface is not in the
    // map to be compared against itself (see PShadowCube.setCullMode() in
    // main.cpp). What remains is the difference between a distance computed
    // here and the same distance computed in ShadowCube.frag from an
    // interpolated position, which is a few ULPs, not millimetres.
    const float CUBE_BIAS_MIN = 0.0015;
    const float CUBE_BIAS_MAX = 0.004;
    const float CUBE_SOFT_MIN = 0.002;
    // No normal offset. It existed to land the lookup in a texel whose record
    // belongs to this surface rather than to the stretch of it half a texel
    // away -- a concern that only arises when the surface is IN the map, which
    // it no longer is. It cost a lateral shift of the shadow edge for that,
    // which is the artifact this whole path was chasing.
    const float NORMAL_OFFSET_TEXELS = 0.0;
    const float NORMAL_OFFSET_MAX = 0.0;
    // Radius of the PCF kernel, in texels of the cube face. This is the only
    // thing that sets how wide a shadow's edge reads now, and it is the one
    // place where widening it does NOT eat the contact: the taps sit around
    // the lookup direction, so the transition straddles the true silhouette
    // instead of spilling to the lit side of it.
    const float PCF_KERNEL_TEXELS = 3.0;

    float rawDist = length(pos - lightPos);
    float texelWorld = 2.0 * rawDist / float(SHADOW_CUBE_RES);

    // sin of the incidence angle. cosI is still floored -- a fragment that
    // close to edge-on receives almost nothing from this light anyway (the
    // BRDF's own NdotL factor), so there is nothing to protect there, and the
    // floor keeps sinI from reaching 1 and spending the full budget on a
    // surface that cannot show the acne it would be paying for.
    float cosI = max(NdotL, 0.15);
    float sinI = sqrt(1.0 - cosI * cosI);

    float offset = min(texelWorld * NORMAL_OFFSET_TEXELS * sinI, NORMAL_OFFSET_MAX);
    vec3 samplePos = pos + N * offset;

    vec3 toFrag = samplePos - lightPos;
    float dist = length(toFrag);

    float bias = clamp(0.5 * texelWorld, CUBE_BIAS_MIN, CUBE_BIAS_MAX);
    // Per-tap band, not the shadow's edge softness -- that is the kernel's job
    // now. This one only keeps a single tap from being a hard step, so the
    // four of them average into something continuous instead of five levels.
    float softEdge = CUBE_SOFT_MIN;

    // The four tap directions: the lookup direction pushed sideways in the
    // plane perpendicular to it. Offsetting by `r` world units at right angles
    // to a direction of length dist turns it by r/dist, and one texel subtends
    // texelWorld/dist, so an offset measured in texelWorld is an offset
    // measured in texels of the face being read -- the same currency the bias
    // is in, and independent of how far the light is.
    vec3 axis = abs(toFrag.y) < 0.99 * dist ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(toFrag, axis));
    vec3 B = normalize(cross(toFrag, T));
    float r = PCF_KERNEL_TEXELS * texelWorld;

    // Rotated grid rather than a 2x2 box: four points on a square grid share
    // two x and two y coordinates, so they straddle a straight silhouette in
    // only three distinct ways and the edge steps in thirds. Rotated, all four
    // cross it at different offsets.
    vec3 d0 = toFrag + r * ( 0.33 * T + 1.00 * B);
    vec3 d1 = toFrag + r * ( 1.00 * T - 0.33 * B);
    vec3 d2 = toFrag + r * (-0.33 * T - 1.00 * B);
    vec3 d3 = toFrag + r * (-1.00 * T + 0.33 * B);

    // One `dist` for all four. Each tap's own direction is longer than toFrag
    // by r^2/(2*dist) -- at three texels that is under a micrometre of world
    // space, far below the bias, and using it would only mean four different
    // thresholds for what is meant to be one test sampled four times.
    vec4 gaps = vec4(dist) - sampleShadowCube4(idx, d0, d1, d2, d3);
    vec4 lit = vec4(1.0) - clamp((gaps - vec4(bias)) / softEdge, 0.0, 1.0);

    // Scratch for the debug view, for whichever light this call was for; the
    // light loop decides which one survives. The tap reported is the one most
    // willing to call this lit, which is what the whole answer would have
    // been before the kernel existed.
    dbgLastValid = true;
    dbgLastGap = min(min(gaps.x, gaps.y), min(gaps.z, gaps.w));
    dbgLastBias = bias;
    dbgLastSoft = softEdge;

    return dot(lit, vec4(0.25));
}

// 1.0: fully lit. 0.0: this light's shadow map says something else is closer
// to the light than `pos` is, i.e. `pos` is in shadow. shadowIndex < 0 skips
// the lookup and lights unconditionally, which is why a light without a slot
// leaks through every wall it reaches. Only LIGHT_POINT ever has a real
// shadowIndex (see LightData::shadowIndex in SceneLights.hpp) -- a direct or
// spot light always lights unconditionally through this same early-out.
//
// NdotL scales the cube path's normal offset, which is why N has to come
// along too.
float shadowFactor(int shadowIndex, int type, vec3 pos, vec3 N, vec3 lightPos, float NdotL) {
    // Shadows off (cheat menu): light everything as if no map existed. Reads
    // gubo.debugFlags directly rather than through debugOn(), which is
    // declared further down the file.
    if(shadowIndex < 0 || (gubo.debugFlags & LIGHT_DEBUG_NO_SHADOWS) != 0) {
        return 1.0;
    }

    // type is always LIGHT_POINT here: a direct or spot light never gets a
    // shadowIndex >= 0 (SceneLights::init), so the check above already
    // caught it. Bias, softening band and normal offset all live inside
    // shadowFromCube() now: they are derived from the distance to the
    // light, which is the one thing this function doesn't have and that
    // one computes anyway. N and NdotL are what scale them, see there.
    return shadowFromCube(shadowIndex, pos, N, lightPos, NdotL);
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
//
// Split in two: hemisphereColor() is the incoming indirect light along a
// direction, with no surface in it at all, because the metals below need it
// sampled along their REFLECTED direction rather than along the normal.
// hemisphericAmbient() is that light landing on a diffuse surface, which is
// what every dielectric in the scene wants and what this function used to be.
vec3 hemisphereColor(vec3 dir) {
    float w = (dot(dir, gubo.ambientDir) + 1.0) / 2.0;   // dot is -1..1, w is 0..1
    if(ubo.interiorAmbient == 1) w = 0.5;
    return mix(gubo.ambientLower, gubo.ambientUpper, w);
}

vec3 hemisphericAmbient(vec3 N, vec3 mD) {
    return hemisphereColor(N) * mD;
}

// The indirect term for a METAL, standing in for hemisphericAmbient() on the
// models that set ubo.metallic.
//
// What a lock and a chain actually look like is mostly not their own colour:
// a metal has no diffuse component to scatter light back with, so nearly
// everything the eye gets off one is a REFLECTION of what is around it. The
// diffuse ambient above cannot express that -- it hands the surface a colour
// picked by where the surface FACES, times an albedo, which is the one thing a
// metal does not do. Left on it, the chain came out very close to black
// wherever no torch reached it directly (its UVs sample the door texture's
// wrought-iron band, albedo ~0.03 linear, so mD kills the term outright) and
// the padlock came out as flat orange fill, i.e. painted plastic.
//
// The scene has no environment map, so the hemisphere IS the environment here:
// the same two colours lights.json authored, read along the mirror direction.
// That is the cheapest honest form of the split-sum ambient specular (E07's
// hemisphere in place of a prefiltered cube map); a real one needs a capture
// pass the project does not have.
vec3 metalAmbient(vec3 N, vec3 V, vec3 mS, float roughness, float F0) {
    vec3 R = reflect(-V, N);

    // A rough metal reflects a BLURRED room, not a sharp one, and with a
    // two-colour hemisphere and no mip chain there is nothing to blur. The
    // stand-in is to slide the sample direction from R (mirror) towards N
    // (what a fully diffuse surface would use) as roughness grows: the two
    // ends are exactly the two ends the real thing interpolates between.
    vec3 D = normalize(mix(R, N, roughness));

    // Schlick once more, but on N.V: there is no half vector here, since the
    // "light" is the whole hemisphere rather than one direction. The ceiling
    // is max(1 - roughness, F0) instead of the plain 1.0 the direct term uses
    // -- a rough metal does not turn into a perfect mirror at the horizon, and
    // letting it reach 1.0 puts a hard bright rim on exactly the pixels that
    // outline a tube, which is the artefact the chains were already fighting.
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float F = F0 + (max(1.0 - roughness, F0) - F0) * pow(1.0 - NdotV, 5.0);

    // mS, not mD: for a metal the specular colour IS the material's colour.
    return hemisphereColor(D) * mS * F;
}

// How much of this fragment's light is indirect, 0..1. Per-model if the
// material set one, the scene's default otherwise.
//
// This is E17's gubo.ambientLight (LambertBlinnTexture.frag), the maze lab's
// one knob for an enclosed space, made per-model so a dungeon corridor and an
// open courtyard can hold different values in the same frame. The maze runs at
// 0.05; lights.json documents what the two ends of this scene use and why.
//
// The cheat gate sits HERE, downstream of both places a share can be authored,
// rather than on gubo.ambientWeight alone: a model carrying its own override
// never reads the global, so zeroing the global left the floor (0.20 in
// materials.json) fully lit indirectly with the cheat off. See
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

// ---------------------------------------------------------------------------
// Procedural grime, for the interior metals (chains, padlock, key). Those
// three have been underground long enough to tarnish -- dust settled in the
// pits, a filmed-over patch here and there -- and none of it is in the flat
// albedo the MGCG pack ships. With no second UV set and no dirt map to
// sample, it is generated from world position: a few octaves of value noise
// read where the fragment actually sits in the room, so neighbouring chain
// links come out weathered differently instead of identically.
//
// main() uses it for two things: roughness UP (grime scatters what bare
// metal would throw back sharply) and the reflection tint DOWN (a filmed
// surface reflects less of the room). Albedo is left alone -- a metal's k is
// forced to 0, so there is no diffuse term for dirt to darken.
//
// Gated in main() to metal && interiorAmbient, so it lands on those three
// and not on the outdoor gate lanterns ("light"), which the weather rinses.
float grimeHash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

// Value noise: hash the eight corners of the cell p falls in, smoothstep the
// fractional position, trilinearly blend. Same idea as Flame.frag's noise2,
// one dimension up.
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

// 0 clean .. 1 filthy. The three objects are ~1-3 world units across, so
// 6 / 18 / 50 per unit place the coarse blotches at a few centimetres and
// the grain below that. smoothstep keeps clean metal genuinely clean and
// drives the dirty patches most of the way, rather than a flat grey veil
// over everything.
float grime(vec3 worldPos) {
    float g = grimeNoise(worldPos *  6.0) * 0.6
            + grimeNoise(worldPos * 18.0) * 0.3
            + grimeNoise(worldPos * 50.0) * 0.1;
    return smoothstep(0.35, 0.80, g);
}
// ---------------------------------------------------------------------------

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

    // Both debug views want the specular gone, and the heatmap additionally
    // wants nothing about the material in the result. Named once because the
    // metal path below has to stand down for both of them: a "no specular"
    // view of a surface whose whole response is specular has to show the
    // diffuse fallback, not the metal.
    bool specularOff = debugOn(LIGHT_DEBUG_NO_SPECULAR) || heatmap;
    bool metal = ubo.metallic == 1 && !specularOff;

    // k is the diffuse share, so forcing it to 1 leaves the specular term
    // multiplied by 0: the highlights go, everything else stays exactly as it
    // was. Done here rather than inside BRDF so that function keeps taking all
    // its inputs as arguments.
    //
    // A metal goes the other way, to 0: the diffuse lobe comes from light that
    // entered the surface and scattered back out, and in a conductor the free
    // electrons absorb that instead of re-emitting it. Forced here rather than
    // written as "k": 0.0 in materials.json so the flag carries the whole
    // definition of "this is a metal" in one place, and so no metal entry can
    // be left with a stray diffuse share by accident.
    float k = specularOff ? 1.0 : (metal ? 0.0 : ubo.k);

    // Grime, for the interior metals only (see grime() above). Everyone else
    // takes ubo.roughness / ubo.mS unchanged: g is 0, so both mixes are the
    // identity. Where it does apply, a dirty patch roughens the surface --
    // ubo.roughness*2 + 0.20, capped short of fully matte -- and mutes the
    // reflection tint towards 0.4 of itself.
    //
    // The wrought-iron chains wear it fully; the cast-brass padlock and key
    // get much less. No per-model flag for that -- it reads the specular
    // colour, which is already the tell: brass is warm (mS.b well under
    // mS.r), steel is all but neutral (ratio ~1).
    float warmth = ubo.mS.b / max(ubo.mS.r, 1e-4);          // ~0.46 brass, ~1.04 steel
    float brassness = 1.0 - smoothstep(0.6, 0.95, warmth);  // 1 brass, 0 steel
    // Two knobs, not one. grimeScale drops the overall bite; the pow() with
    // an exponent above 1 for brass crushes the mid-grey coverage so only the
    // few concentrated hotspots survive -- that is what breaks up the big
    // soft blob rather than just fading it.
    float grimeScale = mix(1.0, 0.30, brassness);
    float g = grime(fragPos);
    g = pow(g, mix(1.0, 2.5, brassness)) * grimeScale;
    g = (metal && ubo.interiorAmbient == 1) ? g : 0.0;
    float roughG = mix(ubo.roughness, min(ubo.roughness * 2.0 + 0.20, 0.95), g);
    vec3  mSG    = ubo.mS * mix(1.0, 0.40, g);

    // Rendering equation: sum over the sources of radiance times BRDF, each
    // term zeroed by shadowFactor() wherever that one light doesn't reach
    // this point. hemisphericAmbient() below is untouched by it on purpose:
    // shadow mapping only ever blocks a light's DIRECT contribution, never
    // the sky-and-ground indirect that term stands in for -- otherwise a
    // shadow would read as a hole into pure black instead of the dim,
    // indirectly-lit area a real one is. That exemption is the hemisphere's
    // alone. The per-light bounce accumulated in the loop below is shadowed
    // like everything else there, for the reason given at its own site.
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
    // The point/spot lights' INDIRECT contribution, accumulated alongside
    // their direct one. Spent below, out of the ambient share rather than
    // added to the frame -- see the blend.
    //
    // In this loop rather than a pointBounce() of its own purely so it can
    // reuse `radiance`, `L` and the light's visibility: those are the whole
    // cost of the term, and computing them twice would double the
    // length()/pow() work and a second shadow-map fetch in the hottest loop
    // in the shader to produce identical numbers. It still skips the BRDF,
    // which is the other expensive half and the one it genuinely does not
    // want -- bounced light arrives from most of the hemisphere, so it has no
    // lobe and no terminator.
    vec3 bounce = vec3(0.0);
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
        dbgLastValid = false;
        float vis = shadowFactor(gubo.lights[i].shadowIndex, gubo.lights[i].type, fragPos, N, gubo.lights[i].pos, NdotL);
        // See the dbg* globals: keep the cube-shadowed light that contributes
        // most radiance here, which is the one whose shadow this fragment is
        // in or out of in any way worth looking at.
        float dbgLum = dot(radiance, vec3(0.2126, 0.7152, 0.0722));
        if(dbgLastValid && dbgLum > dbgBestLum) {
            dbgBestLum = dbgLum;
            dbgCube = true;
            dbgGap = dbgLastGap;
            dbgBias = dbgLastBias;
            dbgSoft = dbgLastSoft;
        }
        Lo += radiance
            * BRDF(N, L, V, mD, mSG, roughG, ubo.F0, k)
            * vis;

        // Wrap-around diffuse, NOT the clamped cosine the BRDF just used:
        // (dot + 1) / 2 instead of max(dot, 0). Light that reaches a surface
        // after bouncing arrives from most of the hemisphere rather than from
        // the flame's own direction, so it has no terminator -- a face turned
        // away from a torch is dimmer than one facing it, not black. Exactly
        // the remap hemisphereColor() does on the ambient axis, applied to
        // the light's direction instead of world up.
        //
        // Direct lights are excluded, and that is deliberate rather than an
        // optimisation: the sun's own indirect contribution is what the
        // hemispheric term already IS (sky above, ground bounce below, E07
        // s.47-54), so feeding it in here would double it. This term exists
        // for the sources the hemisphere cannot represent -- the ones with a
        // position, that light one end of a corridor and not the other. It is
        // therefore entirely independent of whether the sun exists at all.
        //
        // `vis` applies here too, and originally it did not: bounced light was
        // held to be what FILLS a shadow rather than something a shadow can
        // block, which is true of the light itself and false of the number
        // used to stand in for it. Its magnitude is `radiance`, the direct,
        // unoccluded, inverse-square arrival from the flame -- so a shadow
        // cast by anything near a torch got filled most brightly at the end
        // nearest the torch, fading along its own length. That is the glow at
        // the start of a barrel's shadow, and no amount of shadow-map work
        // could reach it: the term was never consulting the shadow map.
        //
        // Shadowing it does not put the scene back to a black shadow, which
        // is what the original reasoning was protecting against.
        // hemisphericAmbient() below is still unshadowed and still the floor
        // under every fragment; what goes away is only the part that was
        // tracking distance to a flame the fragment cannot see.
        if(gubo.lights[i].type != LIGHT_DIRECT) {
            bounce += radiance * (dot(N, L) * 0.5 + 0.5) * vis;
        }
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
    //
    // The metals take the same share of the frame, spent on a reflection of
    // the room instead of on a diffuse bounce -- see metalAmbient(). Same
    // blend, same weight: what changes is only what the indirect light does
    // once it lands.
    // Indirect light now has two sources, and they answer different questions.
    // hemisphericAmbient() is "what arrives from the sky and from the ground",
    // which is the right model outdoors and vacuous in a sealed corridor. The
    // bounce is "what arrives from the torches after hitting a wall", which is
    // the only indirect light there actually is down there. Summed, because
    // they are genuinely two different sources -- but summed INSIDE the
    // ambient bucket, so the pair still cannot take more than `aw` of the
    // frame and the E17 blend's guarantee below is untouched.
    //
    // mD for the same reason hemisphericAmbient() applies it: this is indirect
    // light landing on a diffuse surface, and it gets reflected by the base
    // colour exactly like direct light does.
    //
    // Metals take the hemisphere alone. Their indirect term is a REFLECTION,
    // sampled along the reflected view direction (metalAmbient()); a wrap
    // diffuse term has no direction to reflect and adding it would just paint
    // a diffuse lobe back onto the one surface type defined by not having one.
    // The chains and the padlock hang in torchlight and get it directly.
    float aw = ambientShare();
    vec3 ambient = metal ? metalAmbient(N, V, mSG, roughG, ubo.F0)
                         : hemisphericAmbient(N, mD) + bounce * gubo.ambientBounce * mD;
    vec3 color = Lo * (1.0 - aw) + ambient * aw;

    if(heatmap) {
        float intensity = dot(color, vec3(0.2126, 0.7152, 0.0722));
        outColor = vec4(heatmapRamp(intensity), 1.0);
        return;
    }

    // See LIGHT_DEBUG_SHADOW_GAP. Sits after the light loop because that is
    // what fills dbgGap, and before the glow/tone-map tail because none of
    // that means anything in a false-colour view.
    if(debugOn(LIGHT_DEBUG_SHADOW_GAP)) {
        vec3 dbg;
        if(!dbgCube)                       dbg = vec3(0.25);
        else if(dbgGap <= 0.0)             dbg = vec3(0.0, 1.0, 0.0);
        else if(dbgGap <= dbgBias)         dbg = vec3(1.0, 0.0, 0.0);
        else if(dbgGap < dbgBias + dbgSoft) dbg = vec3(1.0, 1.0, 0.0);
        else                               dbg = vec3(0.0, 0.0, 0.4);
        outColor = vec4(dbg, 1.0);
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
    if(ubo.glow != 0.0) {
        // ubo.glow (set in main.cpp's updateUniformBuffer, see gazedGlowKind
        // and gazedInteractionDisabled there) packs two things into one
        // scalar: rounded magnitude selects the category color (1 = Door,
        // gold; 2 = Pickup, blue/purple; 3 = Candle, warm flame orange;
        // 4 = WallTorch, bright fire yellow), then a negative sign overrides
        // that with red -- the player is
        // aimed at something disabled right now (e.g. a locked door with no
        // matching key, or a candle with nothing to light it with). Same
        // aura either way, just a different color, so everything below is
        // shared.
        const vec3 GLOW_COLOR_DOOR = vec3(1.0, 0.78, 0.25);
        const vec3 GLOW_COLOR_PICKUP = vec3(0.55, 0.35, 1.0);
        // Hotter and redder than the door's gold, so the two read apart at a
        // glance: a candle promises FIRE, and the aura is the only cue the
        // player gets before pressing the key.
        const vec3 GLOW_COLOR_CANDLE = vec3(1.0, 0.45, 0.10);
        // Brighter and yellower than the candle's ember orange: a burning
        // wall torch is a strong light source, and this is the cue that it's
        // the thing to light your own torch from.
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
        color = mix(color, color * 0.15, edgeOutline * glowStrength);

        float flow = 0.5 + 0.5 * sin(dot(fragPos, vec3(1.3, 0.9, 1.1)) * 2.2 - ubo.time * 2.0);
        vec3 glowRaw = GLOW_COLOR * glowStrength * rim * flow * 0.9;

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

    // Distance fog: fades toward black (matching buildPostAttachments()'s
    // background clear colour, see main.cpp) as fragPos gets further from
    // the eye, so the geometry visibility cull (GEOM_CULL_* in main.cpp)
    // reads as things dissolving into a dark, atmospheric haze -- the classic
    // "heavy fog hides the draw distance" trick -- rather than popping out of
    // view partway through the screen. Exponential-SQUARED falloff (as
    // opposed to plain exponential) rather than a hard linear ramp: it stays
    // close to 1 near the camera, where clarity still matters for gameplay
    // (reading a door, spotting a key), and only starts biting hard in the
    // back half of its range, right where the cull needs it to. Computed in
    // HDR linear space, before Composite.frag's tone map: fogging AFTER
    // tonemapping would fight the display-referred curve instead of blending
    // like a physical haze would.
    float fogDist = length(fragPos - gubo.eyePos);
    float fogFactor = exp(-pow(gubo.fogDensity * fogDist, 2.0));
    color = mix(vec3(0.0), color, fogFactor);

    // Written linear and unclamped into an R16G16B16A16_SFLOAT attachment, so
    // a surface that receives more than a unit of light keeps saying so rather
    // than being cut off at white. Composite.frag tone maps it down at the end
    // of the chain; the bloom passes in between are what read the excess.
    outColor = vec4(color, 1.0);
}
