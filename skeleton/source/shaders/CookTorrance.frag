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
    int flatNormals;  // 1: ignore the vertex normal, use the face's own
    // Unused here, declared to keep this block identical to the one Flame.vert
    // and Flame.frag see: both pipelines share DSLlocal and one C++ struct.
    float time;
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
    // -1: unshadowed (the lanterns, the spot). Else which slot of shadowMaps
    // / lightSpace below holds this light's shadow map. Set by SceneLights
    // from lights.json's "castsShadow", see the struct comment there.
    int shadowIndex;
};

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    vec3 ambientUpper;   // indirect light from the sky
    vec3 ambientLower;   // indirect light bounced off the ground
    vec3 ambientDir;     // axis the two blend along, i.e. world up
    int debugFlags;      // LIGHT_DEBUG_* bits, set by the cheat menu
    Light lights[MAX_LIGHTS];
} gubo;

// Shadow sampling, set 2: its own descriptor set because it belongs to
// neither "once a frame" (set 0) nor "once an object" (set 1) -- it's once
// per SHADOW-CASTING LIGHT, six fixed slots that exist for the run of the
// program. lightSpace is the same view-projection matrix Shadow.vert used to
// render each map, needed again here to place fragPos in that light's space.
//
// Six SEPARATE sampler bindings rather than one binding declared as an array
// of 6: Scene::init's descriptor-pool accounting (Scene.hpp, the loop that
// does `texturesInPool += 1` per binding) counts bindings, not the
// descriptors an array binding actually needs, and every existing binding in
// this project has count 1. An array binding would silently under-reserve
// the pool. Six ordinary bindings sidestep that instead of relying on a path
// nothing else here exercises.
layout(binding = 0, set = 2) uniform ShadowUniformBufferObject {
    mat4 lightSpace[NUM_SHADOW_LIGHTS];
} shadowUbo;

layout(binding = 1, set = 2) uniform sampler2D shadowMap0;
layout(binding = 2, set = 2) uniform sampler2D shadowMap1;
layout(binding = 3, set = 2) uniform sampler2D shadowMap2;
layout(binding = 4, set = 2) uniform sampler2D shadowMap3;
layout(binding = 5, set = 2) uniform sampler2D shadowMap4;
layout(binding = 6, set = 2) uniform sampler2D shadowMap5;

// Stands in for shadowMaps[idx], which the six-separate-bindings choice above
// rules out. NUM_SHADOW_LIGHTS is 6 (LightConstants.glsl); if that ever
// changes, a case has to be added or removed here by hand.
float sampleShadowMap(int idx, vec2 uv) {
    if(idx == 0) return texture(shadowMap0, uv).r;
    if(idx == 1) return texture(shadowMap1, uv).r;
    if(idx == 2) return texture(shadowMap2, uv).r;
    if(idx == 3) return texture(shadowMap3, uv).r;
    if(idx == 4) return texture(shadowMap4, uv).r;
    return texture(shadowMap5, uv).r;
}

// 1.0: fully lit. 0.0: this light's shadow map says something else is closer
// to the light than `pos` is, i.e. `pos` is in shadow. shadowIndex < 0 skips
// the lookup entirely (the lanterns and the spot: see the scope note in
// notes.md on why only the sun and the torches got this).
float shadowFactor(int shadowIndex, vec3 pos) {
    // Shadows off (cheat menu): light everything as if no map existed. Reads
    // gubo.debugFlags directly rather than through debugOn(), which is
    // declared further down the file.
    if(shadowIndex < 0 || (gubo.debugFlags & LIGHT_DEBUG_NO_SHADOWS) != 0) {
        return 1.0;
    }

    vec4 lightClip = shadowUbo.lightSpace[shadowIndex] * vec4(pos, 1.0);
    // w is 1 for the sun's orthographic matrix and only actually divides
    // anything for the torches' perspective ones, but doing it unconditionally
    // costs nothing and keeps this one code path for both projection kinds.
    vec3 lightNDC = lightClip.xyz / lightClip.w;

    // GLM_FORCE_DEPTH_ZERO_TO_ONE (Starter.hpp) means lightNDC.z is already
    // Vulkan's 0..1 depth range, same as what's stored in the shadow map; only
    // XY need remapping from NDC's -1..1 to a texture's 0..1.
    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;

    // Outside the map (a torch's cone, or the sun's fixed ortho box, doesn't
    // reach here): nothing to compare against, so don't shadow it. Missing
    // this check would sample garbage at the map's clamped edge instead.
    if(shadowUV.x < 0.0 || shadowUV.x > 1.0 ||
       shadowUV.y < 0.0 || shadowUV.y > 1.0 ||
       lightNDC.z < 0.0 || lightNDC.z > 1.0) {
        return 1.0;
    }

    float closestDepth = sampleShadowMap(shadowIndex, shadowUV);
    // Shader-side depth bias: Pipeline::create (Starter.hpp) hard-codes
    // depthBiasEnable false, so there is no hardware slope-scaled bias
    // available, and this is the substitute. Too small and most of the scene
    // shadows itself in stripes ("acne"); too large and shadows visibly
    // detach from their casters ("peter-panning"). 0.0015 is a starting point
    // for this scene's depth ranges, not a derived value.
    const float bias = 0.0015;
    return (lightNDC.z - bias > closestDepth) ? 0.0 : 1.0;
}

// Whether one of the debug views from LightConstants.glsl is on. All of them
// are off in a normal frame, so this is a uniform branch: every pixel of every
// draw takes the same side of it, which is the cheap kind on a GPU.
bool debugOn(int flag) {
    return (gubo.debugFlags & flag) != 0;
}

const float PI = 3.14159265359;

// Hemispheric ambient, E07 s.47-54. Indirect light, blended by which way the
// surface faces: aligned with ambientDir is all sky, opposite is all ground.
vec3 hemisphericAmbient(vec3 N, vec3 mD) {
    float w = (dot(N, gubo.ambientDir) + 1.0) / 2.0;   // dot is -1..1, w is 0..1
    return mix(gubo.ambientLower, gubo.ambientUpper, w) * mD;
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
    vec3 radiance = lt.color * pow(lt.g / max(dist, 0.0001), lt.beta);

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

// HDR tone map, L09 s.45. Divides by luminance rather than per-channel, which
// would desaturate highlights towards white.
vec3 toneMap(vec3 c) {
    float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return c / (Y + 1.0);
}

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

    // k is the diffuse share, so forcing it to 1 leaves the specular term
    // multiplied by 0: the highlights go, everything else stays exactly as it
    // was. Done here rather than inside BRDF so that function keeps taking all
    // its inputs as arguments.
    float k = debugOn(LIGHT_DEBUG_NO_SPECULAR) ? 1.0 : ubo.k;

    // Rendering equation: sum over the sources of radiance times BRDF, each
    // term zeroed by shadowFactor() wherever that one light doesn't reach
    // this point. Ambient below is untouched by it on purpose: shadow mapping
    // only ever blocks a light's DIRECT contribution, never the indirect
    // bounce hemisphericAmbient() stands in for -- otherwise a shadow would
    // read as a hole into pure black instead of the dim, indirectly-lit area
    // a real one is.
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < gubo.lightCount; i++) {
        vec3 L = lightDirection(gubo.lights[i], fragPos);
        Lo += lightRadiance(gubo.lights[i], fragPos)
            * BRDF(N, L, V, mD, ubo.mS, ubo.roughness, ubo.F0, k)
            * shadowFactor(gubo.lights[i].shadowIndex, fragPos);
    }

    vec3 color = Lo + hemisphericAmbient(N, mD);

    // Debug view: no tone map, so anything above 1 is clipped by the hardware
    // instead of being compressed back into range. Flat white areas are where
    // the tone map was doing the work.
    if(!debugOn(LIGHT_DEBUG_NO_TONEMAP)) {
        color = toneMap(color);
    }

    // Written linear, not gamma-encoded: the swapchain is B8G8R8A8_SRGB, so the
    // hardware does the linear-to-sRGB encode on write.
    outColor = vec4(color, 1.0);
}
