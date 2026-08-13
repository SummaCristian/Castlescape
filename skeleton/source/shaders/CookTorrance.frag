#version 450
#extension GL_ARB_separate_shader_objects : enable
// glslc's #include support. Not part of core GLSL, which has no include at all.
#extension GL_GOOGLE_include_directive : require

// MAX_LIGHTS and the LIGHT_* type tags, the same file SceneLights.hpp includes.
// Found through the -I that CMake passes to glslc.
#include "custom/LightConstants.glsl"

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Naming convention in this file: the light-model quantities keep the short
// names L09 and E06 give them (mD, mS, N, L, V, h, D, G, F), so the code below
// can be read side by side with the slides. Everything else is spelled out.
// The roughness is `rho` on the slides and `roughness` here, because a single
// Greek letter transliterated is worse than the word it stands for.
//
// Same block the vertex shader declares, and the same set/binding: the material
// parameters live next to the matrices because both are per-instance. Only the
// material is read here, the matrices are the vertex stage's business.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 mS;          // specular color of the material
    float roughness;  // rho: width of the microfacet distribution, 0 = mirror
    float F0;         // reflectance when looking straight at the surface
    float k;          // diffuse share of the BRDF, specular gets (1 - k)
} ubo;

layout(binding = 1, set = 1) uniform sampler2D albedoMap;

struct Light {
    vec3 pos;       // point/spot only
    float g;        // distance at which the light is exactly `color`
    vec3 dir;       // direct: travel direction. spot: aim direction
    float beta;     // decay exponent: 0 constant, 1 linear, 2 quadratic
    vec3 color;     // l, the emitted color
    float cosIn;    // spot: cosine of the half inner angle
    float cosOut;   // spot: cosine of the half outer angle
    int type;
};

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 eyePos;
    int lightCount;
    vec3 ambientUpper;   // indirect light arriving from the sky
    vec3 ambientLower;   // indirect light bounced off the ground
    vec3 ambientDir;     // the axis the two blend along, i.e. world up
    Light lights[MAX_LIGHTS];
} gubo;

// Hemispheric ambient light, E07 slides 47-54. This is the scene's indirect
// lighting: everything that reaches a surface without coming straight from a
// source, which in a real room is most of it.
//
// A constant ambient term claims light arrives equally from every direction.
// Outdoors that is plainly false: a surface facing up sees sky, one facing down
// sees dirt, and the two are different colors. So the term is made to depend on
// the normal, blending the two by the cosine of the angle with `ambientDir`.
// Aligned with it gives pure sky, opposite gives pure ground, perpendicular
// gives half of each.
//
// It costs one dot product and one mix over a constant, and it is what stops
// surfaces facing away from every light from being flat black.
vec3 hemisphericAmbient(vec3 N, vec3 mD) {
    // dot() runs from -1 to 1, the weight has to run from 0 to 1.
    float w = (dot(N, gubo.ambientDir) + 1.0) / 2.0;
    vec3 lA = mix(gubo.ambientLower, gubo.ambientUpper, w);

    // The ambient BRDF term is a constant, and for a diffuse surface that
    // constant is its base color: indirect light is reflected the same way
    // direct light is.
    return lA * mD;
}

// Direction from the shaded point TOWARDS the light, `lx` in L09's notation.
// For a direct light it is a constant: the source is infinitely far away, so
// every point in the scene sees it from the same angle. For point and spot it
// aims at the lamp and therefore changes across the surface, which is the whole
// reason a point light wraps around an object and a direct one does not.
vec3 lightDirection(Light lt, vec3 pos) {
    if(lt.type == LIGHT_DIRECT) {
        return -lt.dir;
    }
    return normalize(lt.pos - pos);
}

// The color that actually arrives, after distance decay and, for a spot, the
// cone. L09 slides 23 and 33.
vec3 lightRadiance(Light lt, vec3 pos) {
    if(lt.type == LIGHT_DIRECT) {
        return lt.color;
    }

    // (g / |p - x|)^beta. Brighter than `color` closer than g, dimmer past it.
    // beta is authored rather than fixed at the physically correct 2, because
    // with no indirect lighting an inverse-square falloff reads as far too dark
    // (L09 slides 20-21).
    float dist = length(lt.pos - pos);
    vec3 radiance = lt.color * pow(lt.g / max(dist, 0.0001), lt.beta);

    if(lt.type == LIGHT_SPOT) {
        // A spot is a point light confined to a cone: same radiance, times a
        // dimming factor that is 1 inside the inner cone, 0 outside the outer
        // one, and linear in between.
        //
        // `dir` is authored as the direction the lamp POINTS, so a lamp aimed at
        // the floor is [0,-1,0], which is the intuitive way to write it. The
        // vector towards the shaded point is therefore -lightDirection(), and
        // the dot product between the two is 1 dead centre in the beam.
        float cosAngle = dot(-lightDirection(lt, pos), lt.dir);
        radiance *= clamp((cosAngle - lt.cosOut) / (lt.cosIn - lt.cosOut), 0.0, 1.0);
    }

    return radiance;
}

const float PI = 3.14159265359;

// ---------------------------------------------------------------------------
// The three terms of the Cook-Torrance specular model (E06 slides 38-47).
//
// The model treats a rough surface as a crowd of microscopic perfect mirrors,
// the microfacets. It never draws them; it describes statistically how many
// face which way. For a given light and viewer, only the facets whose normal is
// exactly h can send light from one to the other, so "how bright is the
// highlight" becomes "what fraction of the facets is oriented along h".
// ---------------------------------------------------------------------------

// D, the distribution term: that fraction. GGX version, E06 slide 45.
// Roughness widens the lobe. What makes GGX preferred over the Blinn
// distribution is its long tail: a narrow core that fades into a wide halo,
// which is how real surfaces behave, instead of dropping off sharply.
float distributionGGX(vec3 N, vec3 h, float roughness) {
    float a2 = roughness * roughness;
    float NdotH = clamp(dot(N, h), 0.0, 1.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// G, the geometric term: microfacets are in relief, so they shadow each other,
// some receiving no light and some reflecting into a blocked direction. It
// matters little head-on and a great deal at grazing angles, where without it a
// rough surface would come out absurdly bright.
//
// This is the microfacet version of E06 slide 47, which takes no parameters at
// all and depends only on the angles.
float geometricTerm(vec3 N, vec3 h, vec3 L, vec3 V) {
    float NdotH = clamp(dot(N, h), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float VdotH = max(dot(V, h), 0.0001);

    return min(1.0, min(2.0 * NdotH * NdotV / VdotH,
                        2.0 * NdotH * NdotL / VdotH));
}

// F, the Fresnel term: the fraction of light reflected rather than entering the
// material, which grows as the view becomes grazing. Schlick's approximation,
// E06 slide 46: it starts at F0 head-on and reaches 1 at the horizon.
//
// This is the term that has no counterpart at all in Blinn, and the one you
// notice most: it is why a matte floor turns into a mirror when you look along
// it. The exponent 5 has no derivation, it is the fit Schlick published.
float fresnelSchlick(vec3 V, vec3 h, float F0) {
    float VdotH = clamp(dot(V, h), 0.0, 1.0);
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}

// The full Cook-Torrance BRDF, in the form of E06 slides 38-39:
//
//   fr = clamp(N.L) * ( k * f_diffuse + (1 - k) * f_specular )
//   f_diffuse  = mD                                   (Lambert, a constant)
//   f_specular = mS * D*F*G / (4 * clamp(N.L) * clamp(N.V))
//
// Two things worth noticing about that shape.
//
// The diffuse and specular halves are *interpolated* by k rather than simply
// added. Adding both at full strength lets a surface reflect more light than it
// receives; interpolating keeps the total in check, which is what the professor
// means by computing the two components in a physically accurate way.
//
// The 4*(N.L)*(N.V) denominator is not a fudge. It is the geometric
// normalization that converts from the area of the microfacets to the area of
// the surface they sit on, and it is what makes the energy work out.
vec3 BRDF(vec3 N, vec3 L, vec3 V, vec3 mD, vec3 mS, float roughness, float F0, float k) {
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float NdotV = clamp(dot(N, V), 0.0, 1.0);
    vec3 h = normalize(L + V);

    float D = distributionGGX(N, h, roughness);
    float G = geometricTerm(N, h, L, V);
    float F = fresnelSchlick(V, h, F0);

    // The denominator goes to zero exactly at the silhouette, where NdotL or
    // NdotV does. The whole term is multiplied by NdotL right after, so the
    // result is zero there anyway; the guard only keeps an inf from being born
    // in the first place, since inf * 0 would be a NaN and NaNs spread.
    vec3 specular = mS * (D * F * G) / max(4.0 * NdotL * NdotV, 0.0001);

    // Lambert for the diffuse half, which is what E06 slide 38 prescribes for
    // this model. Oren-Nayar would go here instead, as its own technique.
    vec3 diffuse = mD;

    // The outer clamp(N.L) is the geometric term of the rendering equation, the
    // one that says a surface lit edge-on receives less energy per unit area.
    // Clamped, so a face turned away from this light contributes zero rather
    // than a negative amount that would eat into what another light put there.
    return NdotL * (k * diffuse + (1.0 - k) * specular);
}

// HDR tone mapping, L09 slide 45. The sum of the light contributions can go
// well past 1.0, and clamping there would flatten every bright area to the same
// white. This compresses instead, dividing by the *luminance* rather than
// per-channel: dividing each channel by itself would desaturate the highlights,
// pulling every bright color towards white.
vec3 toneMap(vec3 c) {
    float Y = dot(c, vec3(0.2126, 0.7152, 0.0722));
    return c / (Y + 1.0);
}

void main() {
    // Interpolated per-vertex normal. Renormalized because linear interpolation
    // across a triangle shortens the vector wherever the three corner normals
    // diverge, which is exactly where smooth shading matters most.
    vec3 N = normalize(fragNorm);

    // Textures are authored in sRGB, the lighting maths needs linear values.
    // The inverse conversion happens on the way out, at the bottom.
    vec3 mD = pow(texture(albedoMap, fragUV).rgb, vec3(2.2));

    vec3 V = normalize(gubo.eyePos - fragPos);

    // The rendering equation for scanline rendering: a sum over the light
    // sources of each one's radiance times the BRDF. Every term is positive
    // (the BRDF clamps both of its own), so a light can only ever add light,
    // never remove what another one put there.
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < gubo.lightCount; i++) {
        vec3 L = lightDirection(gubo.lights[i], fragPos);
        Lo += lightRadiance(gubo.lights[i], fragPos)
            * BRDF(N, L, V, mD, ubo.mS, ubo.roughness, ubo.F0, ubo.k);
    }

    vec3 color = toneMap(Lo + hemisphericAmbient(N, mD));
    outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
