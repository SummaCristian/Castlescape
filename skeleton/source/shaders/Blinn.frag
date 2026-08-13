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
// names L09 gives them (mD, mS, N, L, V, h), so the code below can be read
// side by side with the slides. Everything else is spelled out. The one
// exception is the specular exponent, which the slides call gamma: that word
// already means the display gamma in this same file (see the end of main), so
// it is specPower here and specularPower in materials.json.
//
// Same block the vertex shader declares, and the same set/binding: the material
// parameters live next to the matrices because both are per-instance. Only mS
// and specPower are read here, the matrices are the vertex stage's business.
layout(binding = 0, set = 1) uniform UniformBufferObject {
    mat4 mvpMat;
    mat4 mMat;
    mat4 nMat;
    vec3 mS;          // specular color of the material
    float specPower;  // specular exponent: high = small tight highlight
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
    Light lights[MAX_LIGHTS];
} gubo;

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

// The BRDF, as the sum of a diffuse and a specular term (L09 slide 38).
//
//   N   surface normal            L   direction towards the light
//   V   direction towards the eye
//   mD  diffuse color (the "base color" of the surface, L09 slide 59)
//   mS  specular color            specPower  specular exponent (gamma on the slides)
//
// Both terms are clamped at zero. Without that, a surface facing away from the
// light would have a negative cosine and the formula would *subtract* light
// from the object (L09 slide 59).
vec3 BRDF(vec3 N, vec3 L, vec3 V, vec3 mD, vec3 mS, float specPower) {
    // Lambert diffuse. Proportional to cos(alpha) between normal and light,
    // and independent of V: a matte surface looks the same from every angle.
    vec3 diffuse = mD * clamp(dot(N, L), 0.0, 1.0);

    // Blinn specular. h is the half vector, halfway between the light and the
    // viewer; the angle between N and h stands in for the angle between the
    // mirror-reflected ray and the viewer, which is what Phong computes the
    // expensive way. Cheaper than Phong and better behaved at grazing angles.
    vec3 h = normalize(L + V);
    vec3 specular = mS * pow(clamp(dot(N, h), 0.0, 1.0), specPower);

    return diffuse + specular;
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
            * BRDF(N, L, V, mD, ubo.mS, ubo.specPower);
    }

    // Crude stand-in for indirect light: a constant times the base color. It is
    // the simplest possible approximation of ambient lighting and it is meant to
    // be replaced by the hemispheric term (E07) rather than kept.
    vec3 ambient = 0.015 * mD;

    vec3 color = toneMap(Lo + ambient);
    outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
