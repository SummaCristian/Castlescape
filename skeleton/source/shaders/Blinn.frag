#version 450
#extension GL_ARB_separate_shader_objects : enable

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

layout(binding = 0, set = 0) uniform GlobalUniformBufferObject {
    vec3 lightDir;
    vec4 lightColor;
    vec3 eyePos;
} gubo;

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
    vec3 L = normalize(-gubo.lightDir);

    // Rendering equation reduced to a single direct light (L09 slide 60):
    // the light's radiance times the BRDF. No distance term, a direct light is
    // infinitely far away so its direction and intensity are the same
    // everywhere in the scene.
    vec3 Lo = gubo.lightColor.rgb * BRDF(N, L, V, mD, ubo.mS, ubo.specPower);

    // Crude stand-in for indirect light: a constant times the base color. It is
    // the simplest possible approximation of ambient lighting and it is meant to
    // be replaced by the hemispheric term (E07) rather than kept.
    vec3 ambient = 0.015 * mD;

    vec3 color = toneMap(Lo + ambient);
    outColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}
