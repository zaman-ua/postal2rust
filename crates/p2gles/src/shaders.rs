pub const VS: &str = r#"#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec2 aUv;
layout(location=2) in vec2 aUv2;
layout(location=3) in vec4 aCol;
uniform mat4 uVP;
uniform vec3 uCam;
out vec2 vUv;
out vec2 vUv2;
out vec3 vCol;
out float vDist;
invariant gl_Position;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vUv = aUv;
    vUv2 = aUv2;
    vCol = aCol.rgb;
    vDist = length(aPos - uCam);
}
"#;

/// Everything is computed in gamma (sRGB-encoded) space like UE2's 8-bit framebuffer; this equals
/// the Bevy port's linear maths (diffuse * pow(light, 2.2)) up to the sRGB/2.2 approximation.
pub const FS: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform sampler2D uLm;
uniform sampler2D uAlphaMap;
uniform int uMode;        // 0 constant light, 1 lightmap, 2 vertex light, 3 unlit, 4 unlit x2 (modulate)
uniform vec3 uConst;
uniform bool uAlphaTest;
uniform bool uForceOpaque;
uniform bool uUseAlphaMap;
uniform vec2 uFog;        // start, end
uniform vec3 uFogColor;
in vec2 vUv;
in vec2 vUv2;
in vec3 vCol;
in float vDist;
out vec4 oColor;
void main() {
    vec4 t = texture(uTex, vUv);
    if (uForceOpaque) t.a = 1.0;
#ifdef ALPHA_TEST
    // Only the masked program has discard: on Mali a discard anywhere in a program disables
    // early depth test / Forward Pixel Kill for every draw that uses it.
    if (uAlphaTest && t.a < 0.5) discard;
#endif
    vec3 light;
    if (uMode == 1) light = texture(uLm, vUv2).rgb * 2.0;
    else if (uMode == 2) light = vCol * 2.0;
    else if (uMode == 3) light = vec3(1.0);
    else if (uMode == 4) light = vec3(2.0);
    else light = uConst;
    float a = t.a;
    if (uUseAlphaMap) a *= texture(uAlphaMap, vUv2).r;
    vec3 c = min(t.rgb * light, vec3(1.0));
    float f = clamp((vDist - uFog.x) / max(uFog.y - uFog.x, 1.0), 0.0, 1.0);
    c = mix(c, uFogColor, f);
    oColor = vec4(c, a);
}
"#;

/// Generic mesh pass (entities from the game's ECS: characters, pickups, doors, effects, sky).
pub const DYN_VS: &str = r#"#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNrm;
layout(location=2) in vec2 aUv;
layout(location=3) in vec2 aUv2;
layout(location=4) in vec4 aCol;
layout(location=5) in vec4 aW;
layout(location=6) in uvec4 aJ;
uniform mat4 uVP;
uniform mat4 uModel;
uniform mat4 uJoints[64];
uniform bool uSkin;
uniform vec3 uCam;
uniform mat3 uUvM;
out vec2 vUv;
out vec2 vUv2;
out vec4 vCol;
out vec3 vNrm;
out float vDist;
invariant gl_Position;
void main() {
    vec4 p = vec4(aPos, 1.0);
    vec3 n = aNrm;
    if (uSkin) {
        mat4 m = aW.x * uJoints[aJ.x] + aW.y * uJoints[aJ.y] + aW.z * uJoints[aJ.z] + aW.w * uJoints[aJ.w];
        p = m * p;
        n = mat3(m) * n;
    } else {
        p = uModel * p;
        n = mat3(uModel) * n;
    }
    gl_Position = uVP * p;
    vUv = (uUvM * vec3(aUv, 1.0)).xy;
    vUv2 = aUv2;
    vCol = aCol;
    vNrm = n;
    vDist = length(p.xyz - uCam);
}
"#;

pub const DYN_FS: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D uTex;
uniform sampler2D uLm;
uniform int uMode;        // 0 dynamic light, 1 lightmap, 2 vertex light, 3 unlit, 4 unlit x2 (modulate)
uniform vec4 uColor;      // material base colour (gamma space)
uniform float uCutoff;
uniform vec3 uAmb;
uniform vec3 uSunDir;     // direction the light travels
uniform vec3 uSunCol;
uniform vec2 uFog;
uniform vec3 uFogColor;
uniform float uFogMix;    // 0: fog off (additive / modulate keep their colour), 1 on
in vec2 vUv;
in vec2 vUv2;
in vec4 vCol;
in vec3 vNrm;
in float vDist;
out vec4 oColor;
void main() {
    vec4 t = texture(uTex, vUv) * uColor;
#ifdef ALPHA_TEST
    if (t.a < uCutoff) discard;
#endif
    vec3 light;
    if (uMode == 1) light = texture(uLm, vUv2).rgb * 2.0;
    else if (uMode == 2) light = pow(max(vCol.rgb, vec3(0.0)), vec3(1.0 / 2.2));
    else if (uMode == 3) light = vec3(1.0);
    else if (uMode == 4) light = vec3(2.0);
    else {
        vec3 n = normalize(vNrm);
        light = uAmb + uSunCol * max(dot(n, -uSunDir), 0.0);
    }
    vec3 c = min(t.rgb * light, vec3(1.0));
    float f = clamp((vDist - uFog.x) / max(uFog.y - uFog.x, 1.0), 0.0, 1.0) * uFogMix;
    c = mix(c, uFogColor, f);
    oColor = vec4(c, t.a);
}
"#;
