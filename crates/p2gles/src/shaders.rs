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
    if (uAlphaTest && t.a < 0.5) discard;
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
