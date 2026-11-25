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

/// HUD / menus (p2canvas): port of canvas.wgsl. Maths in linear space like the Bevy version; the
/// texels are sRGB bytes and the framebuffer is not, so decode on input and encode on output.
pub const CANVAS_VS: &str = r#"#version 300 es
precision highp float;
layout(location=0) in vec3 aPos;
layout(location=2) in vec2 aUv;
layout(location=4) in vec4 aCol;
uniform mat4 uVP;
out vec2 vUv;
out vec4 vCol;
void main() {
    gl_Position = uVP * vec4(aPos, 1.0);
    vUv = aUv;
    vCol = aCol;
}
"#;

pub const CANVAS_FS: &str = r#"#version 300 es
precision highp float;
uniform sampler2D uTex0;
uniform sampler2D uTex1;
uniform sampler2D uTex2;
uniform vec4 uTint;
uniform vec4 uSlotColor[3];
uniform vec4 uXu[3];
uniform vec4 uXv[3];
uniform ivec3 uKinds;
uniform ivec4 uOps;       // color_op, alpha_op, invert_mask, mode (0 plain, 1 combiner, 2 shader opacity)
uniform vec4 uParams;     // modulate, alpha test (negative = off)
in vec2 vUv;
in vec4 vCol;
out vec4 oColor;
vec2 xf(int i) {
    return vec2(uXu[i].x * vUv.x + uXu[i].y * vUv.y + uXu[i].z, uXv[i].x * vUv.x + uXv[i].y * vUv.y + uXv[i].z);
}
vec4 lin(vec4 c) { return vec4(pow(c.rgb, vec3(2.2)), c.a); }
vec4 pick(int kind, vec4 s, vec4 color) {
    if (kind == 1) return s * color;
    if (kind == 2) return color;
    return vec4(1.0);
}
void main() {
    vec4 s0 = lin(texture(uTex0, xf(0)));
    vec4 s1 = lin(texture(uTex1, xf(1)));
    vec4 s2 = lin(texture(uTex2, xf(2)));
    vec4 c1 = pick(uKinds.x, s0, uSlotColor[0]);
    vec4 c2 = pick(uKinds.y, s1, uSlotColor[1]);
    vec4 mk = pick(uKinds.z, s2, uSlotColor[2]);
    vec4 o = c1;
    if (uOps.w == 1) {
        float ma = mk.a;
        if (uOps.z == 1) ma = 1.0 - ma;
        vec3 rgb = c1.rgb;
        if (uOps.x == 1) rgb = c2.rgb;
        else if (uOps.x == 2) rgb = c1.rgb * c2.rgb;
        else if (uOps.x == 3) rgb = c1.rgb + c2.rgb;
        else if (uOps.x == 4) rgb = c1.rgb - c2.rgb;
        else if (uOps.x == 5) rgb = mix(c2.rgb, c1.rgb, ma);
        else if (uOps.x == 6) rgb = c1.rgb + c2.rgb * ma;
        else if (uOps.x == 7) rgb = mk.rgb;
        float a = ma;
        if (uOps.y == 1) a = c1.a * c2.a;
        else if (uOps.y == 2) a = c1.a + c2.a;
        else if (uOps.y == 3) a = c1.a;
        else if (uOps.y == 4) a = c2.a;
        o = vec4(clamp(rgb * uParams.x, 0.0, 1.0), clamp(a, 0.0, 1.0));
    } else if (uOps.w == 2) {
        o = vec4(c1.rgb, mk.a);
    }
    vec4 color = o * uTint * vCol;
    if (uParams.y >= 0.0 && color.a < uParams.y) discard;
    oColor = vec4(pow(max(color.rgb, vec3(0.0)), vec3(1.0 / 2.2)), color.a);
}
"#;
