//! The 3D mesh pass: skinned, instanced and blended draws with UE2-style lighting.

use super::{Dynamic, MAX_JOINTS};
use crate::program;
use crate::shaders::{DYN_FS, DYN_VS};
use bevy_math::{Mat4, Vec3};
use glow::HasContext;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blend {
    Opaque,
    /// Alpha test at `Draw::cutoff`.
    Mask,
    /// Source alpha over.
    Alpha,
    /// Additive.
    Add,
    /// Destination * source.
    Multiply,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Light {
    /// Ambient + sun (dynamic actors).
    Dynamic,
    Lightmap,
    /// Per-vertex baked light in the colour attribute (linear, as the game stores it).
    Vertex,
    Unlit,
    /// Unlit, texture * 2 (UE2 modulate).
    Unlit2x,
}

#[derive(Clone)]
pub struct Draw {
    pub mesh: u64,
    pub tex: Option<u64>,
    pub lightmap: Option<u64>,
    pub model: Mat4,
    /// Final skinning matrices (joint world * inverse bind). Replaces `model` when present.
    pub joints: Option<std::sync::Arc<Vec<Mat4>>>,
    /// Base colour, gamma space.
    pub color: [f32; 4],
    pub blend: Blend,
    pub cutoff: f32,
    pub two_sided: bool,
    pub light: Light,
    /// UV transform, column-major 3x3 (`Mat3::to_cols_array`).
    pub uv: [f32; 9],
    /// Sort key for blended draws (distance to the camera).
    pub depth: f32,
    /// Render after the rest with a cleared depth buffer (first-person weapon).
    pub overlay: bool,
}

pub const UV_IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[derive(Clone, Copy)]
pub struct Env {
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_color: [f32; 3],
    pub ambient: [f32; 3],
    pub sun_dir: Vec3,
    pub sun_color: [f32; 3],
}

/// Per-draw uniforms last sent to a program during the current pass: a uniform keeps its value,
/// so only changes are sent.
#[derive(Default)]
struct Sent {
    inst: Option<bool>,
    skin: Option<bool>,
    /// Identity of the skeleton in `uJoints` (the sections of a character share one).
    joints: Option<*const Mat4>,
    uv: Option<[f32; 9]>,
    mode: Option<i32>,
    color: Option<[f32; 4]>,
    cutoff: Option<f32>,
    fog_mix: Option<f32>,
}

/// Records `value`; true if it differs from what was sent before.
fn changed<T: PartialEq>(sent: &mut Option<T>, value: T) -> bool {
    if sent.as_ref() == Some(&value) {
        return false;
    }
    *sent = Some(value);
    true
}

pub(super) struct Prog {
    p: glow::Program,
    vp: Option<glow::UniformLocation>,
    model: Option<glow::UniformLocation>,
    joints: Option<glow::UniformLocation>,
    skin: Option<glow::UniformLocation>,
    inst: Option<glow::UniformLocation>,
    cam: Option<glow::UniformLocation>,
    uvm: Option<glow::UniformLocation>,
    mode: Option<glow::UniformLocation>,
    color: Option<glow::UniformLocation>,
    cutoff: Option<glow::UniformLocation>,
    amb: Option<glow::UniformLocation>,
    sun_dir: Option<glow::UniformLocation>,
    sun_col: Option<glow::UniformLocation>,
    fog: Option<glow::UniformLocation>,
    fog_color: Option<glow::UniformLocation>,
    fog_mix: Option<glow::UniformLocation>,
}

pub(super) fn compile(gl: &glow::Context, masked: bool) -> Prog {
    unsafe {
        let fs = if masked { program::with_define(DYN_FS, "ALPHA_TEST") } else { DYN_FS.into() };
        let p = program::link(gl, "dyn", DYN_VS, &fs);
        gl.use_program(Some(p));
        let l = |n: &str| gl.get_uniform_location(p, n);
        if let Some(t) = l("uTex") {
            gl.uniform_1_i32(Some(&t), 0);
        }
        if let Some(t) = l("uLm") {
            gl.uniform_1_i32(Some(&t), 1);
        }
        Prog {
            p,
            vp: l("uVP"),
            model: l("uModel"),
            joints: l("uJoints[0]"),
            skin: l("uSkin"),
            inst: l("uInst"),
            cam: l("uCam"),
            uvm: l("uUvM"),
            mode: l("uMode"),
            color: l("uColor"),
            cutoff: l("uCutoff"),
            amb: l("uAmb"),
            sun_dir: l("uSunDir"),
            sun_col: l("uSunCol"),
            fog: l("uFog"),
            fog_color: l("uFogColor"),
            fog_mix: l("uFogMix"),
        }
    }
}

impl Dynamic {
    /// Draws `draws` (opaque and alpha-tested first, blended far to near, overlays last). The caller
    /// has cleared and set the viewport; depth state is set here.
    pub fn render(&mut self, gl: &glow::Context, vp: Mat4, cam: Vec3, env: &Env, draws: &mut Vec<Draw>) {
        self.draws = 0;
        self.tris = 0;
        let rank = |d: &Draw| -> u8 {
            (d.overlay as u8) * 4
                + match d.blend {
                    Blend::Opaque => 0,
                    Blend::Mask => 1,
                    _ => 2,
                }
        };
        // Stable order: pass, then (opaque/mask) by mesh+texture to cut state changes, blended far -> near.
        draws.sort_by(|a, b| {
            rank(a).cmp(&rank(b)).then_with(|| {
                if rank(a) % 4 == 2 {
                    b.depth.partial_cmp(&a.depth).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    (a.tex, a.mesh).cmp(&(b.tex, b.mesh))
                }
            })
        });
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.front_face(glow::CCW);
            gl.active_texture(glow::TEXTURE0);
            let mut cur_prog = usize::MAX;
            let mut sent: [Sent; 2] = Default::default();
            let mut joint_data: Vec<f32> = Vec::new();
            let mut cur_vao = None;
            let mut cur_tex: Option<Option<u64>> = None;
            let mut cur_state: Option<(Blend, bool)> = None;
            let mut overlay_cleared = false;
            let mut cur_cw = false;
            let mut i = 0;
            let mut inst_data: Vec<f32> = Vec::new();
            while i < draws.len() {
                let d = &draws[i];
                // Run of identical draws that differ only by transform: one instanced call.
                let batchable = d.joints.is_none() && d.blend != Blend::Alpha && d.blend != Blend::Add && d.blend != Blend::Multiply;
                let mut run = 1;
                if batchable {
                    while i + run < draws.len() {
                        let o = &draws[i + run];
                        if o.mesh != d.mesh || o.joints.is_some() || o.tex != d.tex || o.lightmap != d.lightmap || o.blend != d.blend || o.cutoff != d.cutoff || o.two_sided != d.two_sided || o.light != d.light || o.color != d.color || o.uv != d.uv || o.overlay != d.overlay {
                            break;
                        }
                        run += 1;
                    }
                }
                i += run;
                let Some(m) = self.meshes.get(&d.mesh) else { continue };
                if d.overlay && !overlay_cleared {
                    gl.depth_mask(true);
                    gl.clear(glow::DEPTH_BUFFER_BIT);
                    overlay_cleared = true;
                }
                let pi = (d.blend == Blend::Mask) as usize;
                let p = &self.progs[pi];
                if cur_prog != pi {
                    gl.use_program(Some(p.p));
                    cur_prog = pi;
                    gl.uniform_matrix_4_f32_slice(p.vp.as_ref(), false, &vp.to_cols_array());
                    gl.uniform_3_f32(p.cam.as_ref(), cam.x, cam.y, cam.z);
                    gl.uniform_3_f32(p.amb.as_ref(), env.ambient[0], env.ambient[1], env.ambient[2]);
                    gl.uniform_3_f32(p.sun_dir.as_ref(), env.sun_dir.x, env.sun_dir.y, env.sun_dir.z);
                    gl.uniform_3_f32(p.sun_col.as_ref(), env.sun_color[0], env.sun_color[1], env.sun_color[2]);
                    gl.uniform_2_f32(p.fog.as_ref(), env.fog_start, env.fog_end);
                    gl.uniform_3_f32(p.fog_color.as_ref(), env.fog_color[0], env.fog_color[1], env.fog_color[2]);
                }
                if cur_state != Some((d.blend, d.two_sided)) {
                    match d.blend {
                        Blend::Opaque | Blend::Mask => {
                            gl.disable(glow::BLEND);
                            gl.depth_mask(true);
                        }
                        Blend::Alpha => {
                            gl.enable(glow::BLEND);
                            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                            gl.depth_mask(false);
                        }
                        Blend::Add => {
                            gl.enable(glow::BLEND);
                            gl.blend_func(glow::SRC_ALPHA, glow::ONE);
                            gl.depth_mask(false);
                        }
                        Blend::Multiply => {
                            gl.enable(glow::BLEND);
                            gl.blend_func(glow::DST_COLOR, glow::ZERO);
                            gl.depth_mask(false);
                        }
                    }
                    if d.two_sided {
                        gl.disable(glow::CULL_FACE);
                    } else {
                        gl.enable(glow::CULL_FACE);
                        gl.cull_face(glow::BACK);
                    }
                    cur_state = Some((d.blend, d.two_sided));
                }
                let tex = d.tex.filter(|t| self.textures.contains_key(t));
                if cur_tex != Some(tex) {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex.map(|t| self.textures[&t]).unwrap_or(self.white)));
                    cur_tex = Some(tex);
                }
                if d.light == Light::Lightmap {
                    gl.active_texture(glow::TEXTURE1);
                    gl.bind_texture(glow::TEXTURE_2D, Some(d.lightmap.and_then(|t| self.textures.get(&t).copied()).unwrap_or(self.white)));
                    gl.active_texture(glow::TEXTURE0);
                }
                if cur_vao != Some(d.mesh) {
                    // The index buffer binding is part of the vertex array.
                    gl.bind_vertex_array(Some(m.vao));
                    cur_vao = Some(d.mesh);
                }
                // Mirrored transforms (negative scale) flip the winding.
                let cw = d.joints.is_none() && d.model.determinant() < 0.0;
                if cw != cur_cw {
                    gl.front_face(if cw { glow::CW } else { glow::CCW });
                    cur_cw = cw;
                }
                let instanced = run > 1;
                let sent = &mut sent[pi];
                if changed(&mut sent.inst, instanced) {
                    gl.uniform_1_i32(p.inst.as_ref(), instanced as i32);
                }
                if instanced {
                    inst_data.clear();
                    for o in &draws[i - run..i] {
                        inst_data.extend_from_slice(&o.model.to_cols_array());
                    }
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.inst_vbo));
                    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, std::slice::from_raw_parts(inst_data.as_ptr() as *const u8, inst_data.len() * 4), glow::STREAM_DRAW);
                    if !m.instanced.replace(true) {
                        for k in 0..4u32 {
                            gl.enable_vertex_attrib_array(7 + k);
                            gl.vertex_attrib_pointer_f32(7 + k, 4, glow::FLOAT, false, 64, k as i32 * 16);
                            gl.vertex_attrib_divisor(7 + k, 1);
                        }
                    }
                } else if m.instanced.replace(false) {
                    for k in 0..4u32 {
                        gl.disable_vertex_attrib_array(7 + k);
                    }
                }
                if changed(&mut sent.skin, d.joints.is_some()) {
                    gl.uniform_1_i32(p.skin.as_ref(), d.joints.is_some() as i32);
                }
                match &d.joints {
                    Some(j) => {
                        if changed(&mut sent.joints, j.as_ptr()) {
                            joint_data.clear();
                            joint_data.extend(j.iter().take(MAX_JOINTS).flat_map(|m| m.to_cols_array()));
                            gl.uniform_matrix_4_f32_slice(p.joints.as_ref(), false, &joint_data);
                        }
                    }
                    // Instances take their transform from the attributes.
                    None if !instanced => gl.uniform_matrix_4_f32_slice(p.model.as_ref(), false, &d.model.to_cols_array()),
                    None => {}
                }
                if changed(&mut sent.uv, d.uv) {
                    gl.uniform_matrix_3_f32_slice(p.uvm.as_ref(), false, &d.uv);
                }
                let mode = match d.light {
                    Light::Dynamic => 0,
                    Light::Lightmap => 1,
                    Light::Vertex => 2,
                    Light::Unlit => 3,
                    Light::Unlit2x => 4,
                };
                if changed(&mut sent.mode, mode) {
                    gl.uniform_1_i32(p.mode.as_ref(), mode);
                }
                if changed(&mut sent.color, d.color) {
                    gl.uniform_4_f32(p.color.as_ref(), d.color[0], d.color[1], d.color[2], d.color[3]);
                }
                if changed(&mut sent.cutoff, d.cutoff) {
                    gl.uniform_1_f32(p.cutoff.as_ref(), d.cutoff);
                }
                let fog_mix = if matches!(d.blend, Blend::Add | Blend::Multiply) { 0.0 } else { 1.0 };
                if changed(&mut sent.fog_mix, fog_mix) {
                    gl.uniform_1_f32(p.fog_mix.as_ref(), fog_mix);
                }
                if instanced {
                    gl.draw_elements_instanced(glow::TRIANGLES, m.count, glow::UNSIGNED_INT, 0, run as i32);
                } else {
                    gl.draw_elements(glow::TRIANGLES, m.count, glow::UNSIGNED_INT, 0);
                }
                self.draws += 1;
                self.tris += (m.count as u64 / 3) * run as u64;
            }
            gl.front_face(glow::CCW);
            gl.bind_vertex_array(None);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
        }
    }
}
