//! Generic mesh pass: meshes handed over per frame by a host (the game's ECS bridge). GPU copies are
//! cached by caller-chosen ids; the host uploads a mesh/texture once and refers to it afterwards.

use crate::renderer::{upload_bc, upload_tex};
use crate::shaders::{DYN_FS, DYN_VS};
use bevy_math::{Mat4, Vec3};
use glow::HasContext;
use p2view::scene::PixelFormat;
use std::collections::{HashMap, HashSet};

pub const MAX_JOINTS: usize = 64;
const STRIDE: usize = 80;

/// CPU mesh as the host has it. Missing attributes are filled with neutral values.
pub struct MeshIn<'a> {
    pub pos: &'a [[f32; 3]],
    pub normal: Option<&'a [[f32; 3]]>,
    pub uv0: Option<&'a [[f32; 2]]>,
    pub uv1: Option<&'a [[f32; 2]]>,
    pub color: Option<&'a [[f32; 4]]>,
    pub joints: Option<&'a [[u16; 4]]>,
    pub weights: Option<&'a [[f32; 4]]>,
    pub indices: &'a [u32],
}

pub struct TexIn<'a> {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// Mip levels, largest first (one is fine: the rest is generated).
    pub mips: Vec<&'a [u8]>,
    pub repeat: bool,
}

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
    pub joints: Option<Vec<Mat4>>,
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

struct GpuMesh {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    ibo: glow::Buffer,
    count: i32,
}

struct Prog {
    p: glow::Program,
    vp: Option<glow::UniformLocation>,
    model: Option<glow::UniformLocation>,
    joints: Option<glow::UniformLocation>,
    skin: Option<glow::UniformLocation>,
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

pub struct Dynamic {
    progs: [Prog; 2],
    meshes: HashMap<u64, GpuMesh>,
    textures: HashMap<u64, glow::Texture>,
    white: glow::Texture,
    aniso: bool,
    pub draws: u32,
    pub tris: u64,
}

fn compile(gl: &glow::Context, masked: bool) -> Prog {
    unsafe {
        let p = gl.create_program().unwrap();
        let fs = if masked { DYN_FS.replacen("#version 300 es\n", "#version 300 es\n#define ALPHA_TEST 1\n", 1) } else { DYN_FS.to_string() };
        for (ty, src) in [(glow::VERTEX_SHADER, DYN_VS.to_string()), (glow::FRAGMENT_SHADER, fs)] {
            let s = gl.create_shader(ty).unwrap();
            gl.shader_source(s, &src);
            gl.compile_shader(s);
            if !gl.get_shader_compile_status(s) {
                panic!("dyn shader: {}", gl.get_shader_info_log(s));
            }
            gl.attach_shader(p, s);
        }
        gl.link_program(p);
        if !gl.get_program_link_status(p) {
            panic!("dyn link: {}", gl.get_program_info_log(p));
        }
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
    pub fn new(gl: &glow::Context) -> Dynamic {
        let aniso = gl.supported_extensions().contains("GL_EXT_texture_filter_anisotropic");
        let progs = [compile(gl, false), compile(gl, true)];
        let white = unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, 1, 1, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&[255, 255, 255, 255])));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            t
        };
        Dynamic { progs, meshes: HashMap::new(), textures: HashMap::new(), white, aniso, draws: 0, tris: 0 }
    }

    pub fn has_mesh(&self, id: u64) -> bool {
        self.meshes.contains_key(&id)
    }

    pub fn has_tex(&self, id: u64) -> bool {
        self.textures.contains_key(&id)
    }

    pub fn add_mesh(&mut self, gl: &glow::Context, id: u64, m: &MeshIn) {
        self.remove_mesh(gl, id);
        let n = m.pos.len();
        if n == 0 || m.indices.is_empty() {
            return;
        }
        let mut buf = Vec::with_capacity(n * STRIDE);
        let f = |buf: &mut Vec<u8>, v: &[f32]| {
            for x in v {
                buf.extend_from_slice(&x.to_ne_bytes());
            }
        };
        for i in 0..n {
            f(&mut buf, &m.pos[i]);
            f(&mut buf, &m.normal.and_then(|a| a.get(i).copied()).unwrap_or([0.0, 1.0, 0.0]));
            f(&mut buf, &m.uv0.and_then(|a| a.get(i).copied()).unwrap_or([0.0; 2]));
            f(&mut buf, &m.uv1.and_then(|a| a.get(i).copied()).unwrap_or([0.0; 2]));
            f(&mut buf, &m.color.and_then(|a| a.get(i).copied()).unwrap_or([1.0; 4]));
            f(&mut buf, &m.weights.and_then(|a| a.get(i).copied()).unwrap_or([1.0, 0.0, 0.0, 0.0]));
            for j in m.joints.and_then(|a| a.get(i).copied()).unwrap_or([0; 4]) {
                buf.extend_from_slice(&j.min(MAX_JOINTS as u16 - 1).to_ne_bytes());
            }
        }
        debug_assert_eq!(buf.len(), n * STRIDE);
        unsafe {
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &buf, glow::STATIC_DRAW);
            let s = STRIDE as i32;
            for (loc, size, off) in [(0u32, 3, 0), (1, 3, 12), (2, 2, 24), (3, 2, 32), (4, 4, 40), (5, 4, 56)] {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, size, glow::FLOAT, false, s, off);
            }
            gl.enable_vertex_attrib_array(6);
            gl.vertex_attrib_pointer_i32(6, 4, glow::UNSIGNED_SHORT, s, 72);
            let ibo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            let bytes: Vec<u8> = m.indices.iter().flat_map(|i| i.to_ne_bytes()).collect();
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &bytes, glow::STATIC_DRAW);
            gl.bind_vertex_array(None);
            self.meshes.insert(id, GpuMesh { vao, vbo, ibo, count: m.indices.len() as i32 });
        }
    }

    pub fn add_tex(&mut self, gl: &glow::Context, id: u64, t: &TexIn) {
        self.remove_tex(gl, id);
        if t.mips.is_empty() || t.width == 0 || t.height == 0 {
            return;
        }
        unsafe {
            let tex = match t.format {
                PixelFormat::Rgba8 => Some(upload_tex(gl, t.width, t.height, &t.mips, t.repeat, self.aniso)),
                f => {
                    let mips: Vec<Vec<u8>> = t.mips.iter().map(|m| m.to_vec()).collect();
                    upload_bc(gl, t.width, t.height, f, &mips, self.aniso)
                }
            };
            if let Some(tex) = tex {
                if !t.repeat {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex));
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                }
                self.textures.insert(id, tex);
            }
        }
    }

    pub fn remove_mesh(&mut self, gl: &glow::Context, id: u64) {
        if let Some(m) = self.meshes.remove(&id) {
            unsafe {
                gl.delete_vertex_array(m.vao);
                gl.delete_buffer(m.vbo);
                gl.delete_buffer(m.ibo);
            }
        }
    }

    pub fn remove_tex(&mut self, gl: &glow::Context, id: u64) {
        if let Some(t) = self.textures.remove(&id) {
            unsafe { gl.delete_texture(t) };
        }
    }

    /// Frees GPU copies whose assets are gone on the host side.
    pub fn retain(&mut self, gl: &glow::Context, meshes: &HashSet<u64>, textures: &HashSet<u64>) {
        let dead: Vec<u64> = self.meshes.keys().filter(|k| !meshes.contains(k)).copied().collect();
        for k in dead {
            self.remove_mesh(gl, k);
        }
        let dead: Vec<u64> = self.textures.keys().filter(|k| !textures.contains(k)).copied().collect();
        for k in dead {
            self.remove_tex(gl, k);
        }
    }

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
            let mut cur_vao = None;
            let mut cur_tex: Option<Option<u64>> = None;
            let mut cur_state: Option<(Blend, bool)> = None;
            let mut overlay_cleared = false;
            for d in draws.iter() {
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
                    gl.bind_vertex_array(Some(m.vao));
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(m.ibo));
                    cur_vao = Some(d.mesh);
                }
                match &d.joints {
                    Some(j) => {
                        gl.uniform_1_i32(p.skin.as_ref(), 1);
                        let n = j.len().min(MAX_JOINTS);
                        let flat: Vec<f32> = j[..n].iter().flat_map(|m| m.to_cols_array()).collect();
                        gl.uniform_matrix_4_f32_slice(p.joints.as_ref(), false, &flat);
                    }
                    None => {
                        gl.uniform_1_i32(p.skin.as_ref(), 0);
                        gl.uniform_matrix_4_f32_slice(p.model.as_ref(), false, &d.model.to_cols_array());
                    }
                }
                gl.uniform_matrix_3_f32_slice(p.uvm.as_ref(), false, &d.uv);
                gl.uniform_1_i32(
                    p.mode.as_ref(),
                    match d.light {
                        Light::Dynamic => 0,
                        Light::Lightmap => 1,
                        Light::Vertex => 2,
                        Light::Unlit => 3,
                        Light::Unlit2x => 4,
                    },
                );
                gl.uniform_4_f32(p.color.as_ref(), d.color[0], d.color[1], d.color[2], d.color[3]);
                gl.uniform_1_f32(p.cutoff.as_ref(), d.cutoff);
                gl.uniform_1_f32(p.fog_mix.as_ref(), if matches!(d.blend, Blend::Add | Blend::Multiply) { 0.0 } else { 1.0 });
                gl.draw_elements(glow::TRIANGLES, m.count, glow::UNSIGNED_INT, 0);
                self.draws += 1;
                self.tris += m.count as u64 / 3;
            }
            gl.bind_vertex_array(None);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
        }
    }
}
