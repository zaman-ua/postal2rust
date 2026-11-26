//! Generic mesh pass: meshes handed over per frame by a host (the game's ECS bridge). GPU copies are
//! cached by caller-chosen ids; the host uploads a mesh/texture once and refers to it afterwards.

use crate::renderer::{upload_bc, upload_tex};
use crate::shaders::{GRADE_FS, GRADE_VS, CANVAS_FS, CANVAS_VS, DYN_FS, DYN_VS, PART_FS, PART_VS};
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

pub struct Dynamic {
    progs: [Prog; 2],
    canvas: CanvasProg,
    part: CanvasProg,
    grade: Option<GradePass>,
    meshes: HashMap<u64, GpuMesh>,
    textures: HashMap<u64, glow::Texture>,
    white: glow::Texture,
    inst_vbo: glow::Buffer,
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
        let canvas = CanvasProg::new(gl, CANVAS_VS, CANVAS_FS, &["uVP", "uTint", "uSlotColor[0]", "uXu[0]", "uXv[0]", "uKinds", "uOps", "uParams", "uTex0", "uTex1", "uTex2"]);
        let part = CanvasProg::new(gl, PART_VS, PART_FS, &["uVP", "uCam", "uParams", "uFog", "uFogColor", "uTex"]);
        let inst_vbo = unsafe { gl.create_buffer().unwrap() };
        Dynamic { progs, canvas, part, grade: None, meshes: HashMap::new(), textures: HashMap::new(), white, inst_vbo, aniso, draws: 0, tris: 0 }
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
                    gl.bind_vertex_array(Some(m.vao));
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(m.ibo));
                    cur_vao = Some(d.mesh);
                }
                // Mirrored transforms (negative scale) flip the winding.
                let cw = d.joints.is_none() && d.model.determinant() < 0.0;
                if cw != cur_cw {
                    gl.front_face(if cw { glow::CW } else { glow::CCW });
                    cur_cw = cw;
                }
                let instanced = run > 1;
                gl.uniform_1_i32(p.inst.as_ref(), instanced as i32);
                if instanced {
                    inst_data.clear();
                    for o in &draws[i - run..i] {
                        inst_data.extend_from_slice(&o.model.to_cols_array());
                    }
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.inst_vbo));
                    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, std::slice::from_raw_parts(inst_data.as_ptr() as *const u8, inst_data.len() * 4), glow::STREAM_DRAW);
                    for k in 0..4u32 {
                        gl.enable_vertex_attrib_array(7 + k);
                        gl.vertex_attrib_pointer_f32(7 + k, 4, glow::FLOAT, false, 64, k as i32 * 16);
                        gl.vertex_attrib_divisor(7 + k, 1);
                    }
                } else {
                    for k in 0..4u32 {
                        gl.disable_vertex_attrib_array(7 + k);
                    }
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

/// How a canvas batch blends with the framebuffer (p2canvas `Blend`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CanvasBlend {
    Opaque,
    Alpha,
    /// src + dst * (1 - src)
    Translucent,
    /// src * dst * 2
    Modulate,
    /// premultiplied alpha
    AlphaModulate,
    Darken,
    Brighten,
    Invisible,
}

#[derive(Clone)]
pub struct CanvasDraw {
    pub mesh: u64,
    pub tex: [Option<u64>; 3],
    pub tint: [f32; 4],
    pub slot_color: [[f32; 4]; 3],
    /// Per slot: (m00, m01, t0, _) and (m10, m11, t1, _).
    pub xu: [[f32; 4]; 3],
    pub xv: [[f32; 4]; 3],
    pub kinds: [i32; 3],
    pub ops: [i32; 4],
    pub modulate: f32,
    pub alpha_test: f32,
    pub blend: CanvasBlend,
}

struct CanvasProg {
    p: glow::Program,
    loc: HashMap<&'static str, Option<glow::UniformLocation>>,
}

impl CanvasProg {
    fn new(gl: &glow::Context, vs: &str, fs: &str, names: &[&'static str]) -> CanvasProg {
        unsafe {
            let p = gl.create_program().unwrap();
            for (ty, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
                let s = gl.create_shader(ty).unwrap();
                gl.shader_source(s, src);
                gl.compile_shader(s);
                if !gl.get_shader_compile_status(s) {
                    panic!("canvas shader: {}", gl.get_shader_info_log(s));
                }
                gl.attach_shader(p, s);
            }
            gl.link_program(p);
            if !gl.get_program_link_status(p) {
                panic!("canvas link: {}", gl.get_program_info_log(p));
            }
            gl.use_program(Some(p));
            let mut loc = HashMap::new();
            for &n in names {
                loc.insert(n, gl.get_uniform_location(p, n));
            }
            for (i, n) in ["uTex0", "uTex1", "uTex2"].iter().enumerate() {
                if let Some(l) = loc.get(n) {
                    gl.uniform_1_i32(l.as_ref(), i as i32);
                }
            }
            if let Some(l) = loc.get("uTex") {
                gl.uniform_1_i32(l.as_ref(), 0);
            }
            CanvasProg { p, loc }
        }
    }
}

impl Dynamic {
    /// Draws the HUD batches (already in painter's order) over the frame: pixel coordinates,
    /// origin at the centre, y up, no depth.
    pub fn render_canvas(&mut self, gl: &glow::Context, size: (u32, u32), draws: &[CanvasDraw]) {
        if draws.is_empty() {
            return;
        }
        let (w, h) = (size.0 as f32, size.1 as f32);
        let vp = Mat4::orthographic_rh_gl(-w * 0.5, w * 0.5, -h * 0.5, h * 0.5, -1000.0, 1000.0);
        let c = &self.canvas;
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::CULL_FACE);
            gl.depth_mask(false);
            gl.use_program(Some(c.p));
            gl.uniform_matrix_4_f32_slice(c.loc["uVP"].as_ref(), false, &vp.to_cols_array());
            let mut cur_vao = None;
            for d in draws {
                let Some(m) = self.meshes.get(&d.mesh) else { continue };
                match d.blend {
                    CanvasBlend::Opaque => gl.disable(glow::BLEND),
                    b => {
                        gl.enable(glow::BLEND);
                        let (s, t) = match b {
                            CanvasBlend::Alpha => (glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA),
                            CanvasBlend::Translucent => (glow::ONE, glow::ONE_MINUS_SRC_COLOR),
                            CanvasBlend::Modulate => (glow::DST_COLOR, glow::SRC_COLOR),
                            CanvasBlend::AlphaModulate => (glow::ONE, glow::ONE_MINUS_SRC_ALPHA),
                            CanvasBlend::Darken => (glow::ZERO, glow::ONE_MINUS_SRC_COLOR),
                            CanvasBlend::Brighten => (glow::ONE, glow::ONE),
                            _ => (glow::ZERO, glow::ONE),
                        };
                        gl.blend_func(s, t);
                    }
                }
                for i in 0..3 {
                    gl.active_texture(glow::TEXTURE0 + i as u32);
                    gl.bind_texture(glow::TEXTURE_2D, Some(d.tex[i].and_then(|t| self.textures.get(&t).copied()).unwrap_or(self.white)));
                }
                gl.uniform_4_f32_slice(c.loc["uTint"].as_ref(), &d.tint);
                let flat = |a: &[[f32; 4]; 3]| -> Vec<f32> { a.iter().flatten().copied().collect() };
                gl.uniform_4_f32_slice(c.loc["uSlotColor[0]"].as_ref(), &flat(&d.slot_color));
                gl.uniform_4_f32_slice(c.loc["uXu[0]"].as_ref(), &flat(&d.xu));
                gl.uniform_4_f32_slice(c.loc["uXv[0]"].as_ref(), &flat(&d.xv));
                gl.uniform_3_i32_slice(c.loc["uKinds"].as_ref(), &d.kinds);
                gl.uniform_4_i32_slice(c.loc["uOps"].as_ref(), &d.ops);
                gl.uniform_4_f32(c.loc["uParams"].as_ref(), d.modulate, d.alpha_test, 0.0, 0.0);
                if cur_vao != Some(d.mesh) {
                    gl.bind_vertex_array(Some(m.vao));
                    cur_vao = Some(d.mesh);
                }
                gl.draw_elements(glow::TRIANGLES, m.count, glow::UNSIGNED_INT, 0);
                self.draws += 1;
                self.tris += m.count as u64 / 3;
            }
            gl.active_texture(glow::TEXTURE0);
            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
        }
    }
}

#[derive(Clone)]
pub struct ParticleDraw {
    pub mesh: u64,
    pub tex: Option<u64>,
    /// p2fx draw style byte: 0 regular, 1 alpha blend, 2 modulated, 3 translucent (additive),
    /// 4 alpha-modulate (premultiplied), 5 darken, 6 brighten.
    pub style: u8,
    pub z_test: bool,
    pub z_write: bool,
    pub alpha_ref: f32,
    pub alpha_test: bool,
    /// Draw order: ascending.
    pub order: f32,
}

impl Dynamic {
    /// Particle and decal batches (world-space meshes, `aW.x` = sub-image blend), after the rest of a
    /// camera's geometry. Depth-tested against it, never culled.
    pub fn render_particles(&mut self, gl: &glow::Context, vp: Mat4, cam: Vec3, env: &Env, draws: &mut Vec<ParticleDraw>) {
        if draws.is_empty() {
            return;
        }
        draws.sort_by(|a, b| a.order.partial_cmp(&b.order).unwrap_or(std::cmp::Ordering::Equal));
        let c = &self.part;
        unsafe {
            gl.use_program(Some(c.p));
            gl.uniform_matrix_4_f32_slice(c.loc["uVP"].as_ref(), false, &vp.to_cols_array());
            gl.uniform_3_f32(c.loc["uCam"].as_ref(), cam.x, cam.y, cam.z);
            gl.uniform_2_f32(c.loc["uFog"].as_ref(), env.fog_start, env.fog_end);
            gl.uniform_3_f32(c.loc["uFogColor"].as_ref(), env.fog_color[0], env.fog_color[1], env.fog_color[2]);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::DEPTH_TEST);
            gl.active_texture(glow::TEXTURE0);
            for d in draws.iter() {
                let Some(m) = self.meshes.get(&d.mesh) else { continue };
                match d.style {
                    0 => gl.disable(glow::BLEND),
                    s => {
                        gl.enable(glow::BLEND);
                        let (a, b) = match s {
                            1 => (glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA),
                            2 => (glow::DST_COLOR, glow::SRC_COLOR),
                            3 => (glow::ONE, glow::ONE),
                            4 => (glow::ONE, glow::ONE_MINUS_SRC_ALPHA),
                            5 => (glow::ZERO, glow::ONE_MINUS_SRC_COLOR),
                            _ => (glow::ONE, glow::ONE_MINUS_SRC_COLOR),
                        };
                        gl.blend_func(a, b);
                    }
                }
                gl.depth_func(if d.z_test { glow::LEQUAL } else { glow::ALWAYS });
                gl.depth_mask(d.z_write || d.style == 0);
                let tex = d.tex.and_then(|t| self.textures.get(&t).copied());
                gl.bind_texture(glow::TEXTURE_2D, Some(tex.unwrap_or(self.white)));
                gl.uniform_4_f32(c.loc["uParams"].as_ref(), d.alpha_ref, d.alpha_test as i32 as f32, d.style as f32, tex.is_some() as i32 as f32);
                gl.bind_vertex_array(Some(m.vao));
                gl.draw_elements(glow::TRIANGLES, m.count, glow::UNSIGNED_INT, 0);
                self.draws += 1;
                self.tris += m.count as u64 / 3;
            }
            gl.bind_vertex_array(None);
            gl.depth_func(glow::LEQUAL);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
        }
    }
}

/// Fullscreen colour grade state (created on first use).
pub struct GradePass {
    prog: CanvasProg,
    tex: glow::Texture,
    vao: glow::VertexArray,
    size: (u32, u32),
}

impl Dynamic {
    /// Brightness/contrast/gamma over the whole frame: copies the framebuffer to a texture and draws
    /// it back through the grade. `offset`/`slope`/`inv_gamma` as the Bevy `Grade` shader.
    pub fn render_grade(&mut self, gl: &glow::Context, size: (u32, u32), offset: f32, slope: f32, inv_gamma: f32) {
        unsafe {
            if self.grade.is_none() {
                let prog = CanvasProg::new(gl, GRADE_VS, GRADE_FS, &["uTex", "uGrade"]);
                let tex = gl.create_texture().unwrap();
                let vao = gl.create_vertex_array().unwrap();
                self.grade = Some(GradePass { prog, tex, vao, size: (0, 0) });
            }
            let g = self.grade.as_mut().unwrap();
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(g.tex));
            if g.size != size {
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGB8 as i32, size.0 as i32, size.1 as i32, 0, glow::RGB, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
                g.size = size;
            }
            gl.copy_tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, 0, 0, size.0 as i32, size.1 as i32);
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.depth_mask(false);
            gl.use_program(Some(g.prog.p));
            gl.uniform_3_f32(g.prog.loc["uGrade"].as_ref(), offset, slope, inv_gamma);
            gl.bind_vertex_array(Some(g.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
        }
    }
}
