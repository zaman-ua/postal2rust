//! Scene baking (static geometry -> a few big buffers, spatial chunks) and the GLES 3.0 draw loop.

use crate::shaders::{FS, VS};
use bevy_math::{Mat4, Vec3, Vec4};
use glow::HasContext;
use p2view::scene::{Blend, MeshData, PixelFormat, SceneData};
use std::collections::HashMap;

const NONE: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy)]
struct V {
    pos: [f32; 3],
    uv: [f32; 2],
    uv2: [f32; 2],
    col: [u8; 4],
}
const VSIZE: i32 = std::mem::size_of::<V>() as i32;

/// Pass order: opaque, alpha-tested, terrain splat layers, then blended (sorted back to front).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Pass {
    Opaque = 0,
    Masked = 1,
    Splat = 2,
    Blended = 3,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Key {
    tex: u32,
    lm: u32,
    alpha_map: u32,
    blend: u8, // Blend as u8
    two_sided: bool,
    mode: u8,
    pass: Pass,
    order: u32,
}

fn blend_id(b: Blend) -> u8 {
    match b {
        Blend::Opaque => 0,
        Blend::Masked => 1,
        Blend::Translucent => 2,
        Blend::Modulate => 3,
        Blend::Additive => 4,
        Blend::Darken => 5,
    }
}

#[derive(Default)]
struct Chunk {
    verts: Vec<V>,
    idx: Vec<u32>,
    lo: Vec3,
    hi: Vec3,
}

/// `glDrawRangeElements` (glow 0.17 has no wrapper), loaded by main.rs from the EGL display.
pub type DrawRangeElementsFn = unsafe extern "system" fn(u32, u32, u32, i32, u32, *const std::ffi::c_void);
pub static DRAW_RANGE_ELEMENTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

struct Draw {
    key: Key,
    first: i32, // first index
    count: i32,
    /// Vertex index range of this draw: passed to glDrawRangeElements so the driver does not scan
    /// the index buffer on the CPU (Panfrost spent ~88% of the frame doing that).
    vmin: u32,
    vmax: u32,
    lo: Vec3,
    hi: Vec3,
    centre: Vec3,
}

pub struct Options {
    pub chunk: f32,
    pub fog: Option<(f32, f32)>,
    pub no_cull: bool,
    /// Actors (map export indices) left out of the bake: movers, drawn by the dynamic pass.
    pub skip_exports: std::collections::HashSet<usize>,
}

struct Baker {
    chunks: HashMap<(Key, i32, i32), Chunk>,
    cell: f32,
}

impl Baker {
    fn chunk(&mut self, key: Key, x: f32, z: f32) -> (&mut Chunk, (i32, i32)) {
        let c = ((x / self.cell).floor() as i32, (z / self.cell).floor() as i32);
        let ch = self.chunks.entry((key, c.0, c.1)).or_insert_with(|| Chunk { lo: Vec3::splat(f32::MAX), hi: Vec3::splat(f32::MIN), ..Default::default() });
        (ch, c)
    }
}

fn push_vert(ch: &mut Chunk, v: V) -> u32 {
    let p = Vec3::from(v.pos);
    ch.lo = ch.lo.min(p);
    ch.hi = ch.hi.max(p);
    ch.verts.push(v);
    (ch.verts.len() - 1) as u32
}

fn pack(c: f32) -> u8 {
    (c * 255.0).round().clamp(0.0, 255.0) as u8
}

pub struct Renderer {
    gl: glow::Context,
    /// [0] no discard (opaque, splat, blended), [1] alpha-tested (masked pass).
    progs: [glow::Program; 2],
    vao: glow::VertexArray,
    draws: Vec<Draw>,
    /// One index buffer per draw record (`P2GLES_ONE_IBO=1`: one shared buffer). Panfrost caches
    /// index min/max per buffer in a small cache; with one shared buffer for ~240 draws the cache
    /// thrashed and the driver rescanned the index ranges on the CPU every frame (~88% of a frame).
    ibos: Vec<Option<glow::Buffer>>,
    order: [Vec<usize>; 4],
    textures: HashMap<u32, glow::Texture>,
    lightmaps: HashMap<u32, glow::Texture>,
    alphas: HashMap<u32, glow::Texture>,
    white: glow::Texture,
    pub fog: (f32, f32),
    pub fog_color: [f32; 3],
    no_cull: bool,
    pub stats: Stats,
    us: [Uniforms; 2],
    pub total_tris: u64,
    /// false: the host clears (multi-camera frames); `render` only draws.
    pub clear: bool,
    /// `P2GLES_SKIP` and `P2GLES_ONE_PROG`.
    skip_passes: [bool; 4],
    one_program: bool,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub draws: u32,
    pub tris: u64,
}

struct Uniforms {
    vp: glow::UniformLocation,
    cam: glow::UniformLocation,
    mode: glow::UniformLocation,
    cst: glow::UniformLocation,
    atest: glow::UniformLocation,
    fopaque: glow::UniformLocation,
    use_alpha: glow::UniformLocation,
    fog: glow::UniformLocation,
    fog_color: glow::UniformLocation,
    has_atest: bool,
}

pub static MIPS_GENERATED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// DXT1/3/5 mips uploaded as-is (GL_EXT_texture_compression_s3tc): 4-8x less memory and
/// texture bandwidth than the RGBA8 decode. The chain stops where the package's does.
pub(crate) unsafe fn upload_bc(gl: &glow::Context, w: u32, h: u32, fmt: PixelFormat, mips: &[Vec<u8>], aniso: bool) -> Option<glow::Texture> {
    let (gl_fmt, block) = match fmt {
        PixelFormat::Bc1 => (0x83F1u32, 8u32), // COMPRESSED_RGBA_S3TC_DXT1_EXT (keeps 1-bit alpha)
        PixelFormat::Bc2 => (0x83F2, 16),      // DXT3
        PixelFormat::Bc3 => (0x83F3, 16),      // DXT5
        PixelFormat::Rgba8 => return None,
    };
    unsafe {
        let t = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(t));
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        let mut n = 0;
        for (i, m) in mips.iter().enumerate() {
            let (lw, lh) = ((w >> i).max(1), (h >> i).max(1));
            let need = (lw.div_ceil(4) * lh.div_ceil(4) * block) as usize;
            if m.len() < need {
                break;
            }
            gl.compressed_tex_image_2d(glow::TEXTURE_2D, i as i32, gl_fmt as i32, lw as i32, lh as i32, 0, need as i32, &m[..need]);
            n += 1;
        }
        if n == 0 || gl.get_error() != glow::NO_ERROR {
            gl.delete_texture(t);
            return None;
        }
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, n - 1);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, if n > 1 { glow::LINEAR_MIPMAP_LINEAR } else { glow::LINEAR } as i32);
        if n > 1 {
            if let Some(a) = std::env::var("P2GLES_ANISO").ok().and_then(|v| v.parse::<f32>().ok()).filter(|_| aniso) {
                gl.tex_parameter_f32(glow::TEXTURE_2D, 0x84FE, a);
            }
        }
        BC_UPLOADED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(t)
    }
}

pub static BC_UPLOADED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub(crate) unsafe fn upload_tex(gl: &glow::Context, w: u32, h: u32, mips: &[&[u8]], repeat: bool, aniso: bool) -> glow::Texture {
    unsafe {
        let t = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(t));
        let wrap = if repeat { glow::REPEAT } else { glow::CLAMP_TO_EDGE } as i32;
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, wrap);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, wrap);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        let mut n = 0;
        for (i, m) in mips.iter().enumerate() {
            let (lw, lh) = ((w >> i).max(1), (h >> i).max(1));
            if m.len() != (lw * lh * 4) as usize {
                break;
            }
            gl.tex_image_2d(glow::TEXTURE_2D, i as i32, glow::RGBA8 as i32, lw as i32, lh as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(m)));
            n += 1;
        }
        if n == 0 {
            return t;
        }
        // Full chain down to 1x1? Packages often ship a partial chain (or none): sampling a
        // large texture minified without mips thrashes the texture cache (Mali-G31: the frame
        // cost barely dropped at quarter resolution). Generate the missing levels on the GPU.
        let full = 32 - w.max(h).leading_zeros() as i32;
        let gen_mips = n < full && std::env::var_os("P2GLES_NO_GENMIPS").is_none();
        if gen_mips {
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_BASE_LEVEL, 0);
            gl.generate_mipmap(glow::TEXTURE_2D);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
            MIPS_GENERATED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else if n > 1 {
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAX_LEVEL, n - 1);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
        } else {
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        }
        // Anisotropic filtering costs up to 8 texture fetches per pixel: opt-in (P2GLES_ANISO=N).
        if n > 1 || gen_mips {
            if let Some(a) = std::env::var("P2GLES_ANISO").ok().and_then(|v| v.parse::<f32>().ok()).filter(|_| aniso) {
                gl.tex_parameter_f32(glow::TEXTURE_2D, 0x84FE, a); // TEXTURE_MAX_ANISOTROPY_EXT
            }
        }
        t
    }
}

impl Renderer {
    pub fn new(gl: glow::Context, data: &SceneData, opts: &Options) -> Renderer {
        let t0 = std::time::Instant::now();
        let ext_aniso = gl.supported_extensions().contains("GL_EXT_texture_filter_anisotropic");
        unsafe {
            // Programs: the same shader with and without the alpha-test discard.
            let build = |masked: bool| -> (glow::Program, Uniforms) {
                let prog = gl.create_program().unwrap();
                let fs = if masked { FS.replacen("#version 300 es\n", "#version 300 es\n#define ALPHA_TEST 1\n", 1) } else { FS.to_string() };
                for (ty, src) in [(glow::VERTEX_SHADER, VS.to_string()), (glow::FRAGMENT_SHADER, fs)] {
                    let s = gl.create_shader(ty).unwrap();
                    gl.shader_source(s, &src);
                    gl.compile_shader(s);
                    if !gl.get_shader_compile_status(s) {
                        panic!("shader: {}", gl.get_shader_info_log(s));
                    }
                    gl.attach_shader(prog, s);
                }
                gl.link_program(prog);
                if !gl.get_program_link_status(prog) {
                    panic!("link: {}", gl.get_program_info_log(prog));
                }
                gl.use_program(Some(prog));
                let loc = |n: &str| gl.get_uniform_location(prog, n);
                let req = |n: &str| loc(n).unwrap_or_else(|| panic!("uniform {n}"));
                // uAlphaTest is optimised out of the program without discard: use a dummy location then.
                let atest = loc("uAlphaTest").unwrap_or_else(|| req("uForceOpaque"));
                let u = Uniforms { vp: req("uVP"), cam: req("uCam"), mode: req("uMode"), cst: req("uConst"), atest, fopaque: req("uForceOpaque"), use_alpha: req("uUseAlphaMap"), fog: req("uFog"), fog_color: req("uFogColor"), has_atest: masked };
                gl.uniform_1_i32(Some(&req("uTex")), 0);
                if let Some(l) = loc("uLm") { gl.uniform_1_i32(Some(&l), 1); }
                if let Some(l) = loc("uAlphaMap") { gl.uniform_1_i32(Some(&l), 2); }
                (prog, u)
            };
            let (p0, u0) = build(false);
            let (p1, u1) = build(true);
            let progs = [p0, p1];
            let us = [u0, u1];

            // ------------- bake -------------
            let mut baker = Baker { chunks: HashMap::new(), cell: opts.chunk };
            let ambient = data.zone.as_ref().map(|z| [z.ambient[0] * z.ambient_brightness, z.ambient[1] * z.ambient_brightness, z.ambient[2] * z.ambient_brightness]).unwrap_or([0.0; 3]);
            let usable: Vec<bool> = data
                .lightmaps
                .iter()
                .map(|l| {
                    let ok = l.width <= 16384 && l.height <= 16384 && l.rgba.len() as u64 == l.width as u64 * l.height as u64 * 4;
                    let n = (l.rgba.len() / 4).max(1) as u64;
                    let sum: u64 = l.rgba.chunks_exact(4).map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64).sum();
                    ok && sum / (3 * n) >= 8
                })
                .collect();
            let mat_key = |mat: u32, mode: u8, lm: u32| -> Key {
                let m = &data.materials[mat as usize];
                let pass = match m.blend {
                    Blend::Opaque => Pass::Opaque,
                    Blend::Masked => Pass::Masked,
                    _ => Pass::Blended,
                };
                Key { tex: m.texture.unwrap_or(NONE), lm, alpha_map: NONE, blend: blend_id(m.blend), two_sided: m.two_sided, mode, pass, order: 0 }
            };
            // The mode for unlit blend types.
            let fix_mode = |k: &mut Key| match k.blend {
                3 => k.mode = 4,
                4 | 5 => k.mode = 3,
                _ => {}
            };

            // World BSP: split triangles into cells.
            for m in &data.world {
                let lm = m.lightmap.filter(|&l| usable.get(l as usize).copied().unwrap_or(false)).filter(|_| !m.uvs_b.is_empty());
                let mut key = mat_key(m.material, if lm.is_some() { 1 } else { 0 }, lm.unwrap_or(NONE));
                fix_mode(&mut key);
                bake_split(&mut baker, key, m, |i| V { pos: m.positions[i], uv: m.uvs[i], uv2: if m.uvs_b.is_empty() { [0.0; 2] } else { m.uvs_b[i] }, col: [0, 0, 0, 255] });
            }
            // Static meshes: bake the transform, one chunk per instance origin cell.
            for inst in data.instances.iter().filter(|i| !opts.skip_exports.contains(&i.export)) {
                let src = &data.meshes[inst.mesh as usize];
                let mat = &data.materials[inst.material as usize];
                if src.positions.is_empty() {
                    continue;
                }
                let colors: Option<Vec<[u8; 4]>> = inst.light.as_ref().filter(|c| !src.source.is_empty() && src.source.iter().all(|&v| (v as usize) < c.len())).map(|c| {
                    let za = inst.zone_ambient.map(|a| a.map(|v| v * 0.65)).unwrap_or(ambient);
                    let g = inst.glow * 0.5;
                    src.source
                        .iter()
                        .map(|&v| {
                            let c = c[v as usize];
                            let f = |k: usize| ((c[k] as f32 + za[k] + g).min(255.0)) as u8;
                            [f(0), f(1), f(2), 255]
                        })
                        .collect()
                });
                let mode = if inst.unlit { 3 } else if colors.is_some() { 2 } else { 0 };
                let mut key = mat_key(inst.material, mode, NONE);
                fix_mode(&mut key);
                let _ = mat;
                let o = inst.transform.w_axis;
                let (ch, _) = baker.chunk(key, o.x, o.z);
                let base = ch.verts.len() as u32;
                for i in 0..src.positions.len() {
                    let p = inst.transform.transform_point3(Vec3::from(src.positions[i]));
                    let col = colors.as_ref().map(|c| c[i]).unwrap_or([0, 0, 0, 255]);
                    push_vert(ch, V { pos: p.to_array(), uv: src.uvs.get(i).copied().unwrap_or([0.0; 2]), uv2: [0.0; 2], col });
                }
                ch.idx.extend(src.indices.iter().map(|&i| i + base));
            }
            // Terrain layers.
            for (li, l) in data.terrain.iter().enumerate() {
                let m = &l.mesh;
                let key = Key { tex: l.texture.unwrap_or(NONE), lm: NONE, alpha_map: if l.order > 0 && l.alpha.is_some() { li as u32 } else { NONE }, blend: 0, two_sided: false, mode: if m.colors.is_empty() { 0 } else { 2 }, pass: if l.order == 0 { Pass::Opaque } else { Pass::Splat }, order: l.order };
                // Referenced triangles only (the bounding corners are appended unreferenced).
                bake_split(&mut baker, key, m, |i| {
                    let c = m.colors.get(i).copied().unwrap_or([0.5, 0.5, 0.5, 1.0]);
                    // linear light factor -> gamma, stored halved (shader doubles it)
                    let g = |v: f32| pack(v.max(0.0).powf(1.0 / 2.2) * 0.5);
                    V { pos: m.positions[i], uv: m.uvs[i], uv2: m.uvs_b.get(i).copied().unwrap_or([0.0; 2]), col: [g(c[0]), g(c[1]), g(c[2]), 255] }
                });
            }

            // ------------- finalise buffers -------------
            let mut keys: Vec<_> = baker.chunks.into_iter().collect();
            keys.sort_by_key(|((k, cx, cz), _)| (k.pass as u8, k.order, k.tex, k.lm, k.alpha_map, k.blend, k.two_sided, k.mode, *cx, *cz));
            let mut verts: Vec<V> = Vec::new();
            let mut idx: Vec<u32> = Vec::new();
            let mut draws = Vec::new();
            let mut total_tris = 0u64;
            for ((key, _, _), ch) in keys {
                if ch.idx.is_empty() {
                    continue;
                }
                let base = verts.len() as u32;
                let first = idx.len() as i32;
                verts.extend_from_slice(&ch.verts);
                idx.extend(ch.idx.iter().map(|i| i + base));
                total_tris += ch.idx.len() as u64 / 3;
                draws.push(Draw { key, first, count: ch.idx.len() as i32, vmin: base, vmax: base + ch.verts.len() as u32 - 1, lo: ch.lo, hi: ch.hi, centre: (ch.lo + ch.hi) * 0.5 });
            }
            let mut order: [Vec<usize>; 4] = Default::default();
            for (i, d) in draws.iter().enumerate() {
                order[d.key.pass as usize].push(i);
            }
            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));
            let vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, std::slice::from_raw_parts(verts.as_ptr() as *const u8, verts.len() * VSIZE as usize), glow::STATIC_DRAW);
            let ibo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, std::slice::from_raw_parts(idx.as_ptr() as *const u8, idx.len() * 4), glow::STATIC_DRAW);
            let mut ibos: Vec<Option<glow::Buffer>> = vec![None; draws.len()];
            if std::env::var_os("P2GLES_ONE_IBO").is_none() {
                for (i, d) in draws.iter().enumerate() {
                    let b = gl.create_buffer().unwrap();
                    gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(b));
                    let sl = &idx[d.first as usize..(d.first + d.count) as usize];
                    gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, std::slice::from_raw_parts(sl.as_ptr() as *const u8, sl.len() * 4), glow::STATIC_DRAW);
                    ibos[i] = Some(b);
                }
                gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
            }
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, VSIZE, 0);
            gl.enable_vertex_attrib_array(1);
            gl.vertex_attrib_pointer_f32(1, 2, glow::FLOAT, false, VSIZE, 12);
            gl.enable_vertex_attrib_array(2);
            gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, VSIZE, 20);
            gl.enable_vertex_attrib_array(3);
            gl.vertex_attrib_pointer_f32(3, 4, glow::UNSIGNED_BYTE, true, VSIZE, 28);
            println!("bake: {} verts ({} MB), {} tris, {} draw records ({} opaque, {} masked, {} splat, {} blended) in {} ms", verts.len(), (verts.len() * VSIZE as usize + idx.len() * 4) >> 20, total_tris, draws.len(), order[0].len(), order[1].len(), order[2].len(), order[3].len(), t0.elapsed().as_millis());

            // ------------- textures -------------
            let t1 = std::time::Instant::now();
            let mut textures = HashMap::new();
            let mut lightmaps = HashMap::new();
            let mut alphas = HashMap::new();
            for d in &draws {
                let k = d.key;
                if k.tex != NONE && !textures.contains_key(&k.tex) {
                    let t = &data.textures[k.tex as usize];
                    if t.format == PixelFormat::Rgba8 {
                        let mips: Vec<&[u8]> = t.mips.iter().map(|m| m.as_slice()).collect();
                        textures.insert(k.tex, upload_tex(&gl, t.width, t.height, &mips, true, ext_aniso));
                    } else if let Some(tx) = upload_bc(&gl, t.width, t.height, t.format, &t.mips, ext_aniso) {
                        textures.insert(k.tex, tx);
                    }
                }
                if k.lm != NONE && !lightmaps.contains_key(&k.lm) {
                    let l = &data.lightmaps[k.lm as usize];
                    lightmaps.insert(k.lm, upload_tex(&gl, l.width, l.height, &[l.rgba.as_slice()], false, false));
                }
                if k.alpha_map != NONE && !alphas.contains_key(&k.alpha_map) {
                    let (w, h, bytes) = data.terrain[k.alpha_map as usize].alpha.as_ref().unwrap();
                    if bytes.len() as u32 >= w * h {
                        let t = gl.create_texture().unwrap();
                        gl.bind_texture(glow::TEXTURE_2D, Some(t));
                        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
                        gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R8 as i32, *w as i32, *h as i32, 0, glow::RED, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&bytes[..(*w * *h) as usize])));
                        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
                        for p in [glow::TEXTURE_MIN_FILTER, glow::TEXTURE_MAG_FILTER] {
                            gl.tex_parameter_i32(glow::TEXTURE_2D, p, glow::LINEAR as i32);
                        }
                        for p in [glow::TEXTURE_WRAP_S, glow::TEXTURE_WRAP_T] {
                            gl.tex_parameter_i32(glow::TEXTURE_2D, p, glow::CLAMP_TO_EDGE as i32);
                        }
                        alphas.insert(k.alpha_map, t);
                    }
                }
            }
            let white = upload_tex(&gl, 1, 1, &[&[153, 153, 153, 255]], true, false);
            println!("textures: {} diffuse ({} DXT as-is, {} with generated mips), {} lightmaps, {} alpha maps uploaded in {} ms", textures.len(), BC_UPLOADED.load(std::sync::atomic::Ordering::Relaxed), MIPS_GENERATED.load(std::sync::atomic::Ordering::Relaxed), lightmaps.len(), alphas.len(), t1.elapsed().as_millis());

            let fog = opts.fog.or(data.zone.as_ref().and_then(|z| z.fog).map(|(_, s, e)| (s, e))).unwrap_or((250.0, 1400.0));
            let fog_color = data.zone.as_ref().and_then(|z| z.fog).map(|(c, _, _)| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]).unwrap_or([0.55, 0.68, 0.85]);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            gl.front_face(glow::CCW);
            // Diagnostics, read once: P2GLES_SKIP=opaque,masked,splat,blend leaves those passes out.
            let skip = std::env::var("P2GLES_SKIP").unwrap_or_default();
            let skip_passes = ["opaque", "masked", "splat", "blend"].map(|pass| skip.contains(pass));
            let one_program = std::env::var_os("P2GLES_ONE_PROG").is_some();
            Renderer { gl, progs, vao, draws, ibos, order, textures, lightmaps, alphas, white, fog, fog_color, no_cull: opts.no_cull, stats: Stats::default(), us, total_tris, clear: true, skip_passes, one_program }
        }
    }

    pub fn render(&mut self, vp: Mat4, cam: Vec3, size: (u32, u32)) {
        let gl = &self.gl;
        // Frustum planes (Gribb/Hartmann) in world space.
        let r = |i: usize| Vec4::new(vp.x_axis[i], vp.y_axis[i], vp.z_axis[i], vp.w_axis[i]);
        let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
        let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2];
        let fog_end = self.fog.1;
        let visible = |d: &Draw, no_cull: bool| -> bool {
            if no_cull {
                return true;
            }
            let near = cam.clamp(d.lo, d.hi);
            if (near - cam).length_squared() > fog_end * fog_end {
                return false;
            }
            for p in &planes {
                let v = Vec3::new(if p.x >= 0.0 { d.hi.x } else { d.lo.x }, if p.y >= 0.0 { d.hi.y } else { d.lo.y }, if p.z >= 0.0 { d.hi.z } else { d.lo.z });
                if p.x * v.x + p.y * v.y + p.z * v.z + p.w < 0.0 {
                    return false;
                }
            }
            true
        };
        let mut stats = Stats::default();
        unsafe {
            gl.viewport(0, 0, size.0 as i32, size.1 as i32);
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.enable(glow::DEPTH_TEST);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            if self.clear {
                gl.clear_color(self.fog_color[0], self.fog_color[1], self.fog_color[2], 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            }
            gl.bind_vertex_array(Some(self.vao));
            for (pg, u) in self.progs.iter().zip(&self.us) {
                gl.use_program(Some(*pg));
                gl.uniform_matrix_4_f32_slice(Some(&u.vp), false, &vp.to_cols_array());
                gl.uniform_3_f32(Some(&u.cam), cam.x, cam.y, cam.z);
                gl.uniform_2_f32(Some(&u.fog), self.fog.0, self.fog.1);
                gl.uniform_3_f32(Some(&u.cst), 0.75, 0.75, 0.75);
            }

            let mut cur_tex = u32::MAX - 1;
            let mut cur_lm = u32::MAX - 1;
            let mut cur_alpha = u32::MAX - 1;
            let no_cull = self.no_cull;
            // Blended draws, back to front.
            let mut blended: Vec<usize> = self.order[3].iter().copied().filter(|&i| visible(&self.draws[i], no_cull)).collect();
            blended.sort_by(|&a, &b| {
                let da = (self.draws[a].centre - cam).length_squared();
                let db = (self.draws[b].centre - cam).length_squared();
                db.total_cmp(&da)
            });
            let lists: [Vec<usize>; 4] = [
                self.order[0].iter().copied().filter(|&i| visible(&self.draws[i], no_cull)).collect(),
                self.order[1].iter().copied().filter(|&i| visible(&self.draws[i], no_cull)).collect(),
                self.order[2].iter().copied().filter(|&i| visible(&self.draws[i], no_cull)).collect(),
                blended,
            ];
            for (pi, list) in lists.iter().enumerate() {
                if self.skip_passes[pi] {
                    continue;
                }
                let ui = (pi == 1 && !self.one_program) as usize;
                let u = &self.us[ui];
                gl.use_program(Some(self.progs[ui]));
                let mut cur_state: Option<Key> = None;
                for &di in list {
                    let d = &self.draws[di];
                    let k = d.key;
                    // Render state.
                    if cur_state.map(|s| (s.pass, s.blend, s.two_sided)) != Some((k.pass, k.blend, k.two_sided)) {
                        if k.two_sided {
                            gl.disable(glow::CULL_FACE);
                        } else {
                            gl.enable(glow::CULL_FACE);
                        }
                        match (k.pass, k.blend) {
                            (Pass::Opaque, _) | (Pass::Masked, _) => {
                                gl.disable(glow::BLEND);
                                gl.depth_mask(true);
                            }
                            (Pass::Splat, _) | (_, 2) => {
                                gl.enable(glow::BLEND);
                                gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                                gl.depth_mask(false);
                            }
                            (_, 3) | (_, 5) => {
                                gl.enable(glow::BLEND);
                                gl.blend_func(glow::DST_COLOR, glow::ZERO);
                                gl.depth_mask(false);
                            }
                            _ => {
                                gl.enable(glow::BLEND);
                                gl.blend_func(glow::SRC_ALPHA, glow::ONE);
                                gl.depth_mask(false);
                            }
                        }
                        if u.has_atest { gl.uniform_1_i32(Some(&u.atest), (k.pass == Pass::Masked) as i32); }
                        gl.uniform_1_i32(Some(&u.fopaque), (k.pass == Pass::Opaque && k.blend == 0) as i32);
                        let fc = match k.blend {
                            3 | 5 => [1.0; 3],
                            4 => [0.0; 3],
                            _ => self.fog_color,
                        };
                        gl.uniform_3_f32(Some(&u.fog_color), fc[0], fc[1], fc[2]);
                    }
                    if cur_state.map(|s| s.mode) != Some(k.mode) {
                        gl.uniform_1_i32(Some(&u.mode), k.mode as i32);
                    }
                    if cur_state.map(|s| s.alpha_map != NONE) != Some(k.alpha_map != NONE) {
                        gl.uniform_1_i32(Some(&u.use_alpha), (k.alpha_map != NONE) as i32);
                    }
                    cur_state = Some(k);
                    if k.tex != cur_tex {
                        gl.active_texture(glow::TEXTURE0);
                        gl.bind_texture(glow::TEXTURE_2D, Some(*self.textures.get(&k.tex).unwrap_or(&self.white)));
                        cur_tex = k.tex;
                    }
                    if k.lm != cur_lm && k.lm != NONE {
                        gl.active_texture(glow::TEXTURE1);
                        gl.bind_texture(glow::TEXTURE_2D, self.lightmaps.get(&k.lm).copied());
                        gl.active_texture(glow::TEXTURE0);
                        cur_lm = k.lm;
                    }
                    if k.alpha_map != cur_alpha && k.alpha_map != NONE {
                        gl.active_texture(glow::TEXTURE2);
                        gl.bind_texture(glow::TEXTURE_2D, self.alphas.get(&k.alpha_map).copied());
                        gl.active_texture(glow::TEXTURE0);
                        cur_alpha = k.alpha_map;
                    }
                    if let Some(b) = self.ibos[di] {
                        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(b));
                        gl.draw_elements(glow::TRIANGLES, d.count, glow::UNSIGNED_INT, 0);
                    } else {
                    match DRAW_RANGE_ELEMENTS.get() {
                        Some(&f) if std::env::var_os("P2GLES_NO_RANGE").is_none() => {
                            let f: DrawRangeElementsFn = std::mem::transmute(f);
                            f(glow::TRIANGLES, d.vmin, d.vmax, d.count, glow::UNSIGNED_INT, (d.first as usize * 4) as *const std::ffi::c_void);
                        }
                        _ => gl.draw_elements(glow::TRIANGLES, d.count, glow::UNSIGNED_INT, d.first * 4),
                    }
                    }
                    stats.draws += 1;
                    stats.tris += d.count as u64 / 3;
                }
            }
        }
        self.stats = stats;
    }

    pub fn read_pixels(&self, w: u32, h: u32) -> Vec<u8> {
        let mut buf = vec![0u8; (w * h * 4) as usize];
        unsafe {
            self.gl.finish();
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            self.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut buf)));
        }
        buf
    }

    pub fn finish(&self) {
        unsafe { self.gl.finish() }
    }
}

/// Splits `m`'s triangles into cells by centroid (x, z), remapping vertices per cell.
fn bake_split(b: &mut Baker, key: Key, m: &MeshData, vert: impl Fn(usize) -> V) {
    let mut remap: HashMap<(i32, i32), HashMap<u32, u32>> = HashMap::new();
    for tri in m.indices.chunks_exact(3) {
        let p = |i: u32| Vec3::from(m.positions[i as usize]);
        let c = (p(tri[0]) + p(tri[1]) + p(tri[2])) / 3.0;
        let (ch, cell) = b.chunk(key, c.x, c.z);
        let rm = remap.entry(cell).or_default();
        for &v in tri {
            let n = *rm.entry(v).or_insert_with(|| push_vert(ch, vert(v as usize)));
            ch.idx.push(n);
        }
    }
}
