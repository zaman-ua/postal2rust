//! Everything p2gles draws besides the baked world, handed over per frame by a host (the game's ECS
//! bridge). GPU copies are cached by caller-chosen ids; the host uploads a mesh or texture once and
//! refers to it afterwards.
//!
//! - [`mesh`]: vertex layout, packing and upload;
//! - `pass`: the 3D mesh pass ([`Dynamic::render`]);
//! - `canvas`, `particles`, `grade`: HUD, p2fx batches, brightness/contrast/gamma.

mod canvas;
mod grade;
mod mesh;
mod particles;
mod pass;

pub use canvas::{CanvasBlend, CanvasDraw};
pub use mesh::{pack_mesh, MeshIn, PackedMesh, MAX_JOINTS};
pub use particles::ParticleDraw;
pub use pass::{Blend, Draw, Env, Light, UV_IDENTITY};

use crate::renderer::{upload_bc, upload_tex};
use crate::shaders::{CANVAS_FS, CANVAS_VS, PART_FS, PART_VS};
use canvas::CanvasProg;
use glow::HasContext;
use mesh::{new_mesh, GpuMesh};
use p2view::scene::PixelFormat;
use std::collections::HashSet;

/// Maps keyed by the host's ids are looked up several times per draw: no SipHash.
type IdMap<V> = std::collections::HashMap<u64, V, foldhash::fast::FixedState>;


pub struct TexIn<'a> {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// Mip levels, largest first (one is fine: the rest is generated).
    pub mips: Vec<&'a [u8]>,
    pub repeat: bool,
}

pub struct Dynamic {
    progs: [pass::Prog; 2],
    canvas: CanvasProg,
    part: CanvasProg,
    grade: Option<grade::GradePass>,
    meshes: IdMap<GpuMesh>,
    textures: IdMap<glow::Texture>,
    white: glow::Texture,
    inst_vbo: glow::Buffer,
    aniso: bool,
    pub draws: u32,
    pub tris: u64,
}

impl Dynamic {
    pub fn new(gl: &glow::Context) -> Dynamic {
        let aniso = gl.supported_extensions().contains("GL_EXT_texture_filter_anisotropic");
        let progs = [pass::compile(gl, false), pass::compile(gl, true)];
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
        Dynamic { progs, canvas, part, grade: None, meshes: Default::default(), textures: Default::default(), white, inst_vbo, aniso, draws: 0, tris: 0 }
    }

    pub fn has_mesh(&self, id: u64) -> bool {
        self.meshes.contains_key(&id)
    }

    pub fn has_tex(&self, id: u64) -> bool {
        self.textures.contains_key(&id)
    }

    pub fn add_mesh(&mut self, gl: &glow::Context, id: u64, m: &MeshIn) {
        match pack_mesh(m) {
            Some(packed) => self.add_packed(gl, id, &packed),
            None => self.remove_mesh(gl, id),
        }
    }

    /// Uploads a mesh, or replaces the contents of the one already known by `id`. HUD and particle
    /// meshes change every frame, so their GL objects are kept and only the data is sent again.
    pub fn add_packed(&mut self, gl: &glow::Context, id: u64, packed: &PackedMesh) {
        unsafe {
            if let Some(mesh) = self.meshes.get_mut(&id) {
                // Changed once: expect it to keep changing.
                gl.bind_vertex_array(Some(mesh.vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(mesh.vbo));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &packed.vertices, glow::DYNAMIC_DRAW);
                gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, &packed.indices, glow::DYNAMIC_DRAW);
                mesh.count = packed.count;
            } else {
                let mesh = new_mesh(gl, &packed.vertices, &packed.indices, packed.count);
                self.meshes.insert(id, mesh);
            }
            gl.bind_vertex_array(None);
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
}
