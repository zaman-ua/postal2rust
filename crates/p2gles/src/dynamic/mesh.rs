//! The vertex layout of `DYN_VS`: packing (no GL, any thread) and upload.

use glow::HasContext;
use std::cell::Cell;

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

/// A mesh in the GPU's vertex layout, ready to upload from whichever thread owns the context.
pub struct PackedMesh {
    pub(super) vertices: Vec<u8>,
    pub(super) indices: Vec<u8>,
    pub(super) count: i32,
}

/// Interleaves a mesh into the layout of `DYN_VS` (no GL calls); `None` if there is nothing to draw.
pub fn pack_mesh(m: &MeshIn) -> Option<PackedMesh> {
    let n = m.pos.len();
    if n == 0 || m.indices.is_empty() {
        return None;
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
    let indices = m.indices.iter().flat_map(|i| i.to_ne_bytes()).collect();
    Some(PackedMesh { vertices: buf, indices, count: m.indices.len() as i32 })
}

pub(super) struct GpuMesh {
    pub(super) vao: glow::VertexArray,
    pub(super) vbo: glow::Buffer,
    pub(super) ibo: glow::Buffer,
    pub(super) count: i32,
    /// Whether this VAO has the per-instance attributes enabled (vertex array state).
    pub(super) instanced: Cell<bool>,
}

/// A new vertex array in the layout of `DYN_VS`; leaves it bound.
pub(super) unsafe fn new_mesh(gl: &glow::Context, vertices: &[u8], indices: &[u8], count: i32) -> GpuMesh {
    unsafe {
        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        let vbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, vertices, glow::STATIC_DRAW);
        let stride = STRIDE as i32;
        for (loc, size, off) in [(0u32, 3, 0), (1, 3, 12), (2, 2, 24), (3, 2, 32), (4, 4, 40), (5, 4, 56)] {
            gl.enable_vertex_attrib_array(loc);
            gl.vertex_attrib_pointer_f32(loc, size, glow::FLOAT, false, stride, off);
        }
        gl.enable_vertex_attrib_array(6);
        gl.vertex_attrib_pointer_i32(6, 4, glow::UNSIGNED_SHORT, stride, 72);
        let ibo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ELEMENT_ARRAY_BUFFER, Some(ibo));
        gl.buffer_data_u8_slice(glow::ELEMENT_ARRAY_BUFFER, indices, glow::STATIC_DRAW);
        GpuMesh { vao, vbo, ibo, count, instanced: Cell::new(false) }
    }
}

