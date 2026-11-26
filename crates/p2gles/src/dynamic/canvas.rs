//! HUD and menus: p2canvas batches in painter's order.

use super::Dynamic;
use crate::program;
use bevy_math::Mat4;
use glow::HasContext;
use std::collections::HashMap;

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

pub(super) struct CanvasProg {
    pub(super) p: glow::Program,
    pub(super) loc: HashMap<&'static str, Option<glow::UniformLocation>>,
}

impl CanvasProg {
    pub(super) fn new(gl: &glow::Context, vs: &str, fs: &str, names: &[&'static str]) -> CanvasProg {
        unsafe {
            let p = program::link(gl, "canvas", vs, fs);
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

