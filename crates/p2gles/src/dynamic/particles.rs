//! p2fx particle and decal batches (UE2 draw styles).

use super::Dynamic;
use super::Env;
use bevy_math::{Mat4, Vec3};
use glow::HasContext;

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
    pub fn render_particles(&mut self, gl: &glow::Context, vp: Mat4, cam: Vec3, env: &Env, draws: &mut [ParticleDraw]) {
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

