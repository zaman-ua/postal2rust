//! Brightness, contrast and gamma over the finished frame.

use super::canvas::CanvasProg;
use super::Dynamic;
use crate::shaders::{GRADE_FS, GRADE_VS};
use glow::HasContext;

/// Fullscreen colour grade state (created on first use).
pub(super) struct GradePass {
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
