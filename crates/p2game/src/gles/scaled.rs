//! `P2GAME_GLES_SCALE=0.75`: the 3D scene is drawn at a fraction of the window resolution and
//! stretched onto it; the HUD stays at native resolution. For GPUs limited by pixel fill (Mali-G31):
//! 0.75 shades 56% of the pixels, 0.5 a quarter.

use glow::HasContext;

pub(super) struct ScaledTarget {
    scale: f32,
    /// (framebuffer, colour, depth, size), rebuilt when the window size changes.
    buffers: Option<(glow::Framebuffer, glow::Renderbuffer, glow::Renderbuffer, (u32, u32))>,
}

impl ScaledTarget {
    /// `None` at full resolution.
    pub fn from_env() -> Option<ScaledTarget> {
        let scale = std::env::var("P2GAME_GLES_SCALE").ok().and_then(|s| s.trim().parse::<f32>().ok())?.clamp(0.25, 1.0);
        (scale < 1.0).then_some(ScaledTarget { scale, buffers: None })
    }

    fn size_for(&self, (w, h): (u32, u32)) -> (u32, u32) {
        let scaled = |v: u32| ((v as f32 * self.scale).round() as u32).max(1);
        (scaled(w), scaled(h))
    }

    /// Binds the offscreen target and returns its size.
    pub fn begin(&mut self, gl: &glow::Context, window: (u32, u32)) -> (u32, u32) {
        let size = self.size_for(window);
        unsafe {
            if self.buffers.is_some_and(|b| b.3 != size) {
                let (fbo, color, depth, _) = self.buffers.take().unwrap();
                gl.delete_framebuffer(fbo);
                gl.delete_renderbuffer(color);
                gl.delete_renderbuffer(depth);
            }
            if self.buffers.is_none() {
                let fbo = gl.create_framebuffer().unwrap();
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
                let attach = |format, point| {
                    let rb = gl.create_renderbuffer().unwrap();
                    gl.bind_renderbuffer(glow::RENDERBUFFER, Some(rb));
                    gl.renderbuffer_storage(glow::RENDERBUFFER, format, size.0 as i32, size.1 as i32);
                    gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, point, glow::RENDERBUFFER, Some(rb));
                    rb
                };
                let color = attach(glow::RGB8, glow::COLOR_ATTACHMENT0);
                let depth = attach(glow::DEPTH_COMPONENT24, glow::DEPTH_ATTACHMENT);
                let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
                assert_eq!(status, glow::FRAMEBUFFER_COMPLETE, "P2GAME_GLES_SCALE: offscreen target incomplete");
                println!("gles: 3D at {}x{} (scale {}), window {}x{}", size.0, size.1, self.scale, window.0, window.1);
                self.buffers = Some((fbo, color, depth, size));
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, self.buffers.map(|b| b.0));
        }
        size
    }

    /// Stretches the finished 3D picture onto the window and leaves the window bound.
    pub fn finish(&self, gl: &glow::Context, window: (u32, u32)) {
        let Some((fbo, _, _, (w, h))) = self.buffers else { return };
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(fbo));
            // The depth buffer is not needed again: tile-based GPUs need not write it out.
            gl.invalidate_framebuffer(glow::READ_FRAMEBUFFER, &[glow::DEPTH_ATTACHMENT]);
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None);
            gl.blit_framebuffer(0, 0, w as i32, h as i32, 0, 0, window.0 as i32, window.1 as i32, glow::COLOR_BUFFER_BIT, glow::LINEAR);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, window.0 as i32, window.1 as i32);
        }
    }
}
