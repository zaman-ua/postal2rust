//! The per-frame system: collect from the ECS, draw, present.

use super::collect::{Cam, Cameras, Hud, Lighting, Pass, SceneMeshes};
use super::Gles;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::PrimaryWindow;
use glow::HasContext;
use glutin::prelude::*;
use p2gles::dynamic::CanvasDraw;
use std::num::NonZeroU32;
use std::time::{Duration, Instant};

#[derive(Default, Clone, Copy)]
struct Drawn {
    draws: u32,
    tris: u64,
}

/// Timing report, printed every 5 s.
pub(super) struct FrameStats {
    frames: u64,
    window_start: Instant,
    window_frames: u32,
    collect: Duration,
    submit: Duration,
}

impl Default for FrameStats {
    fn default() -> Self {
        FrameStats { frames: 0, window_start: Instant::now(), window_frames: 0, collect: Duration::ZERO, submit: Duration::ZERO }
    }
}

impl FrameStats {
    fn record(&mut self, drawn: Drawn, collect: Duration, submit: Duration) {
        self.frames += 1;
        self.window_frames += 1;
        self.collect += collect;
        self.submit += submit;
        let elapsed = self.window_start.elapsed();
        if elapsed.as_secs() < 5 {
            return;
        }
        let per_frame = |d: Duration| d.as_secs_f64() * 1000.0 / self.window_frames as f64;
        let ms = per_frame(elapsed);
        println!("gles: {ms:.1} ms/frame ({:.0} fps), {} draws, {} tris; ECS collect {:.2} ms, GL submit {:.2} ms (per frame)", 1000.0 / ms, drawn.draws, drawn.tris, per_frame(self.collect), per_frame(self.submit));
        self.window_start = Instant::now();
        self.window_frames = 0;
        self.collect = Duration::ZERO;
        self.submit = Duration::ZERO;
    }
}

impl Gles {
    fn resize(&mut self, size: (u32, u32)) {
        if self.size != size {
            self.surface.resize(&self.ctx, NonZeroU32::new(size.0).unwrap(), NonZeroU32::new(size.1).unwrap());
            self.size = size;
        }
    }

    /// Draws every camera's pass in order, then the HUD over them.
    fn draw(&mut self, cams: &[Cam], passes: &mut [Pass], hud: &[CanvasDraw], lighting: &Lighting) -> Drawn {
        let size = self.size;
        let mut drawn = Drawn::default();
        unsafe {
            self.gl.viewport(0, 0, size.0 as i32, size.1 as i32);
        }
        for (cam, pass) in cams.iter().zip(passes) {
            unsafe {
                self.gl.depth_mask(true);
                match cam.clear {
                    Some(color) => {
                        let s = Srgba::from(color);
                        self.gl.clear_color(s.red, s.green, s.blue, 1.0);
                        self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                    }
                    None => self.gl.clear(glow::DEPTH_BUFFER_BIT),
                }
            }
            if let Some(world) = self.renderer.as_mut().filter(|_| cam.draws_world()) {
                if let Some((start, end, color)) = cam.fog {
                    world.fog = (start, end);
                    world.fog_color = color;
                }
                world.render(cam.view_proj, cam.pos, size);
                drawn.draws += world.stats.draws;
                drawn.tris += world.stats.tris;
            }
            let env = lighting.env(cam.fog);
            self.dynamic.render(&self.gl, cam.view_proj, cam.pos, &env, &mut pass.meshes);
            self.dynamic.render_particles(&self.gl, cam.view_proj, cam.pos, &env, &mut pass.particles);
            drawn.draws += self.dynamic.draws;
            drawn.tris += self.dynamic.tris;
        }
        // `render_canvas` adds to the counters of the last pass.
        let before = self.dynamic.draws;
        self.dynamic.render_canvas(&self.gl, size, hud);
        drawn.draws += self.dynamic.draws - before;
        drawn
    }

    /// The finished frame as a top-down RGBA image.
    fn read_frame(&self) -> Image {
        let (w, h) = self.size;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        unsafe {
            self.gl.finish();
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            self.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut pixels)));
        }
        let flipped = pixels.chunks_exact((w * 4) as usize).rev().flatten().copied().collect();
        Image::new(
            bevy::render::render_resource::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            bevy::render::render_resource::TextureDimension::D2,
            flipped,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            bevy::asset::RenderAssetUsages::default(),
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn draw_frame(
    gles: Option<NonSendMut<Gles>>,
    window: Query<&Window, With<PrimaryWindow>>,
    cameras: Cameras,
    scene: SceneMeshes,
    hud: Hud,
    meshes: Res<Assets<Mesh>>,
    images: Res<Assets<Image>>,
    grade: Query<&crate::settings::Grade>,
    screenshots: Query<Entity, With<Screenshot>>,
    mut commands: Commands,
) {
    let Some(mut gles) = gles else { return };
    let gles = &mut *gles;
    let Some(window) = window.iter().next() else { return };
    let start = Instant::now();
    let cams = cameras.active();
    if cams.is_empty() {
        return;
    }
    gles.resize((window.physical_width().max(1), window.physical_height().max(1)));
    let mut passes: Vec<Pass> = cams.iter().map(|_| Pass::default()).collect();
    scene.collect_meshes(&cams, gles, &meshes, &images, &mut passes);
    scene.collect_particles(&cams, gles, &meshes, &images, &mut passes);
    let canvas = hud.collect(gles, &meshes, &images);
    let collected = start.elapsed();

    let drawn = gles.draw(&cams, &mut passes, &canvas, &cameras.lighting());
    // Brightness / contrast / gamma from the options menu (a component of the canvas camera).
    if let Some(g) = grade.iter().next() {
        gles.dynamic.render_grade(&gles.gl, gles.size, g.offset, g.slope, g.inv_gamma);
    }
    // Screenshot requests (Bevy's own capture needs its renderer): read the finished frame.
    if !screenshots.is_empty() {
        let image = gles.read_frame();
        for entity in &screenshots {
            commands.trigger(ScreenshotCaptured { entity, image: image.clone() });
            commands.entity(entity).despawn();
        }
    }
    let _ = gles.surface.swap_buffers(&gles.ctx);
    gles.stats.record(drawn, collected, start.elapsed() - collected);
    if gles.stats.frames % 300 == 0 {
        gles.drop_dead_assets(&meshes, &images);
    }
}
