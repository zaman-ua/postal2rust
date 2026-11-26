//! The per-frame system: collect from the ECS and hand the frame to the backend.

use super::backend::{CameraFrame, Cmd, Done, Drawn, Frame};
use super::collect::{Cameras, Hud, Pass, SceneMeshes};
use super::Gles;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::PrimaryWindow;
use std::time::{Duration, Instant};

/// Timing report, printed every 5 s.
pub(super) struct FrameStats {
    frames: u64,
    window_start: Instant,
    window_frames: u32,
    collect: Duration,
    wait: Duration,
    render: Duration,
    drawn: Drawn,
}

impl Default for FrameStats {
    fn default() -> Self {
        FrameStats { frames: 0, window_start: Instant::now(), window_frames: 0, collect: Duration::ZERO, wait: Duration::ZERO, render: Duration::ZERO, drawn: Drawn::default() }
    }
}

impl FrameStats {
    fn record(&mut self, collect: Duration, wait: Duration, threaded: bool) {
        self.frames += 1;
        self.window_frames += 1;
        self.collect += collect;
        self.wait += wait;
        let elapsed = self.window_start.elapsed();
        if elapsed.as_secs() < 5 {
            return;
        }
        let frames = self.window_frames as f64;
        let per_frame = |d: Duration| d.as_secs_f64() * 1000.0 / frames;
        let ms = per_frame(elapsed);
        let wait = if threaded { format!(", main thread waited {:.2} ms for it", per_frame(self.wait)) } else { String::new() };
        println!(
            "gles: {ms:.1} ms/frame ({:.0} fps), {} draws, {} tris; ECS collect {:.2} ms, GL draw+swap {:.2} ms{wait} (per frame)",
            1000.0 / ms,
            self.drawn.draws,
            self.drawn.tris,
            per_frame(self.collect),
            per_frame(self.render)
        );
        self.window_start = Instant::now();
        self.window_frames = 0;
        self.collect = Duration::ZERO;
        self.wait = Duration::ZERO;
        self.render = Duration::ZERO;
    }
}

impl Gles {
    /// Takes in the answer to a drawn frame.
    fn finished(&mut self, done: Done, commands: &mut Commands) {
        self.stats.render += done.busy;
        self.stats.drawn = done.drawn;
        let Some((width, height, pixels)) = done.capture else { return };
        // Rows come bottom-up.
        let flipped = pixels.chunks_exact((width * 4) as usize).rev().flatten().copied().collect();
        let image = Image::new(
            bevy::render::render_resource::Extent3d { width, height, depth_or_array_layers: 1 },
            bevy::render::render_resource::TextureDimension::D2,
            flipped,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            bevy::asset::RenderAssetUsages::default(),
        );
        for entity in self.capturing.drain(..) {
            commands.trigger(ScreenshotCaptured { entity, image: image.clone() });
            commands.entity(entity).try_despawn();
        }
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
    let mut passes: Vec<Pass> = cams.iter().map(|_| Pass::default()).collect();
    scene.collect_meshes(&cams, gles, &meshes, &images, &mut passes);
    scene.collect_particles(&cams, gles, &meshes, &images, &mut passes);
    let canvas = hud.collect(gles, &meshes, &images);
    let lighting = cameras.lighting();
    let collected = start.elapsed();

    // The previous frame, if the render thread is still on it.
    if let Some(done) = gles.link.wait() {
        gles.finished(done, &mut commands);
    }
    let waited = start.elapsed() - collected;

    // Screenshot requests (Bevy's own capture needs its renderer): read the finished frame. A
    // request stays an entity until its picture arrives, so one capture runs at a time.
    let capture = gles.capturing.is_empty() && {
        gles.capturing.extend(screenshots.iter());
        !gles.capturing.is_empty()
    };
    let cameras = cams
        .into_iter()
        .zip(passes)
        .map(|(cam, pass)| CameraFrame {
            view_proj: cam.view_proj,
            pos: cam.pos,
            clear: cam.clear.map(|color| {
                let s = Srgba::from(color);
                [s.red, s.green, s.blue]
            }),
            fog: cam.fog,
            world: cam.draws_world(),
            env: lighting.env(cam.fog),
            meshes: pass.meshes,
            particles: pass.particles,
        })
        .collect();
    gles.link.send(Cmd::Frame(Box::new(Frame {
        size: (window.physical_width().max(1), window.physical_height().max(1)),
        cameras,
        hud: canvas,
        hud_size: hud.size(),
        // Brightness / contrast / gamma from the options menu (a component of the canvas camera).
        grade: grade.iter().next().map(|g| (g.offset, g.slope, g.inv_gamma)),
        capture,
    })));
    // Inline, the frame is already drawn.
    if !gles.link.threaded() {
        if let Some(done) = gles.link.wait() {
            gles.finished(done, &mut commands);
        }
    }
    gles.stats.record(collected, waited, gles.link.threaded());
    if gles.stats.frames % 300 == 0 {
        gles.drop_dead_assets(&meshes, &images);
    }
}
