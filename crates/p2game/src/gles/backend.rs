//! The GL side of the bridge: owns the context and draws what it is told through [`Cmd`]s, with no
//! access to the ECS. [`Link`] runs it either inline or on its own thread.
//!
//! On its own thread the game and the renderer overlap: while frame N is being submitted to the GPU
//! (and swap waits for it) the main thread already runs the game for frame N+1. At most one frame
//! is in flight.

use super::scaled::ScaledTarget;
use glow::HasContext;
use glutin::context::{NotCurrentContext, NotCurrentGlContext, PossiblyCurrentContext};
use glutin::display::{Display, GlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, SwapInterval, WindowSurface};
use bevy::math::{Mat4, Vec3};
use p2gles::dynamic::{CanvasDraw, Draw, Dynamic, Env, PackedMesh, ParticleDraw, TexIn};
use p2gles::renderer::{Options, Renderer};
use p2view::scene::{PixelFormat, SceneData};
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A texture's data, owned so it can cross to the render thread.
pub(super) struct Texture {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    /// Mip levels, largest first.
    pub mips: Vec<Vec<u8>>,
    pub repeat: bool,
}

/// What one camera draws.
pub(super) struct CameraFrame {
    pub view_proj: Mat4,
    pub pos: Vec3,
    /// Colour to clear to; `None` keeps the previous camera's picture (depth is always cleared).
    pub clear: Option<[f32; 3]>,
    pub fog: Option<(f32, f32, [f32; 3])>,
    /// The baked world belongs to this camera.
    pub world: bool,
    pub env: Env,
    pub meshes: Vec<Draw>,
    pub particles: Vec<ParticleDraw>,
}

/// Everything needed to draw and present one frame.
pub(super) struct Frame {
    /// Window size in pixels.
    pub size: (u32, u32),
    pub cameras: Vec<CameraFrame>,
    pub hud: Vec<CanvasDraw>,
    /// The size of the canvas the HUD is laid out on.
    pub hud_size: (u32, u32),
    /// Brightness/contrast/gamma: (offset, slope, inverse gamma).
    pub grade: Option<(f32, f32, f32)>,
    /// Read the finished frame back.
    pub capture: bool,
}

pub(super) enum Cmd {
    Mesh(u64, PackedMesh),
    RemoveMesh(u64),
    Texture(u64, Texture),
    /// Free every GPU mesh and texture whose key is not listed.
    Retain(HashSet<u64>, HashSet<u64>),
    DropWorld,
    LoadWorld(Box<SceneData>, HashSet<usize>),
    Frame(Box<Frame>),
}

#[derive(Default, Clone, Copy)]
pub(super) struct Drawn {
    pub draws: u32,
    pub tris: u64,
}

/// The answer to a [`Cmd::Frame`].
pub(super) struct Done {
    pub drawn: Drawn,
    /// Time spent drawing and presenting.
    pub busy: Duration,
    /// (width, height, RGBA rows bottom-up) when the frame asked for a capture.
    pub capture: Option<(u32, u32, Vec<u8>)>,
}

/// The window-system objects, made on the main thread and handed to whichever thread draws.
pub(super) struct Surfaces {
    pub display: Display,
    pub context: NotCurrentContext,
    pub surface: Surface<WindowSurface>,
    pub size: (u32, u32),
    pub vsync: bool,
}

pub(super) struct Backend {
    gl: glow::Context,
    ctx: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    display: Display,
    /// The current level's static world.
    renderer: Option<Renderer>,
    /// Everything else, and the cache of GPU meshes and textures.
    dynamic: Dynamic,
    /// Lower-resolution 3D (`P2GAME_GLES_SCALE`).
    scaled: Option<ScaledTarget>,
    size: (u32, u32),
}

fn load_gl(display: &Display) -> glow::Context {
    // SAFETY: the loader resolves symbols of the current context's display.
    unsafe { glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s)) }
}

impl Backend {
    /// Makes the context current on the calling thread, where it stays.
    fn new(s: Surfaces) -> Result<Backend, String> {
        let ctx = s.context.make_current(&s.surface).map_err(|e| e.to_string())?;
        let swap = if s.vsync { SwapInterval::Wait(NonZeroU32::new(1).unwrap()) } else { SwapInterval::DontWait };
        if let Err(e) = s.surface.set_swap_interval(&ctx, swap) {
            eprintln!("gles: swap interval: {e}");
        }
        let gl = load_gl(&s.display);
        let dynamic = Dynamic::new(&gl);
        Ok(Backend { gl, ctx, surface: s.surface, display: s.display, renderer: None, dynamic, scaled: ScaledTarget::from_env(), size: s.size })
    }

    fn execute(&mut self, cmd: Cmd) -> Option<Done> {
        match cmd {
            Cmd::Mesh(key, mesh) => self.dynamic.add_packed(&self.gl, key, &mesh),
            Cmd::RemoveMesh(key) => self.dynamic.remove_mesh(&self.gl, key),
            Cmd::Texture(key, t) => {
                let mips = t.mips.iter().map(Vec::as_slice).collect();
                self.dynamic.add_tex(&self.gl, key, &TexIn { width: t.width, height: t.height, format: t.format, mips, repeat: t.repeat });
            }
            Cmd::Retain(meshes, textures) => self.dynamic.retain(&self.gl, &meshes, &textures),
            Cmd::DropWorld => self.renderer = None,
            Cmd::LoadWorld(data, skip_exports) => self.load_world(&data, skip_exports),
            Cmd::Frame(frame) => return Some(self.present(*frame)),
        }
        None
    }

    /// Bakes a level's static geometry into GL buffers (replaces the previous level's).
    fn load_world(&mut self, data: &SceneData, skip_exports: HashSet<usize>) {
        let dre = self.display.get_proc_address(c"glDrawRangeElements");
        if !dre.is_null() {
            let _ = p2gles::renderer::DRAW_RANGE_ELEMENTS.set(dre as usize);
        }
        self.renderer = None; // free the old level's GL objects first
        let t = Instant::now();
        let opts = Options { chunk: 48.0, fog: None, no_cull: false, skip_exports };
        // The renderer owns its own function table for the same context.
        let mut r = Renderer::new(load_gl(&self.display), data, &opts);
        // The bridge clears per camera.
        r.clear = false;
        self.renderer = Some(r);
        println!("gles: world baked in {:.2} s", t.elapsed().as_secs_f32());
    }

    fn present(&mut self, mut frame: Frame) -> Done {
        let start = Instant::now();
        if self.size != frame.size {
            self.surface.resize(&self.ctx, NonZeroU32::new(frame.size.0).unwrap(), NonZeroU32::new(frame.size.1).unwrap());
            self.size = frame.size;
        }
        let mut drawn = self.draw_cameras(&mut frame.cameras);
        // `render_canvas` adds to the counters of the last pass.
        let before = self.dynamic.draws;
        self.dynamic.render_canvas(&self.gl, frame.hud_size, &frame.hud);
        drawn.draws += self.dynamic.draws - before;
        if let Some((offset, slope, inv_gamma)) = frame.grade {
            self.dynamic.render_grade(&self.gl, self.size, offset, slope, inv_gamma);
        }
        let capture = frame.capture.then(|| self.read_frame());
        let _ = self.surface.swap_buffers(&self.ctx);
        Done { drawn, busy: start.elapsed(), capture }
    }

    /// Draws every camera's pass in order, at the 3D resolution, leaving the window framebuffer
    /// bound with the whole window as viewport for the HUD.
    fn draw_cameras(&mut self, cameras: &mut [CameraFrame]) -> Drawn {
        let window = self.size;
        let size = match &mut self.scaled {
            Some(target) => target.begin(&self.gl, window),
            None => window,
        };
        let mut drawn = Drawn::default();
        unsafe {
            self.gl.viewport(0, 0, size.0 as i32, size.1 as i32);
        }
        for cam in cameras {
            unsafe {
                self.gl.depth_mask(true);
                match cam.clear {
                    Some([r, g, b]) => {
                        self.gl.clear_color(r, g, b, 1.0);
                        self.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                    }
                    None => self.gl.clear(glow::DEPTH_BUFFER_BIT),
                }
            }
            if let Some(world) = self.renderer.as_mut().filter(|_| cam.world) {
                if let Some((start, end, color)) = cam.fog {
                    world.fog = (start, end);
                    world.fog_color = color;
                }
                world.render(cam.view_proj, cam.pos, size);
                drawn.draws += world.stats.draws;
                drawn.tris += world.stats.tris;
            }
            self.dynamic.render(&self.gl, cam.view_proj, cam.pos, &cam.env, &mut cam.meshes);
            self.dynamic.render_particles(&self.gl, cam.view_proj, cam.pos, &cam.env, &mut cam.particles);
            drawn.draws += self.dynamic.draws;
            drawn.tris += self.dynamic.tris;
        }
        if let Some(target) = &self.scaled {
            target.finish(&self.gl, window);
        }
        drawn
    }

    fn read_frame(&self) -> (u32, u32, Vec<u8>) {
        let (w, h) = self.size;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        unsafe {
            self.gl.finish();
            self.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            self.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut pixels)));
        }
        (w, h, pixels)
    }
}

/// How the main thread reaches the backend.
pub(super) enum Link {
    /// Commands run as they are sent; a frame's answer waits for [`Link::wait`].
    Inline { backend: Box<Backend>, done: Option<Done> },
    Thread { commands: Option<Sender<Cmd>>, done: Receiver<Done>, in_flight: bool, thread: Option<JoinHandle<()>> },
}

impl Link {
    pub fn inline(surfaces: Surfaces) -> Result<Link, String> {
        Ok(Link::Inline { backend: Box::new(Backend::new(surfaces)?), done: None })
    }

    pub fn thread(surfaces: Surfaces) -> Result<Link, String> {
        let (commands, queue) = channel::<Cmd>();
        let (done_tx, done) = channel();
        let (ready_tx, ready) = channel();
        let thread = std::thread::Builder::new()
            .name("p2gles-render".into())
            .spawn(move || {
                let mut backend = match Backend::new(surfaces) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let _ = ready_tx.send(Ok(()));
                // Ends when the main thread drops its sender (or stops listening).
                for cmd in queue {
                    if let Some(answer) = backend.execute(cmd) {
                        if done_tx.send(answer).is_err() {
                            break;
                        }
                    }
                }
            })
            .map_err(|e| format!("render thread: {e}"))?;
        ready.recv().map_err(|_| "render thread died while starting".to_string())??;
        Ok(Link::Thread { commands: Some(commands), done, in_flight: false, thread: Some(thread) })
    }

    pub fn threaded(&self) -> bool {
        matches!(self, Link::Thread { .. })
    }

    pub fn send(&mut self, cmd: Cmd) {
        match self {
            Link::Inline { backend, done } => {
                if let Some(answer) = backend.execute(cmd) {
                    *done = Some(answer);
                }
            }
            Link::Thread { commands, in_flight, .. } => {
                *in_flight |= matches!(cmd, Cmd::Frame(_));
                // A dead render thread has already printed its panic.
                commands.as_ref().and_then(|c| c.send(cmd).ok()).expect("the p2gles render thread stopped");
            }
        }
    }

    /// The answer to the last frame sent, waiting for it if it is still being drawn. Must be called
    /// before the next frame is sent.
    pub fn wait(&mut self) -> Option<Done> {
        match self {
            Link::Inline { done, .. } => done.take(),
            Link::Thread { done, in_flight, .. } => std::mem::take(in_flight).then(|| done.recv().expect("the p2gles render thread stopped")),
        }
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        // The thread must stop using the window's surface before the window goes away.
        if let Link::Thread { commands, thread, .. } = self {
            drop(commands.take());
            if let Some(t) = thread.take() {
                let _ = t.join();
            }
        }
    }
}
