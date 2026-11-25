//! p2gles: stage-0 thin OpenGL ES 3.0 map renderer for POSTAL 2 (UE2-style baked lighting).
//!
//! Usage: p2gles [--game DIR] [--map suburbs-3] [--res 800x480] [--cam x,y,z,yaw,pitch]
//!               [--bench SECONDS] [--screenshot out.png] [--fog start,end] [--chunk METRES]
//! Interactive: WASD move, Q/E down/up, Shift fast, mouse look, Esc quit.

mod renderer;
mod shaders;

use bevy_math::{Mat4, Vec3};
use glutin::config::{Api, ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{GetGlDisplay, GlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use p2view::scene::{self, Options};
use raw_window_handle::HasWindowHandle;
use std::collections::HashSet;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const ROT: f32 = std::f32::consts::TAU / 65536.0;

struct Args {
    game: PathBuf,
    map: String,
    res: (u32, u32),
    cam: Option<[f32; 5]>,
    bench: Option<f32>,
    shot: Option<PathBuf>,
    fog: Option<(f32, f32)>,
    chunk: f32,
    no_cull: bool,
    vsync: bool,
    fullscreen: bool,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let nums = |s: String| s.split(',').filter_map(|x| x.trim().parse::<f32>().ok()).collect::<Vec<_>>();
    Args {
        game: get("--game").map(PathBuf::from).unwrap_or_else(|| format!("{}/.local/share/Steam/steamapps/common/POSTAL2Complete", std::env::var("HOME").unwrap_or_default()).into()),
        map: get("--map").unwrap_or_else(|| "suburbs-3".into()),
        res: get("--res").and_then(|s| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))).unwrap_or((800, 480)),
        cam: get("--cam").map(nums).filter(|v| v.len() == 5).map(|v| [v[0], v[1], v[2], v[3], v[4]]),
        bench: get("--bench").and_then(|s| s.parse().ok()),
        shot: get("--screenshot").map(PathBuf::from),
        fog: get("--fog").map(nums).filter(|v| v.len() == 2).map(|v| (v[0], v[1])),
        chunk: get("--chunk").and_then(|s| s.parse().ok()).unwrap_or(48.0),
        no_cull: a.iter().any(|x| x == "--no-cull"),
        vsync: a.iter().any(|x| x == "--vsync"),
        fullscreen: a.iter().any(|x| x == "--fullscreen"),
    }
}

struct Gfx {
    window: Window,
    surface: Surface<WindowSurface>,
    ctx: PossiblyCurrentContext,
    r: renderer::Renderer,
}

struct App {
    args: Args,
    data: Option<scene::SceneData>,
    gfx: Option<Gfx>,
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    keys: HashSet<KeyCode>,
    last: Instant,
    start: Instant,
    frames: u64,
    win_frames: u32,
    win_start: Instant,
    win_draws: u64,
    win_tris: u64,
    shot_done: bool,
}

impl App {
    fn view_proj(&self) -> Mat4 {
        let dir = Vec3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos());
        let (w, h) = self.args.res;
        let far = self.gfx.as_ref().map(|g| g.r.fog.1 * 1.05).unwrap_or(1500.0);
        Mat4::perspective_rh_gl(75f32.to_radians(), w as f32 / h as f32, 0.1, far.max(10.0)) * Mat4::look_to_rh(self.pos, dir, Vec3::Y)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let (w, h) = self.args.res;
        let attrs = Window::default_attributes().with_title(format!("p2gles - {}", self.args.map)).with_inner_size(PhysicalSize::new(w, h)).with_resizable(false);
        // --fullscreen: borderless over the whole screen (--res must match the screen mode).
        let attrs = if self.args.fullscreen { attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))) } else { attrs };
        let template = ConfigTemplateBuilder::new().with_api(Api::GLES3).with_depth_size(24).with_alpha_size(0);
        let (window, config) = DisplayBuilder::new()
            .with_preference(glutin_winit::ApiPreference::PreferEgl)
            .with_window_attributes(Some(attrs))
            .build(el, template, |configs| configs.reduce(|a, b| if b.num_samples() < a.num_samples() { b } else { a }).unwrap())
            .expect("create display");
        let window = window.expect("window");
        let raw = window.window_handle().unwrap().as_raw();
        let display = config.display();
        let cattrs = ContextAttributesBuilder::new().with_context_api(ContextApi::Gles(Some(Version::new(3, 0)))).build(Some(raw));
        let ctx = unsafe { display.create_context(&config, &cattrs).expect("create GLES 3.0 context") };
        let sattrs = window.build_surface_attributes(Default::default()).expect("surface attrs");
        let surface = unsafe { display.create_window_surface(&config, &sattrs).expect("surface") };
        let ctx = ctx.make_current(&surface).expect("make current");
        let swap = if self.args.vsync { SwapInterval::Wait(NonZeroU32::new(1).unwrap()) } else { SwapInterval::DontWait };
        if let Err(e) = surface.set_swap_interval(&ctx, swap) {
            eprintln!("swap interval: {e}");
        }
        let gl = unsafe { glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s)) };
        let dre = display.get_proc_address(c"glDrawRangeElements");
        if !dre.is_null() {
            let _ = renderer::DRAW_RANGE_ELEMENTS.set(dre as usize);
        }
        {
            use glow::HasContext;
            unsafe {
                println!("GL_VERSION  : {}", gl.get_parameter_string(glow::VERSION));
                println!("GL_RENDERER : {}", gl.get_parameter_string(glow::RENDERER));
                println!("GLSL        : {}", gl.get_parameter_string(glow::SHADING_LANGUAGE_VERSION));
            }
        }
        let data = self.data.take().unwrap();
        let ropts = renderer::Options { chunk: self.args.chunk, fog: self.args.fog, no_cull: self.args.no_cull };
        let r = renderer::Renderer::new(gl, &data, &ropts);
        // Camera.
        match self.args.cam {
            Some(v) => {
                self.pos = scene::to_bevy([v[0], v[1], v[2]]);
                self.yaw = v[3] * ROT;
                self.pitch = v[4] * ROT;
            }
            None => {
                let (p, d) = data.spawn;
                self.pos = p + Vec3::Y * 1.2;
                self.yaw = d.z.atan2(d.x);
                self.pitch = 0.0;
            }
        }
        println!("camera: pos {:?} yaw {:.1} deg pitch {:.1} deg; fog {:?}", self.pos, self.yaw.to_degrees(), self.pitch.to_degrees(), r.fog);
        if self.args.bench.is_none() && self.args.shot.is_none() {
            let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            window.set_cursor_visible(false);
        }
        self.gfx = Some(Gfx { window, surface, ctx, r });
        self.last = Instant::now();
        self.start = Instant::now();
        self.win_start = Instant::now();
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: winit::event::DeviceId, ev: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = ev {
            if self.args.bench.is_none() && self.args.shot.is_none() {
                self.yaw += delta.0 as f32 * 0.0025;
                self.pitch = (self.pitch - delta.1 as f32 * 0.0025).clamp(-1.55, 1.55);
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(k) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        if k == KeyCode::Escape {
                            el.exit();
                        }
                        self.keys.insert(k);
                    } else {
                        self.keys.remove(&k);
                    }
                }
            }
            WindowEvent::RedrawRequested => self.frame(el),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }
}

impl App {
    fn frame(&mut self, el: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        if self.args.bench.is_none() && self.args.shot.is_none() {
            let dir = Vec3::new(self.yaw.cos() * self.pitch.cos(), self.pitch.sin(), self.yaw.sin() * self.pitch.cos());
            let right = dir.cross(Vec3::Y).normalize();
            let speed = if self.keys.contains(&KeyCode::ShiftLeft) { 40.0 } else { 10.0 };
            let mut mv = Vec3::ZERO;
            for (k, v) in [(KeyCode::KeyW, dir), (KeyCode::KeyS, -dir), (KeyCode::KeyD, right), (KeyCode::KeyA, -right), (KeyCode::KeyE, Vec3::Y), (KeyCode::KeyQ, -Vec3::Y)] {
                if self.keys.contains(&k) {
                    mv += v;
                }
            }
            self.pos += mv * speed * dt;
        }
        let vp = self.view_proj();
        let (w, h) = self.args.res;
        let pos = self.pos;
        let g = self.gfx.as_mut().unwrap();
        g.r.render(vp, pos, (w, h));
        self.frames += 1;
        // Screenshot: render a few frames then read back.
        if let Some(path) = self.args.shot.clone() {
            if self.frames >= 4 && !self.shot_done {
                let px = g.r.read_pixels(w, h);
                let mut flipped = Vec::with_capacity(px.len());
                for row in (0..h as usize).rev() {
                    flipped.extend_from_slice(&px[row * w as usize * 4..(row + 1) * w as usize * 4]);
                }
                let f = std::fs::File::create(&path).expect("create png");
                let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                enc.write_header().unwrap().write_image_data(&flipped).unwrap();
                println!("screenshot -> {} ({} draws, {} tris)", path.display(), g.r.stats.draws, g.r.stats.tris);
                self.shot_done = true;
                el.exit();
                return;
            }
        }
        g.surface.swap_buffers(&g.ctx).expect("swap");
        // Bench accounting (wall clock, including swap).
        self.win_frames += 1;
        self.win_draws += g.r.stats.draws as u64;
        self.win_tris += g.r.stats.tris;
        if let Some(secs) = self.args.bench {
            let el_win = self.win_start.elapsed();
            if el_win >= Duration::from_secs(5) {
                let n = self.win_frames as f64;
                let ms = el_win.as_secs_f64() * 1000.0 / n;
                println!("[{:5.1}s] avg {:.2} ms/frame  {:.1} fps  {} draws/frame  {} tris/frame", self.start.elapsed().as_secs_f32(), ms, 1000.0 / ms, self.win_draws / self.win_frames as u64, self.win_tris / self.win_frames as u64);
                self.win_frames = 0;
                self.win_draws = 0;
                self.win_tris = 0;
                self.win_start = Instant::now();
            }
            if self.start.elapsed().as_secs_f32() >= secs {
                g.r.finish();
                el.exit();
            }
        }
    }
}

fn main() {
    let args = parse_args();
    let t0 = Instant::now();
    let lib = upkg::library::Library::new(&args.game).unwrap_or_else(|e| panic!("cannot index game install at {}: {e}", args.game.display()));
    let path = lib.map_path(&args.map).unwrap_or_else(|| panic!("map '{}' not found", args.map));
    let map = Arc::new(upkg::Package::open(&path).unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display())));
    let data = scene::build_scene(&lib, &map, &Options { gpu_bc: std::env::var_os("P2GLES_NO_BC").is_none(), projectors: false }).unwrap_or_else(|e| panic!("build_scene: {e}"));
    let s = &data.stats;
    println!("build_scene {} in {} ms (wall {} ms): bsp_polys={} instances={} world_batches={} materials={} textures={} lightmaps={} terrain_layers={} spawn_found={}", args.map, s.load_ms, t0.elapsed().as_millis(), s.bsp_polys, data.instances.len(), data.world.len(), data.materials.len(), data.textures.len(), data.lightmaps.len(), data.terrain.len(), data.spawn_found);
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        args,
        data: Some(data),
        gfx: None,
        pos: Vec3::ZERO,
        yaw: 0.0,
        pitch: 0.0,
        keys: HashSet::new(),
        last: Instant::now(),
        start: Instant::now(),
        frames: 0,
        win_frames: 0,
        win_start: Instant::now(),
        win_draws: 0,
        win_tris: 0,
        shot_done: false,
    };
    el.run_app(&mut app).unwrap();
}
