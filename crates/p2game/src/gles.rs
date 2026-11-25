//! `P2GAME_GLES=1`: Bevy runs the game (script, physics, AI, sound, ECS) with its renderer switched
//! off (`WgpuSettings::backends = None` → no RenderApp, no extract/queue/prepare, ~40 ms per
//! frame saved on Mali) and p2gles draws the frame: the baked world from `SceneData`, then (later
//! passes) dynamic meshes read from the ECS.
//!
//! The GL context is created on Bevy's own winit window (EGL, GLES 3.0) on the main thread; every
//! GL call happens in `Last` through the non-send [`Gles`] resource.

use bevy::camera::Camera3d;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use glutin::config::{Api, ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{Display, DisplayApiPreference, GlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use p2gles::renderer::{Options, Renderer};
use p2view::scene::SceneData;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::num::NonZeroU32;

pub fn enabled() -> bool {
    std::env::var_os("P2GAME_GLES").is_some_and(|v| v != "0")
}

/// GL state: context + surface exist once the window does; the renderer once a level was loaded.
pub struct Gles {
    ctx: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    display: Display,
    renderer: Option<Renderer>,
    size: (u32, u32),
    frames: u64,
    win_start: std::time::Instant,
    win_frames: u32,
}

impl Gles {
    fn create(window: &(impl HasWindowHandle + HasDisplayHandle), size: (u32, u32), vsync: bool) -> Result<Gles, String> {
        let raw_display = window.display_handle().map_err(|e| e.to_string())?.as_raw();
        let raw_window = window.window_handle().map_err(|e| e.to_string())?.as_raw();
        // SAFETY: the handles belong to Bevy's live winit window, which outlives this resource
        // (both are dropped with the app on the main thread).
        let display = unsafe { Display::new(raw_display, DisplayApiPreference::Egl) }.map_err(|e| format!("EGL display: {e}"))?;
        let template = ConfigTemplateBuilder::new().compatible_with_native_window(raw_window).with_api(Api::GLES3).with_depth_size(24).with_alpha_size(0).build();
        let config = unsafe { display.find_configs(template) }.map_err(|e| e.to_string())?.reduce(|a, b| if b.num_samples() < a.num_samples() { b } else { a }).ok_or("no EGL config")?;
        let cattrs = ContextAttributesBuilder::new().with_context_api(ContextApi::Gles(Some(Version::new(3, 0)))).build(Some(raw_window));
        let ctx = unsafe { display.create_context(&config, &cattrs) }.map_err(|e| format!("GLES 3.0 context: {e}"))?;
        let sattrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(raw_window, NonZeroU32::new(size.0.max(1)).unwrap(), NonZeroU32::new(size.1.max(1)).unwrap());
        let surface = unsafe { display.create_window_surface(&config, &sattrs) }.map_err(|e| format!("surface: {e}"))?;
        let ctx = ctx.make_current(&surface).map_err(|e| e.to_string())?;
        let swap = if vsync { SwapInterval::Wait(NonZeroU32::new(1).unwrap()) } else { SwapInterval::DontWait };
        if let Err(e) = surface.set_swap_interval(&ctx, swap) {
            eprintln!("gles: swap interval: {e}");
        }
        Ok(Gles { ctx, surface, display, renderer: None, size, frames: 0, win_start: std::time::Instant::now(), win_frames: 0 })
    }

    /// Bakes the level's static geometry into GL buffers (replaces the previous level's).
    pub fn load_world(&mut self, data: &SceneData) {
        let gl = unsafe { glow::Context::from_loader_function_cstr(|s| self.display.get_proc_address(s)) };
        let dre = self.display.get_proc_address(c"glDrawRangeElements");
        if !dre.is_null() {
            let _ = p2gles::renderer::DRAW_RANGE_ELEMENTS.set(dre as usize);
        }
        self.renderer = None; // free the old level's GL objects first
        let t = std::time::Instant::now();
        let opts = Options { chunk: 48.0, fog: None, no_cull: false };
        self.renderer = Some(Renderer::new(gl, data, &opts));
        println!("gles: world baked in {:.2} s", t.elapsed().as_secs_f32());
    }
}

pub struct GlesPlugin;

impl Plugin for GlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, init_gles).add_systems(Last, draw_frame);
    }
}

fn init_gles(world: &mut World) {
    // Context creation needs the primary window's native handles.
    if world.get_non_send_resource::<Gles>().is_some() {
        return;
    }
    let Some(win) = world.query_filtered::<&Window, With<PrimaryWindow>>().iter(world).next() else { return };
    let size = (win.physical_width(), win.physical_height());
    let vsync = !matches!(win.present_mode, bevy::window::PresentMode::AutoNoVsync | bevy::window::PresentMode::Immediate);
    let made = bevy::winit::WINIT_WINDOWS.with_borrow(|ww| ww.windows.values().next().map(|w| Gles::create(&**w, size, vsync)));
    match made {
        Some(Ok(g)) => {
            println!("gles: context ready ({}x{})", size.0, size.1);
            world.insert_non_send_resource(g);
        }
        Some(Err(e)) => panic!("gles: {e}"),
        None => {}
    }
}

fn draw_frame(world: &mut World) {
    if world.get_non_send_resource::<Gles>().is_none() {
        return;
    }
    let (view, cam, fov, aspect) = {
        let mut q = world.query_filtered::<(&GlobalTransform, &Projection, &Camera), With<Camera3d>>();
        let mut best: Option<(isize, Mat4, Vec3, f32, f32)> = None;
        for (gt, proj, c) in q.iter(world) {
            if !c.is_active {
                continue;
            }
            let Projection::Perspective(p) = proj else { continue };
            if best.as_ref().is_none_or(|b| c.order < b.0) {
                best = Some((c.order, gt.to_matrix().inverse(), gt.translation(), p.fov, p.aspect_ratio));
            }
        }
        let Some((_, v, t, f, a)) = best else { return };
        (v, t, f, a)
    };
    let Some(win) = world.query_filtered::<&Window, With<PrimaryWindow>>().iter(world).next() else { return };
    let size = (win.physical_width().max(1), win.physical_height().max(1));
    let mut g = world.non_send_resource_mut::<Gles>();
    let g = &mut *g;
    if g.size != size {
        g.surface.resize(&g.ctx, NonZeroU32::new(size.0).unwrap(), NonZeroU32::new(size.1).unwrap());
        g.size = size;
    }
    let Some(r) = g.renderer.as_mut() else { return };
    let far = (r.fog.1 * 1.05).max(10.0);
    let vp = Mat4::perspective_rh_gl(fov, aspect, 0.1, far) * view;
    r.render(vp, cam, size);
    let _ = g.surface.swap_buffers(&g.ctx);
    g.frames += 1;
    g.win_frames += 1;
    if g.win_start.elapsed().as_secs() >= 5 {
        let ms = g.win_start.elapsed().as_secs_f64() * 1000.0 / g.win_frames as f64;
        println!("gles: {ms:.1} ms/frame ({:.0} fps), {} draws, {} tris", 1000.0 / ms, r.stats.draws, r.stats.tris);
        g.win_start = std::time::Instant::now();
        g.win_frames = 0;
    }
}
