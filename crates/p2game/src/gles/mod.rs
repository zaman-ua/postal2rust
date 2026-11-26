//! `P2GAME_GLES=1`: Bevy runs the game (script, physics, AI, sound, ECS) with its renderer switched
//! off (`WgpuSettings::backends = None` → no RenderApp, no extract/queue/prepare, ~40 ms per
//! frame saved on Mali) and p2gles draws the frame: the baked world from `SceneData`, then dynamic
//! meshes, particles and the HUD read from the ECS.
//!
//! The GL context is created on Bevy's own winit window (EGL, GLES 3.0) on the main thread; every
//! GL call happens there, through the non-send [`Gles`] resource.
//!
//! - [`bake`]: which level geometry goes into the static world buffers, and when;
//! - [`collect`]: the ECS read into p2gles draw lists (cameras, meshes, particles, canvas);
//! - [`assets`]: GPU copies of Bevy meshes and images;
//! - [`frame`]: the per-frame system that ties them together.

mod assets;
mod bake;
mod collect;
mod frame;

pub(crate) use bake::{bake_world, classify_baked, queue_bake, LevelBake};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use glutin::config::{Api, ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, NotCurrentGlContext, PossiblyCurrentContext, Version};
use glutin::display::{Display, DisplayApiPreference, GlDisplay};
use glutin::prelude::*;
use glutin::surface::{Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface};
use p2gles::dynamic::Dynamic;
use p2gles::renderer::{Options, Renderer};
use p2view::scene::SceneData;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashSet;
use std::num::NonZeroU32;

pub fn enabled() -> bool {
    std::env::var_os("P2GAME_GLES").is_some_and(|v| v != "0")
}

/// GL state: context + surface exist once the window does; the renderer once a level was baked.
pub struct Gles {
    gl: glow::Context,
    ctx: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    display: Display,
    /// The current level's static world.
    renderer: Option<Renderer>,
    /// Everything else, and the cache of GPU meshes and textures.
    dynamic: Dynamic,
    stale: assets::Stale,
    size: (u32, u32),
    stats: frame::FrameStats,
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
        let gl = load_gl(&display);
        let dynamic = Dynamic::new(&gl);
        Ok(Gles { gl, ctx, surface, display, renderer: None, dynamic, stale: default(), size, stats: default() })
    }

    /// Frees the previous level's world; the new one is baked a couple of script frames later.
    pub fn drop_world(&mut self) {
        self.renderer = None;
    }

    /// Bakes the level's static geometry into GL buffers (replaces the previous level's).
    fn load_world(&mut self, data: &SceneData, skip_exports: HashSet<usize>) {
        let dre = self.display.get_proc_address(c"glDrawRangeElements");
        if !dre.is_null() {
            let _ = p2gles::renderer::DRAW_RANGE_ELEMENTS.set(dre as usize);
        }
        self.drop_world(); // free the old level's GL objects first
        let t = std::time::Instant::now();
        let opts = Options { chunk: 48.0, fog: None, no_cull: false, skip_exports };
        // The renderer owns its own function table for the same context.
        let mut r = Renderer::new(load_gl(&self.display), data, &opts);
        // The bridge clears per camera.
        r.clear = false;
        self.renderer = Some(r);
        println!("gles: world baked in {:.2} s", t.elapsed().as_secs_f32());
    }
}

fn load_gl(display: &Display) -> glow::Context {
    // SAFETY: the loader resolves symbols of the current context's display.
    unsafe { glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s)) }
}

pub struct GlesPlugin;

impl Plugin for GlesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<bake::GlBaked>().add_systems(PreUpdate, init_gles).add_systems(Last, (assets::track_assets, frame::draw_frame).chain());
    }
}

fn init_gles(world: &mut World) {
    if world.get_non_send::<Gles>().is_some() {
        return;
    }
    // Context creation needs the primary window's native handles.
    let Some(win) = world.query_filtered::<&Window, With<PrimaryWindow>>().iter(world).next() else { return };
    let size = (win.physical_width(), win.physical_height());
    let vsync = !matches!(win.present_mode, bevy::window::PresentMode::AutoNoVsync | bevy::window::PresentMode::Immediate);
    let made = bevy::winit::WINIT_WINDOWS.with_borrow(|ww| ww.windows.values().next().map(|w| Gles::create(&**w, size, vsync)));
    match made {
        Some(Ok(g)) => {
            println!("gles: context ready ({}x{})", size.0, size.1);
            world.insert_non_send(g);
        }
        Some(Err(e)) => {
            eprintln!("gles: cannot create the OpenGL ES 3.0 context on the game window: {e}\nrun without P2GAME_GLES=1 to use Bevy's renderer, or check EGL/Mesa (eglinfo, GL_VERSION >= ES 3.0)");
            std::process::exit(1);
        }
        None => {}
    }
}
