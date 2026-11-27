//! `P2GAME_GLES=1`: Bevy runs the game (script, physics, AI, sound, ECS) with its renderer switched
//! off (`WgpuSettings::backends = None` → no RenderApp, no extract/queue/prepare, ~40 ms per
//! frame saved on Mali) and p2gles draws the frame: the baked world from `SceneData`, then dynamic
//! meshes, particles and the HUD read from the ECS.
//!
//! The GL context is created on Bevy's own winit window (EGL, GLES 3.0). All GL calls happen in
//! [`backend`], by default on a render thread of its own (`P2GAME_GLES_THREAD=0`: on the main
//! thread); the main thread reaches it through the non-send [`Gles`] resource.
//!
//! - [`bake`]: which level geometry goes into the static world buffers, and when;
//! - [`collect`]: the ECS read into p2gles draw lists (cameras, meshes, particles, canvas);
//! - [`assets`]: which Bevy meshes and images have GPU copies;
//! - [`frame`]: the per-frame system that ties them together;
//! - [`backend`]: the GL side; [`scaled`]: lower-resolution 3D;
//! - [`view`]: what a camera sees.

mod assets;
mod backend;
mod bake;
mod collect;
mod frame;
mod scaled;
mod view;

pub(crate) use bake::{bake_world, classify_baked, queue_bake, LevelBake};
pub(crate) use view::{fog_end, View};

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use glutin::config::{Api, ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextApi, ContextAttributesBuilder, Version};
use glutin::display::{Display, DisplayApiPreference, GlDisplay};
use glutin::surface::{SurfaceAttributesBuilder, WindowSurface};
use p2view::scene::SceneData;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::HashSet;
use std::num::NonZeroU32;


pub fn enabled() -> bool {
    std::env::var_os("P2GAME_GLES").is_some_and(|v| v != "0")
}

/// The main thread's side of the renderer.
pub struct Gles {
    link: backend::Link,
    assets: assets::Resident,
    stats: frame::FrameStats,
    /// Screenshot entities waiting for the frame being drawn.
    capturing: Vec<Entity>,
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
        let context = unsafe { display.create_context(&config, &cattrs) }.map_err(|e| format!("GLES 3.0 context: {e}"))?;
        let sattrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(raw_window, NonZeroU32::new(size.0.max(1)).unwrap(), NonZeroU32::new(size.1.max(1)).unwrap());
        let surface = unsafe { display.create_window_surface(&config, &sattrs) }.map_err(|e| format!("surface: {e}"))?;
        let surfaces = backend::Surfaces { display, context, surface, size, vsync };
        let wanted = std::env::var_os("P2GAME_GLES_THREAD").is_none_or(|v| v != "0");
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        let link = if wanted && cores > 1 { backend::Link::thread(surfaces)? } else { backend::Link::inline(surfaces)? };
        println!("gles: rendering on {}", if link.threaded() { "its own thread" } else { "the main thread" });
        Ok(Gles { link, assets: default(), stats: default(), capturing: Vec::new() })
    }

    /// Frees the previous level's world; the new one is baked a couple of script frames later.
    pub fn drop_world(&mut self) {
        self.link.send(backend::Cmd::DropWorld);
    }

    /// Bakes the level's static geometry into GL buffers (replaces the previous level's).
    fn load_world(&mut self, data: SceneData, skip_exports: HashSet<usize>) {
        self.link.send(backend::Cmd::LoadWorld(Box::new(data), skip_exports));
    }
}

pub struct GlesPlugin;

impl Plugin for GlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, init_gles).add_systems(Last, (assets::track_assets, frame::draw_frame).chain());
        app.add_systems(Startup, trim_bevy_systems);
        if std::env::var_os("P2GAME_GLES_BEVY_FULL").is_none() && std::env::var("P2GAME_GLES_TRIM").is_ok_and(|t| t.split(',').any(|x| x == "visif")) {
            // Per-view frustum culling: p2gles does its own. A run condition, not removal, so the
            // order of the rest of PostUpdate stays exactly as Bevy built it.
            use bevy::camera::visibility::VisibilitySystems;
            app.configure_sets(PostUpdate, (VisibilitySystems::CheckVisibility, VisibilitySystems::MarkNewlyHiddenEntitiesInvisible).run_if(|| false));
        }
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

/// Whether the p2gles frame drops the piece of Bevy's render bookkeeping named `key`
/// (`P2GAME_GLES_TRIM`, a comma-separated list; default `mat,spec,wire,prev,statics`). Systems
/// removed by [`trim_bevy_systems`]: `mat`, `spec`, `mesh`, `vis`, `prev` (motion-vector history).
/// `wire`: no `WireframePlugin`, whose wireframes only Bevy's renderer draws. `statics`: baked
/// static mesh entities are despawned after the bake, so Bevy's visibility, bounds and
/// asset-change systems stop visiting them (~2.8 ms of `PostUpdate` per frame on suburbs-3).
pub(crate) fn trimmed(key: &str) -> bool {
    static ONLY: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let only = ONLY.get_or_init(|| {
        (enabled() && std::env::var_os("P2GAME_GLES_BEVY_FULL").is_none()).then(|| std::env::var("P2GAME_GLES_TRIM").unwrap_or_else(|_| "mat,spec,wire,prev,statics".into()))
    });
    only.as_deref().is_some_and(|only| only.split(',').any(|x| x == key))
}

/// Bevy's renderer is absent, but its main-world bookkeeping for it still runs every frame:
/// per-view frustum culling (`ViewVisibility`, which p2gles does not read: it culls by itself) and
/// marking `Mesh3d` changed when a mesh or material asset changes (for re-extraction and pipeline
/// specialisation). Each of the latter scans every mesh entity with an `AssetChanged` filter. p2gles
/// tracks assets through `AssetEvent`s instead ([`assets`]), so these are removed from `PostUpdate`.
/// `P2GAME_GLES_BEVY_FULL=1` keeps them; `P2GAME_LIST_SYSTEMS=1` prints what `PostUpdate` runs.
fn trim_bevy_systems(world: &mut World) {
    use bevy::camera::visibility::VisibilitySystems;
    use bevy::ecs::schedule::ScheduleCleanupPolicy::RemoveSystemsOnly;
    use bevy::ecs::schedule::ScheduleLabel;
    use bevy::pbr::{check_entities_needing_specialization as spec, mark_meshes_as_changed_if_their_materials_changed as mat, StandardMaterial};
    if std::env::var_os("P2GAME_GLES_BEVY_FULL").is_some() {
        return;
    }
    world.resource_scope(|world, mut schedules: Mut<Schedules>| {
        let mut removed = 0;
        let mut rm = |n: Result<usize, bevy::ecs::schedule::ScheduleError>| removed += n.unwrap_or(0);
        let on = trimmed;
        if on("prev") {
            // Previous-frame transforms for motion vectors, which p2gles does not draw.
            rm(schedules.remove_systems_in_set(PreUpdate, bevy::pbr::update_mesh_previous_global_transforms, world, RemoveSystemsOnly));
        }
        if on("vis") {
            rm(schedules.remove_systems_in_set(PostUpdate, VisibilitySystems::CheckVisibility, world, RemoveSystemsOnly));
            rm(schedules.remove_systems_in_set(PostUpdate, VisibilitySystems::MarkNewlyHiddenEntitiesInvisible, world, RemoveSystemsOnly));
        }
        if on("mesh") {
            rm(schedules.remove_systems_in_set(PostUpdate, bevy::mesh::mark_3d_meshes_as_changed_if_their_assets_changed, world, RemoveSystemsOnly));
        }
        macro_rules! material {
            ($($m:ty),*) => {$(
                if on("mat") { rm(schedules.remove_systems_in_set(PostUpdate, mat::<$m>, world, RemoveSystemsOnly)); }
                if on("spec") { rm(schedules.remove_systems_in_set(PostUpdate, spec::<$m>, world, RemoveSystemsOnly)); }
            )*};
        }
        material!(StandardMaterial, p2view::baked::BakedMaterial, p2view::outdoor::SplatMaterial, p2fx::ParticleMaterial);
        println!("gles: removed {removed} of Bevy's render bookkeeping systems from PostUpdate");
        if std::env::var_os("P2GAME_LIST_SYSTEMS").is_some() {
            for label in [PreUpdate.intern(), Update.intern(), PostUpdate.intern(), Last.intern(), First.intern()] {
                if let Some(s) = schedules.get_mut(label) {
                    let _ = s.initialize(world);
                    if let Ok(it) = s.systems() {
                        for (_, sys) in it {
                            println!("system {label:?}: {}", sys.name());
                        }
                    }
                }
            }
        }
    });
}
