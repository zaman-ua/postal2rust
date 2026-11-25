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
use bevy::asset::AssetEvent;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::{InheritedVisibility, NoFrustumCulling, RenderLayers, VisibilityRange};
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::camera::ClearColorConfig;
use p2gles::dynamic::{Blend, CanvasBlend, CanvasDraw, Draw, ParticleDraw, Dynamic, Env, Light, MeshIn, TexIn, UV_IDENTITY};
use p2gles::renderer::{Options, Renderer};
use std::collections::HashSet;
use p2view::scene::SceneData;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::num::NonZeroU32;

pub fn enabled() -> bool {
    std::env::var_os("P2GAME_GLES").is_some_and(|v| v != "0")
}

/// GL state: context + surface exist once the window does; the renderer once a level was loaded.
pub struct Gles {
    gl: glow::Context,
    ctx: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    display: Display,
    renderer: Option<Renderer>,
    dynamic: Dynamic,
    stale_meshes: HashSet<u64>,
    stale_tex: HashSet<u64>,
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
        let gl = unsafe { glow::Context::from_loader_function_cstr(|s| display.get_proc_address(s)) };
        let dynamic = Dynamic::new(&gl);
        Ok(Gles { gl, dynamic, stale_meshes: HashSet::new(), stale_tex: HashSet::new(), ctx, surface, display, renderer: None, size, frames: 0, win_start: std::time::Instant::now(), win_frames: 0 })
    }

    /// Bakes the level's static geometry into GL buffers (replaces the previous level's).
    pub fn load_world(&mut self, data: &SceneData, skip_exports: HashSet<usize>) {
        let gl = unsafe { glow::Context::from_loader_function_cstr(|s| self.display.get_proc_address(s)) };
        let dre = self.display.get_proc_address(c"glDrawRangeElements");
        if !dre.is_null() {
            let _ = p2gles::renderer::DRAW_RANGE_ELEMENTS.set(dre as usize);
        }
        self.renderer = None; // free the old level's GL objects first
        let t = std::time::Instant::now();
        let opts = Options { chunk: 48.0, fog: None, no_cull: false, skip_exports };
        let mut r = Renderer::new(gl, data, &opts);
        r.clear = false;
        self.renderer = Some(r);
        println!("gles: world baked in {:.2} s", t.elapsed().as_secs_f32());
    }
}

/// Entities whose geometry is baked into p2gles' world buffers: the generic pass skips them.
#[derive(Resource, Default)]
pub struct GlBaked(pub HashSet<Entity>);

/// Set by `load_scene` when a level was spawned; [`classify_baked`] (right after it in the chain,
/// before anything else spawns meshes) then records which mesh entities the bake covers.
#[derive(Resource)]
pub struct ClassifyBaked {
    pub before: HashSet<Entity>,
    pub movers: HashSet<Entity>,
    /// Static mesh instances (all map actors' entities): the only ones with a distance range that are baked.
    pub statics: HashSet<Entity>,
}

pub struct GlesPlugin;

impl Plugin for GlesPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GlBaked>().add_systems(PreUpdate, init_gles).add_systems(Last, (track_assets, draw_frame).chain());
    }
}

/// Runs right after `load_scene` in the Update chain.
pub fn classify_baked(mut commands: Commands, pending: Option<Res<ClassifyBaked>>, meshes: Query<(Entity, Option<&RenderLayers>, Has<VisibilityRange>), With<Mesh3d>>, baked: Option<ResMut<GlBaked>>) {
    let (Some(p), Some(mut baked)) = (pending, baked) else { return };
    // Baked: BSP, terrain layers and static mesh instances on the default layer. Not baked: movers,
    // the sky zone (own layer) and terrain decorations (ranged entities that are no map actor).
    baked.0 = meshes
        .iter()
        .filter(|(e, l, ranged)| !p.before.contains(e) && !p.movers.contains(e) && l.is_none_or(|l| l.intersects(&RenderLayers::layer(0))) && (!ranged || p.statics.contains(e)))
        .map(|(e, _, _)| e)
        .collect();
    println!("gles: {} scene mesh entities baked into the GL world ({} mover entities stay dynamic)", baked.0.len(), p.movers.len());
    commands.remove_resource::<ClassifyBaked>();
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

fn aid<T: Asset>(id: AssetId<T>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}

/// Meshes/images changed or removed since last frame: drop their GPU copies, they re-upload on demand.
fn track_assets(mut g: Option<NonSendMut<Gles>>, mut mesh_ev: MessageReader<AssetEvent<Mesh>>, mut img_ev: MessageReader<AssetEvent<Image>>) {
    let Some(g) = g.as_mut() else { return };
    for ev in mesh_ev.read() {
        match ev {
            AssetEvent::Modified { id } | AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                g.stale_meshes.insert(aid(*id));
            }
            _ => {}
        }
    }
    for ev in img_ev.read() {
        match ev {
            AssetEvent::Modified { id } | AssetEvent::Removed { id } | AssetEvent::Unused { id } => {
                g.stale_tex.insert(aid(*id));
            }
            _ => {}
        }
    }
}

fn upload_mesh(g: &mut Gles, id: AssetId<Mesh>, mesh: &Mesh) -> bool {
    let key = aid(id);
    if g.dynamic.has_mesh(key) && !g.stale_meshes.contains(&key) {
        return true;
    }
    g.stale_meshes.remove(&key);
    let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { return false };
    let f3 = |a: bevy::mesh::MeshVertexAttribute| match mesh.attribute(a) {
        Some(VertexAttributeValues::Float32x3(v)) => Some(v.as_slice()),
        _ => None,
    };
    let f2 = |a: bevy::mesh::MeshVertexAttribute| match mesh.attribute(a) {
        Some(VertexAttributeValues::Float32x2(v)) => Some(v.as_slice()),
        _ => None,
    };
    let color = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
        Some(VertexAttributeValues::Float32x4(v)) => Some(v.as_slice()),
        _ => None,
    };
    let joints: Option<Vec<[u16; 4]>> = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) {
        Some(VertexAttributeValues::Uint16x4(v)) => Some(v.clone()),
        _ => None,
    };
    let blend_w: Option<Vec<[f32; 4]>> = match mesh.attribute(p2fx::ATTRIBUTE_BLEND) {
        Some(VertexAttributeValues::Float32(v)) => Some(v.iter().map(|&b| [b, 0.0, 0.0, 0.0]).collect()),
        _ => None,
    };
    let weights = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT) {
        Some(VertexAttributeValues::Float32x4(v)) => Some(v.as_slice()),
        _ => blend_w.as_deref(),
    };
    let indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U32(i)) => i.clone(),
        Some(Indices::U16(i)) => i.iter().map(|&v| v as u32).collect(),
        None => (0..pos.len() as u32).collect(),
    };
    let m = MeshIn { pos, normal: f3(Mesh::ATTRIBUTE_NORMAL), uv0: f2(Mesh::ATTRIBUTE_UV_0), uv1: f2(Mesh::ATTRIBUTE_UV_1), color, joints: joints.as_deref(), weights, indices: &indices };
    let gl = &g.gl as *const glow::Context;
    // SAFETY: `gl` and `dynamic` are disjoint fields of Gles.
    g.dynamic.add_mesh(unsafe { &*gl }, key, &m);
    g.dynamic.has_mesh(key)
}

fn upload_image(g: &mut Gles, id: AssetId<Image>, img: &Image) -> bool {
    use bevy::render::render_resource::TextureFormat as F;
    use p2view::scene::PixelFormat as P;
    let key = aid(id);
    if g.dynamic.has_tex(key) && !g.stale_tex.contains(&key) {
        return true;
    }
    g.stale_tex.remove(&key);
    let Some(data) = img.data.as_ref() else { return false };
    let (fmt, block) = match img.texture_descriptor.format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb => (P::Rgba8, 0usize),
        F::Bc1RgbaUnorm | F::Bc1RgbaUnormSrgb => (P::Bc1, 8),
        F::Bc2RgbaUnorm | F::Bc2RgbaUnormSrgb => (P::Bc2, 16),
        F::Bc3RgbaUnorm | F::Bc3RgbaUnormSrgb => (P::Bc3, 16),
        _ => return false,
    };
    let (w, h) = (img.width(), img.height());
    let levels = img.texture_descriptor.mip_level_count.max(1);
    let mut mips: Vec<&[u8]> = Vec::new();
    let mut off = 0usize;
    for l in 0..levels {
        let (lw, lh) = ((w >> l).max(1) as usize, (h >> l).max(1) as usize);
        let n = if block == 0 { lw * lh * 4 } else { lw.div_ceil(4) * lh.div_ceil(4) * block };
        if off + n > data.len() {
            break;
        }
        mips.push(&data[off..off + n]);
        off += n;
    }
    let repeat = match &img.sampler {
        bevy::image::ImageSampler::Descriptor(d) => !matches!(d.address_mode_u, bevy::image::ImageAddressMode::ClampToEdge),
        _ => true,
    };
    let t = TexIn { width: w, height: h, format: fmt, mips, repeat };
    let gl = &g.gl as *const glow::Context;
    // SAFETY: disjoint fields.
    g.dynamic.add_tex(unsafe { &*gl }, key, &t);
    g.dynamic.has_tex(key)
}

/// Material data the pass needs, whichever Bevy material type carried it.
struct Mat<'a> {
    base: &'a StandardMaterial,
    lightmap: Option<&'a Handle<Image>>,
}

fn srgb(c: Color) -> [f32; 4] {
    let s = Srgba::from(c);
    [s.red, s.green, s.blue, s.alpha]
}

fn draw_frame(world: &mut World) {
    if world.get_non_send_resource::<Gles>().is_none() {
        return;
    }
    let Some(win) = world.query_filtered::<&Window, With<PrimaryWindow>>().iter(world).next() else { return };
    let size = (win.physical_width().max(1), win.physical_height().max(1));
    // Cameras, lowest order first.
    struct Cam {
        order: isize,
        view: Mat4,
        pos: Vec3,
        fov: f32,
        aspect: f32,
        near: f32,
        far: f32,
        layers: RenderLayers,
        clear: Option<Color>,
        clear_color: bool,
        fog: Option<(f32, f32, [f32; 3])>,
    }
    let default_clear = world.get_resource::<ClearColor>().map(|c| c.0).unwrap_or(Color::BLACK);
    let mut cams: Vec<Cam> = Vec::new();
    {
        let mut q = world.query_filtered::<(&GlobalTransform, &Projection, &Camera, Option<&RenderLayers>, Option<&DistanceFog>), With<Camera3d>>();
        for (gt, proj, c, layers, fog) in q.iter(world) {
            let Projection::Perspective(p) = proj else { continue };
            if !c.is_active {
                continue;
            }
            let (clear_color, clear) = match &c.clear_color {
                ClearColorConfig::Default => (true, Some(default_clear)),
                ClearColorConfig::Custom(col) => (true, Some(*col)),
                ClearColorConfig::None => (false, None),
            };
            let fog = fog.and_then(|f| match f.falloff {
                FogFalloff::Linear { start, end } => {
                    let s = Srgba::from(f.color);
                    Some((start, end, [s.red, s.green, s.blue]))
                }
                _ => None,
            });
            cams.push(Cam { order: c.order, view: gt.to_matrix().inverse(), pos: gt.translation(), fov: p.fov, aspect: p.aspect_ratio, near: p.near.max(0.05), far: p.far, layers: layers.cloned().unwrap_or_default(), clear, clear_color, fog });
        }
    }
    cams.sort_by_key(|c| c.order);
    if cams.is_empty() {
        return;
    }
    // Sun / ambient for dynamic actors.
    let sun_dir = {
        let mut q = world.query::<(&DirectionalLight, &GlobalTransform)>();
        q.iter(world).next().map(|(_, gt)| Vec3::from(gt.forward())).unwrap_or(Vec3::new(0.3, -1.0, 0.2).normalize())
    };

    // Collect draws per camera pass.
    let baked: HashSet<Entity> = std::mem::take(&mut world.resource_mut::<GlBaked>().0);
    let mut gles = world.remove_non_send_resource::<Gles>().unwrap();
    {
        let g = &mut gles;
        if g.size != size {
            g.surface.resize(&g.ctx, NonZeroU32::new(size.0).unwrap(), NonZeroU32::new(size.1).unwrap());
            g.size = size;
        }
    }
    let mut passes: Vec<Vec<Draw>> = cams.iter().map(|_| Vec::new()).collect();
    let mut parts: Vec<Vec<ParticleDraw>> = cams.iter().map(|_| Vec::new()).collect();
    let mut qd = world.query::<(
        Entity,
        &Mesh3d,
        &GlobalTransform,
        &InheritedVisibility,
        Option<&RenderLayers>,
        Option<&SkinnedMesh>,
        Option<&MeshMaterial3d<StandardMaterial>>,
        Option<&MeshMaterial3d<p2view::baked::BakedMaterial>>,
        Option<&Aabb>,
        Option<&NoFrustumCulling>,
        Option<&VisibilityRange>,
    )>();
    let mut jq = world.query::<&GlobalTransform>();
    {
        let meshes = world.resource::<Assets<Mesh>>();
        let images = world.resource::<Assets<Image>>();
        let std_mats = world.resource::<Assets<StandardMaterial>>();
        let baked_mats = world.resource::<Assets<p2view::baked::BakedMaterial>>();
        let inv_bind = world.resource::<Assets<SkinnedMeshInverseBindposes>>();
        // Frustum planes per camera (world space).
        let vps: Vec<Mat4> = cams.iter().map(|c| Mat4::perspective_rh_gl(c.fov, c.aspect, c.near, c.far.max(c.near + 1.0)) * c.view).collect();
        let planes: Vec<[Vec4; 6]> = vps
            .iter()
            .map(|vp| {
                let r = |i: usize| Vec4::new(vp.x_axis[i], vp.y_axis[i], vp.z_axis[i], vp.w_axis[i]);
                let (r0, r1, r2, r3) = (r(0), r(1), r(2), r(3));
                [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2]
            })
            .collect();
        for (e, m, gt, vis, layers, skin, sm, bm, aabb, nocull, range) in qd.iter(world) {
            if !vis.get() || baked.contains(&e) {
                continue;
            }
            let Some(mesh) = meshes.get(&m.0) else { continue };
            let mat = match (bm, sm) {
                (Some(b), _) => baked_mats.get(&b.0).map(|b| Mat { base: &b.base, lightmap: Some(&b.extension.lightmap) }),
                (None, Some(s)) => std_mats.get(&s.0).map(|s| Mat { base: s, lightmap: None }),
                _ => None,
            };
            let Some(mat) = mat else { continue };
            let el = layers.cloned().unwrap_or_default();
            let model = gt.to_matrix();
            // Skinning matrices (joint world * inverse bind); skinned meshes ignore their own transform.
            let joints = match skin {
                Some(sk) => {
                    let Some(ib) = inv_bind.get(&sk.inverse_bindposes) else { continue };
                    let mut out = Vec::with_capacity(sk.joints.len());
                    for (j, ibp) in sk.joints.iter().zip(ib.iter()) {
                        let jm = jq.get(world, *j).map(|t| t.to_matrix()).unwrap_or(Mat4::IDENTITY);
                        out.push(jm * *ibp);
                    }
                    Some(out)
                }
                None => None,
            };
            let center = model.transform_point3(aabb.map(|a| Vec3::from(a.center)).unwrap_or(Vec3::ZERO));
            let radius = aabb.map(|a| Vec3::from(a.half_extents).length() * model.x_axis.truncate().length().max(model.y_axis.truncate().length()).max(model.z_axis.truncate().length())).unwrap_or(0.0);
            let cullable = nocull.is_none() && skin.is_none() && aabb.is_some();
            let base = mat.base;
            let (blend, cutoff) = match base.alpha_mode {
                AlphaMode::Opaque => (Blend::Opaque, 0.0),
                AlphaMode::Mask(c) => (Blend::Mask, c),
                AlphaMode::Add => (Blend::Add, 0.0),
                AlphaMode::Multiply => (Blend::Multiply, 0.0),
                _ => (Blend::Alpha, 0.0),
            };
            let has_color = mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_some();
            let has_uv1 = mesh.attribute(Mesh::ATTRIBUTE_UV_1).is_some();
            let lin = base.base_color.to_linear();
            let (light, color) = if blend == Blend::Multiply {
                // Modulate: base colour 4.595 linear = texture * 2 in UE2's gamma space.
                (if lin.red > 2.0 { Light::Unlit2x } else { Light::Unlit }, [1.0; 4])
            } else if mat.lightmap.is_some() && has_color {
                (Light::Vertex, srgb(base.base_color))
            } else if mat.lightmap.is_some() && has_uv1 {
                (Light::Lightmap, srgb(base.base_color))
            } else if mat.lightmap.is_some() || base.unlit {
                (Light::Unlit, srgb(base.base_color))
            } else {
                (Light::Dynamic, srgb(base.base_color))
            };
            let uv = bevy::math::Mat3::from(base.uv_transform).to_cols_array();
            for (ci, c) in cams.iter().enumerate() {
                if !c.layers.intersects(&el) {
                    continue;
                }
                let dist = (center - c.pos).length();
                if range.is_some_and(|r| dist > r.end_margin.end) {
                    continue;
                }
                if cullable {
                    if let Some((_, end, _)) = c.fog {
                        if dist - radius > end * 1.05 {
                            continue;
                        }
                    }
                    if planes[ci].iter().any(|p| p.truncate().dot(center) + p.w < -radius * p.truncate().length()) {
                        continue;
                    }
                }
                passes[ci].push(Draw {
                    mesh: aid(m.0.id()),
                    tex: base.base_color_texture.as_ref().map(|h| aid(h.id())),
                    lightmap: mat.lightmap.map(|h| aid(h.id())),
                    model,
                    joints: joints.clone(),
                    color,
                    blend,
                    cutoff,
                    two_sided: base.double_sided || base.cull_mode.is_none(),
                    light,
                    uv,
                    depth: dist,
                    overlay: false,
                });
            }
            // Upload on first use (needs `gles`, which is out of the world right now).
            upload_mesh(&mut gles, m.0.id(), mesh);
            if let Some(h) = &base.base_color_texture {
                if let Some(img) = images.get(h) {
                    upload_image(&mut gles, h.id(), img);
                }
            }
            if let Some(h) = mat.lightmap {
                if let Some(img) = images.get(h) {
                    upload_image(&mut gles, h.id(), img);
                }
            }
        }
    }
    // p2fx particle / decal batches.
    {
        let mut pq = world.query::<(&Mesh3d, &MeshMaterial3d<p2fx::ParticleMaterial>, &InheritedVisibility, Option<&RenderLayers>)>();
        let meshes = world.resource::<Assets<Mesh>>();
        let images = world.resource::<Assets<Image>>();
        let mats = world.resource::<Assets<p2fx::ParticleMaterial>>();
        for (m, mm, vis, layers) in pq.iter(world) {
            if !vis.get() {
                continue;
            }
            let (Some(mesh), Some(pm)) = (meshes.get(&m.0), mats.get(&mm.0)) else { continue };
            if mesh.count_vertices() <= 3 {
                continue; // placeholder batch
            }
            if !upload_mesh(&mut gles, m.0.id(), mesh) {
                continue;
            }
            let tex = pm.texture.as_ref().and_then(|h| images.get(h).filter(|img| upload_image(&mut gles, h.id(), img)).map(|_| aid(h.id())));
            let el = layers.cloned().unwrap_or_default();
            for (ci, c) in cams.iter().enumerate() {
                if c.layers.intersects(&el) {
                    parts[ci].push(ParticleDraw { mesh: aid(m.0.id()), tex, style: pm.key.style, z_test: pm.key.z_test, z_write: pm.key.z_write, alpha_ref: pm.params.x, alpha_test: pm.params.y > 0.5, order: pm.order });
                }
            }
        }
    }
    world.resource_mut::<GlBaked>().0 = baked;

    // HUD / menus: p2canvas batches (Mesh2d + CanvasMaterial), in painter's order (z).
    let mut canvas: Vec<CanvasDraw> = Vec::new();
    {
        use p2canvas::eval::Blend as CB;
        use p2canvas::render::CanvasMaterial;
        let mut cq = world.query::<(&Mesh2d, &MeshMaterial2d<CanvasMaterial>, &GlobalTransform, &InheritedVisibility)>();
        let meshes = world.resource::<Assets<Mesh>>();
        let images = world.resource::<Assets<Image>>();
        let mats = world.resource::<Assets<CanvasMaterial>>();
        let mut items: Vec<(f32, &Mesh2d, &CanvasMaterial)> = cq.iter(world).filter(|(_, _, _, v)| v.get()).filter_map(|(m, mm, gt, _)| mats.get(&mm.0).map(|cm| (gt.translation().z, m, cm))).collect();
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, m, cm) in items {
            let Some(mesh) = meshes.get(&m.0) else { continue };
            if !upload_mesh(&mut gles, m.0.id(), mesh) {
                continue;
            }
            let d = &cm.uniform_data;
            let mut tex = [None; 3];
            for (i, h) in [&cm.tex0, &cm.tex1, &cm.tex2].into_iter().enumerate() {
                if let Some(h) = h {
                    if let Some(img) = images.get(h) {
                        if upload_image(&mut gles, h.id(), img) {
                            tex[i] = Some(aid(h.id()));
                        }
                    }
                }
            }
            let row = |x: &p2canvas::eval::UvXform| ([x.m[0], x.m[1], x.t[0], 0.0], [x.m[2], x.m[3], x.t[1], 0.0]);
            let (u0, v0) = row(&d.xforms[0]);
            let (u1, v1) = row(&d.xforms[1]);
            let (u2, v2) = row(&d.xforms[2]);
            canvas.push(CanvasDraw {
                mesh: aid(m.0.id()),
                tex,
                tint: d.tint,
                slot_color: d.slot_colors,
                xu: [u0, u1, u2],
                xv: [v0, v1, v2],
                kinds: [d.kinds[0] as i32, d.kinds[1] as i32, d.kinds[2] as i32],
                ops: [d.ops[0] as i32, d.ops[1] as i32, d.ops[2] as i32, d.ops[3] as i32],
                modulate: d.modulate,
                alpha_test: d.alpha_test,
                blend: match cm.blend {
                    CB::Opaque => CanvasBlend::Opaque,
                    CB::Alpha => CanvasBlend::Alpha,
                    CB::Translucent => CanvasBlend::Translucent,
                    CB::Modulate => CanvasBlend::Modulate,
                    CB::AlphaModulate => CanvasBlend::AlphaModulate,
                    CB::Darken => CanvasBlend::Darken,
                    CB::Brighten => CanvasBlend::Brighten,
                    CB::Invisible => CanvasBlend::Invisible,
                },
            });
        }
    }

    // Draw.
    let g = &mut gles;
    let t_draw = std::time::Instant::now();
    let mut draws_n = 0u32;
    let mut tris_n = 0u64;
    unsafe {
        use glow::HasContext;
        g.gl.viewport(0, 0, size.0 as i32, size.1 as i32);
        g.gl.depth_mask(true);
    }
    for (ci, c) in cams.iter().enumerate() {
        unsafe {
            use glow::HasContext;
            g.gl.depth_mask(true);
            match c.clear {
                Some(col) => {
                    let s = Srgba::from(col);
                    g.gl.clear_color(s.red, s.green, s.blue, 1.0);
                    g.gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                }
                None => g.gl.clear(glow::DEPTH_BUFFER_BIT),
            }
            let _ = c.clear_color;
        }
        let far = c.far.max(c.near + 1.0);
        let vp = Mat4::perspective_rh_gl(c.fov, c.aspect, c.near, far) * c.view;
        let (fs, fe, fc) = c.fog.unwrap_or((1.0e8, 2.0e8, [0.0; 3]));
        let env = Env { fog_start: fs, fog_end: fe, fog_color: fc, ambient: [0.55; 3], sun_dir, sun_color: [0.5; 3] };
        // The baked world belongs to the default layer's camera.
        if c.layers.intersects(&RenderLayers::layer(0)) {
            if let Some(r) = g.renderer.as_mut() {
                if let Some((s, e, col)) = c.fog {
                    r.fog = (s, e);
                    r.fog_color = col;
                }
                r.render(vp, c.pos, size);
                draws_n += r.stats.draws;
                tris_n += r.stats.tris;
            }
        }
        let gl = &g.gl as *const glow::Context;
        // SAFETY: disjoint fields (gl is only read).
        g.dynamic.render(unsafe { &*gl }, vp, c.pos, &env, &mut passes[ci]);
        g.dynamic.render_particles(unsafe { &*gl }, vp, c.pos, &env, &mut parts[ci]);
        draws_n += g.dynamic.draws;
        tris_n += g.dynamic.tris;
    }
    {
        let gl = &g.gl as *const glow::Context;
        // SAFETY: disjoint fields (gl is only read).
        g.dynamic.render_canvas(unsafe { &*gl }, size, &canvas);
        draws_n += g.dynamic.draws;
    }
    // Screenshot requests (Bevy's own capture needs its renderer): read the finished frame.
    let shots: Vec<Entity> = world.query_filtered::<Entity, With<bevy::render::view::screenshot::Screenshot>>().iter(world).collect();
    if !shots.is_empty() {
        use glow::HasContext;
        let (w, h) = size;
        let mut buf = vec![0u8; (w * h * 4) as usize];
        unsafe {
            g.gl.finish();
            g.gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            g.gl.read_pixels(0, 0, w as i32, h as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::Slice(Some(&mut buf)));
        }
        let row = (w * 4) as usize;
        let mut flipped = Vec::with_capacity(buf.len());
        for y in (0..h as usize).rev() {
            flipped.extend_from_slice(&buf[y * row..(y + 1) * row]);
        }
        for e in shots {
            let image = Image::new(
                bevy::render::render_resource::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                bevy::render::render_resource::TextureDimension::D2,
                flipped.clone(),
                bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
                bevy::asset::RenderAssetUsages::default(),
            );
            world.trigger(bevy::render::view::screenshot::ScreenshotCaptured { entity: e, image });
            if let Ok(ec) = world.get_entity_mut(e) {
                ec.despawn();
            }
        }
    }
    let _ = g.surface.swap_buffers(&g.ctx);
    g.frames += 1;
    g.win_frames += 1;
    let _ = t_draw;
    if g.win_start.elapsed().as_secs() >= 5 {
        let ms = g.win_start.elapsed().as_secs_f64() * 1000.0 / g.win_frames as f64;
        println!("gles: {ms:.1} ms/frame ({:.0} fps), {} draws, {} tris", 1000.0 / ms, draws_n, tris_n);
        g.win_start = std::time::Instant::now();
        g.win_frames = 0;
    }
    // Free GPU copies of assets that no longer exist.
    if g.frames % 300 == 0 {
        let meshes: HashSet<u64> = world.resource::<Assets<Mesh>>().ids().map(aid).collect();
        let tex: HashSet<u64> = world.resource::<Assets<Image>>().ids().map(aid).collect();
        let gl = &g.gl as *const glow::Context;
        g.dynamic.retain(unsafe { &*gl }, &meshes, &tex);
    }
    world.insert_non_send_resource(gles);
}
