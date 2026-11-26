//! Reads one frame out of the ECS into p2gles draw lists.

use super::assets::aid;
use super::bake::GlStatic;
use super::view::{fog_end, View};
use super::Gles;
use bevy::camera::primitives::Aabb;
use bevy::camera::visibility::{InheritedVisibility, NoFrustumCulling, RenderLayers, VisibilityRange};
use bevy::camera::{Camera3d, ClearColorConfig};
use bevy::ecs::system::SystemParam;
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::prelude::*;
use bevy::sprite_render::MeshMaterial2d;
use p2canvas::render::CanvasMaterial;
use p2gles::dynamic::{Blend, CanvasBlend, CanvasDraw, Draw, Env, Light, ParticleDraw};
use std::collections::HashMap;
use std::sync::Arc;

/// An active 3D camera.
pub(super) struct Cam {
    order: isize,
    pub view_proj: Mat4,
    pub pos: Vec3,
    layers: RenderLayers,
    /// Colour to clear to; `None` keeps the previous camera's picture (depth is always cleared).
    pub clear: Option<Color>,
    pub fog: Option<(f32, f32, [f32; 3])>,
    view: View,
}

impl Cam {
    /// The baked world belongs to the default layer's camera.
    pub fn draws_world(&self) -> bool {
        self.layers.intersects(&RenderLayers::layer(0))
    }
}

/// What one camera draws besides the baked world.
#[derive(Default)]
pub(super) struct Pass {
    pub meshes: Vec<Draw>,
    pub particles: Vec<ParticleDraw>,
}

#[derive(SystemParam)]
pub(super) struct Cameras<'w, 's> {
    cameras: Query<'w, 's, (&'static GlobalTransform, &'static Projection, &'static Camera, Option<&'static RenderLayers>, Option<&'static DistanceFog>), With<Camera3d>>,
    clear: Option<Res<'w, ClearColor>>,
    suns: Query<'w, 's, (&'static DirectionalLight, &'static GlobalTransform)>,
    ambient: Option<Res<'w, GlobalAmbientLight>>,
}

/// Sun and ambient light for dynamic actors, as linear light.
pub(super) struct Lighting {
    ambient: [f32; 3],
    sun_dir: Vec3,
    sun_color: [f32; 3],
}

impl Lighting {
    pub fn env(&self, fog: Option<(f32, f32, [f32; 3])>) -> Env {
        let (fog_start, fog_end, fog_color) = fog.unwrap_or((1.0e8, 2.0e8, [0.0; 3]));
        Env { fog_start, fog_end, fog_color, ambient: self.ambient, sun_dir: self.sun_dir, sun_color: self.sun_color }
    }
}

fn scaled(color: Color, k: f32) -> [f32; 3] {
    let c = color.to_linear();
    [c.red * k, c.green * k, c.blue * k]
}

fn srgb(color: Color) -> [f32; 4] {
    let s = Srgba::from(color);
    [s.red, s.green, s.blue, s.alpha]
}

impl Cameras<'_, '_> {
    /// Active perspective cameras, lowest order first.
    pub fn active(&self) -> Vec<Cam> {
        let default_clear = self.clear.as_ref().map(|c| c.0).unwrap_or(Color::BLACK);
        let mut cams: Vec<Cam> = self
            .cameras
            .iter()
            .filter_map(|(transform, projection, camera, layers, fog)| {
                let Projection::Perspective(p) = projection else { return None };
                if !camera.is_active {
                    return None;
                }
                let view_proj = View::projection(p, transform.to_matrix());
                let view = View::new(view_proj, transform.translation(), fog_end(fog));
                let clear = match camera.clear_color {
                    ClearColorConfig::Default => Some(default_clear),
                    ClearColorConfig::Custom(color) => Some(color),
                    ClearColorConfig::None => None,
                };
                let fog = fog.and_then(|f| match f.falloff {
                    FogFalloff::Linear { start, end } => {
                        let [r, g, b, _] = srgb(f.color);
                        Some((start, end, [r, g, b]))
                    }
                    _ => None,
                });
                Some(Cam { order: camera.order, view_proj, pos: transform.translation(), layers: layers.cloned().unwrap_or_default(), clear, fog, view })
            })
            .collect();
        cams.sort_by_key(|c| c.order);
        cams
    }

    /// The Bevy scene is calibrated for camera exposure EV100 9.7 / 1.2 (spawn_lights):
    /// linear = lux * exposure (/ pi for the sun's Lambert term).
    pub fn lighting(&self) -> Lighting {
        let exposure = 2f32.powf(-9.7) / 1.2;
        let (sun_dir, sun_color) = match self.suns.iter().max_by(|a, b| a.0.illuminance.total_cmp(&b.0.illuminance)) {
            Some((light, transform)) => (Vec3::from(transform.forward()), scaled(light.color, light.illuminance * exposure / std::f32::consts::PI)),
            None => (Vec3::new(0.3, -1.0, 0.2).normalize(), [0.2; 3]),
        };
        let ambient = self.ambient.as_ref().map(|a| scaled(a.color, a.brightness * exposure)).unwrap_or([0.3; 3]);
        Lighting { ambient, sun_dir, sun_color }
    }
}

#[derive(SystemParam)]
pub(super) struct SceneMeshes<'w, 's> {
    /// Mesh entities outside the baked world.
    entities: Query<
        'w,
        's,
        (
            Entity,
            &'static Mesh3d,
            &'static GlobalTransform,
            &'static InheritedVisibility,
            Option<&'static RenderLayers>,
            Option<&'static SkinnedMesh>,
            Option<&'static MeshMaterial3d<StandardMaterial>>,
            Option<&'static MeshMaterial3d<p2view::baked::BakedMaterial>>,
            Option<&'static Aabb>,
            Has<NoFrustumCulling>,
            Option<&'static VisibilityRange>,
        ),
        Without<GlStatic>,
    >,
    particles: Query<'w, 's, (&'static Mesh3d, &'static MeshMaterial3d<p2fx::ParticleMaterial>, &'static InheritedVisibility, Option<&'static RenderLayers>)>,
    joints: Query<'w, 's, &'static GlobalTransform>,
    materials: Res<'w, Assets<StandardMaterial>>,
    baked_materials: Res<'w, Assets<p2view::baked::BakedMaterial>>,
    particle_materials: Res<'w, Assets<p2fx::ParticleMaterial>>,
    inverse_bindposes: Res<'w, Assets<SkinnedMeshInverseBindposes>>,
}

/// Skinning matrices of one frame, keyed by (root joint, joint count): a character's mesh sections
/// share one skeleton.
type Skeletons = HashMap<(Entity, usize), Arc<Vec<Mat4>>>;

/// Material data the pass needs, whichever Bevy material type carried it.
struct Material<'a> {
    base: &'a StandardMaterial,
    lightmap: Option<&'a Handle<Image>>,
}

/// How p2gles should light and blend a mesh.
fn shading(mat: &Material, mesh: &Mesh) -> (Blend, f32, Light, [f32; 4]) {
    let base = mat.base;
    let (blend, cutoff) = match base.alpha_mode {
        AlphaMode::Opaque => (Blend::Opaque, 0.0),
        AlphaMode::Mask(c) => (Blend::Mask, c),
        AlphaMode::Add => (Blend::Add, 0.0),
        AlphaMode::Multiply => (Blend::Multiply, 0.0),
        _ => (Blend::Alpha, 0.0),
    };
    if blend == Blend::Multiply {
        // Modulate: base colour 4.595 linear = texture * 2 in UE2's gamma space.
        let light = if base.base_color.to_linear().red > 2.0 { Light::Unlit2x } else { Light::Unlit };
        return (blend, cutoff, light, [1.0; 4]);
    }
    let has = |a| mesh.attribute(a).is_some();
    let light = if mat.lightmap.is_some() && has(Mesh::ATTRIBUTE_COLOR) {
        Light::Vertex
    } else if mat.lightmap.is_some() && has(Mesh::ATTRIBUTE_UV_1) {
        Light::Lightmap
    } else if mat.lightmap.is_some() || base.unlit {
        Light::Unlit
    } else {
        Light::Dynamic
    };
    (blend, cutoff, light, srgb(base.base_color))
}

impl SceneMeshes<'_, '_> {
    /// Adds every visible mesh entity outside the baked world to the passes of the cameras that see
    /// it, uploading what they need.
    pub fn collect_meshes(&self, cams: &[Cam], gles: &mut Gles, meshes: &Assets<Mesh>, images: &Assets<Image>, passes: &mut [Pass]) {
        let mut skeletons = Skeletons::new();
        for (entity, mesh3d, transform, visible, layers, skin, standard, baked_material, aabb, no_culling, range) in &self.entities {
            if !visible.get() {
                continue;
            }
            let Some(mesh) = meshes.get(&mesh3d.0) else { continue };
            let mat = match (baked_material, standard) {
                (Some(b), _) => self.baked_materials.get(&b.0).map(|b| Material { base: &b.base, lightmap: Some(&b.extension.lightmap) }),
                (None, Some(s)) => self.materials.get(&s.0).map(|s| Material { base: s, lightmap: None }),
                _ => None,
            };
            let Some(mat) = mat else { continue };
            let layers = layers.cloned().unwrap_or_default();
            let model = transform.to_matrix();
            // Skinned meshes ignore their own transform.
            let joints = match skin {
                Some(skin) => match self.skeleton(skin, &mut skeletons) {
                    Some(joints) => Some(joints),
                    None => continue,
                },
                None => None,
            };
            // Culling sphere. Rough bounds for characters: 3 m around the skeleton root (ragdolls
            // stay close).
            let root = skin.and_then(|s| s.joints.first()).and_then(|&j| self.joints.get(j).ok());
            let (center, radius, cullable) = match (root, aabb) {
                // The skeleton's bounds replace `NoFrustumCulling`, which rigs carry because Bevy
                // cannot bound a skinned mesh.
                (Some(root), _) => (root.translation(), 3.0, true),
                (None, aabb) => {
                    let scale = model.x_axis.truncate().length().max(model.y_axis.truncate().length()).max(model.z_axis.truncate().length());
                    let center = model.transform_point3(aabb.map(|a| Vec3::from(a.center)).unwrap_or(Vec3::ZERO));
                    let radius = aabb.map(|a| Vec3::from(a.half_extents).length() * scale).unwrap_or(0.0);
                    (center, radius, !no_culling && aabb.is_some() && skin.is_none())
                }
            };
            let (blend, cutoff, light, color) = shading(&mat, mesh);
            let base = mat.base;
            let uv = bevy::math::Mat3::from(base.uv_transform).to_cols_array();
            let mut seen = false;
            for (cam, pass) in cams.iter().zip(passes.iter_mut()) {
                if !cam.layers.intersects(&layers) {
                    continue;
                }
                let dist = (center - cam.pos).length();
                if range.is_some_and(|r| dist > r.end_margin.end) || (cullable && !cam.view.sees(center, radius)) {
                    continue;
                }
                seen = true;
                pass.meshes.push(Draw {
                    mesh: aid(mesh3d.0.id()),
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
                    // Bevy sorts translucent draws by distance + depth_bias (bigger = drawn later).
                    depth: dist - base.depth_bias,
                    overlay: false,
                });
            }
            if !seen {
                continue;
            }
            gles.upload_mesh(mesh3d.0.id(), mesh);
            for handle in base.base_color_texture.iter().chain(mat.lightmap) {
                if let Some(img) = images.get(handle) {
                    gles.upload_image(handle.id(), img);
                }
            }
        }
    }

    /// Skinning matrices (joint world * inverse bind), computed once per skeleton and frame.
    fn skeleton(&self, skin: &SkinnedMesh, cache: &mut Skeletons) -> Option<Arc<Vec<Mat4>>> {
        let key = (*skin.joints.first()?, skin.joints.len());
        if let Some(matrices) = cache.get(&key) {
            return Some(matrices.clone());
        }
        let inverse = self.inverse_bindposes.get(&skin.inverse_bindposes)?;
        let matrices: Arc<Vec<Mat4>> = Arc::new(skin.joints.iter().zip(inverse.iter()).map(|(&joint, bind)| self.joints.get(joint).map(|t| t.to_matrix()).unwrap_or(Mat4::IDENTITY) * *bind).collect());
        cache.insert(key, matrices.clone());
        Some(matrices)
    }

    /// p2fx particle and decal batches.
    pub fn collect_particles(&self, cams: &[Cam], gles: &mut Gles, meshes: &Assets<Mesh>, images: &Assets<Image>, passes: &mut [Pass]) {
        for (mesh3d, material, visible, layers) in &self.particles {
            if !visible.get() {
                continue;
            }
            let (Some(mesh), Some(pm)) = (meshes.get(&mesh3d.0), self.particle_materials.get(&material.0)) else { continue };
            if mesh.count_vertices() <= 3 {
                continue; // placeholder batch
            }
            if !gles.upload_mesh(mesh3d.0.id(), mesh) {
                continue;
            }
            let tex = pm.texture.as_ref().and_then(|h| images.get(h).filter(|img| gles.upload_image(h.id(), img)).map(|_| aid(h.id())));
            let layers = layers.cloned().unwrap_or_default();
            for (cam, pass) in cams.iter().zip(passes.iter_mut()) {
                if cam.layers.intersects(&layers) {
                    pass.particles.push(ParticleDraw { mesh: aid(mesh3d.0.id()), tex, style: pm.key.style, z_test: pm.key.z_test, z_write: pm.key.z_write, alpha_ref: pm.params.x, alpha_test: pm.params.y > 0.5, order: pm.order });
                }
            }
        }
    }
}

/// HUD and menus: p2canvas batches.
#[derive(SystemParam)]
pub(super) struct Hud<'w, 's> {
    batches: Query<'w, 's, (&'static Mesh2d, &'static MeshMaterial2d<CanvasMaterial>, &'static GlobalTransform, &'static InheritedVisibility)>,
    materials: Res<'w, Assets<CanvasMaterial>>,
    canvas: Res<'w, p2canvas::HudCanvas>,
}

impl Hud<'_, '_> {
    /// The size of the canvas the batches are laid out on.
    pub fn size(&self) -> (u32, u32) {
        (self.canvas.0.size_x.max(1) as u32, self.canvas.0.size_y.max(1) as u32)
    }

    /// The batches in painter's order (z).
    pub fn collect(&self, gles: &mut Gles, meshes: &Assets<Mesh>, images: &Assets<Image>) -> Vec<CanvasDraw> {
        let mut batches: Vec<(f32, &Mesh2d, &CanvasMaterial)> = self.batches.iter().filter(|(.., visible)| visible.get()).filter_map(|(mesh, material, transform, _)| self.materials.get(&material.0).map(|m| (transform.translation().z, mesh, m))).collect();
        batches.sort_by(|a, b| a.0.total_cmp(&b.0));
        batches
            .into_iter()
            .filter_map(|(_, mesh2d, material)| {
                let mesh = meshes.get(&mesh2d.0)?;
                if !gles.upload_mesh(mesh2d.0.id(), mesh) {
                    return None;
                }
                let textures = [&material.tex0, &material.tex1, &material.tex2].map(|handle| {
                    let handle = handle.as_ref()?;
                    gles.upload_image(handle.id(), images.get(handle)?).then(|| aid(handle.id()))
                });
                Some(canvas_draw(aid(mesh2d.0.id()), textures, material))
            })
            .collect()
    }
}

fn canvas_draw(mesh: u64, tex: [Option<u64>; 3], material: &CanvasMaterial) -> CanvasDraw {
    use p2canvas::eval::Blend as B;
    let d = &material.uniform_data;
    CanvasDraw {
        mesh,
        tex,
        tint: d.tint,
        slot_color: d.slot_colors,
        xu: d.xforms.map(|x| [x.m[0], x.m[1], x.t[0], 0.0]),
        xv: d.xforms.map(|x| [x.m[2], x.m[3], x.t[1], 0.0]),
        kinds: d.kinds.map(|k| k as i32),
        ops: d.ops.map(|o| o as i32),
        modulate: d.modulate,
        alpha_test: d.alpha_test,
        blend: match material.blend {
            B::Opaque => CanvasBlend::Opaque,
            B::Alpha => CanvasBlend::Alpha,
            B::Translucent => CanvasBlend::Translucent,
            B::Modulate => CanvasBlend::Modulate,
            B::AlphaModulate => CanvasBlend::AlphaModulate,
            B::Darken => CanvasBlend::Darken,
            B::Brighten => CanvasBlend::Brighten,
            B::Invisible => CanvasBlend::Invisible,
        },
    }
}
