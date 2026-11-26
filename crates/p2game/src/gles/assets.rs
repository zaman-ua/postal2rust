//! GPU copies of Bevy's mesh and image assets: uploaded on first use, dropped when the asset
//! changes or goes away.

use super::backend::{Cmd, Texture};
use super::Gles;
use bevy::asset::AssetEvent;
use bevy::mesh::{Indices, MeshVertexAttribute, VertexAttributeValues};
use bevy::prelude::*;
use p2gles::dynamic::{pack_mesh, MeshIn};
use std::borrow::Cow;
use std::collections::HashSet;

/// What the backend holds a GPU copy of.
#[derive(Default)]
pub(super) struct Resident {
    meshes: HashSet<u64>,
    textures: HashSet<u64>,
    /// Copies to rebuild on their next use.
    stale_meshes: HashSet<u64>,
    stale_textures: HashSet<u64>,
}

/// The key p2gles knows an asset by.
pub(super) fn aid<T: Asset>(id: AssetId<T>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut h);
    h.finish()
}

fn changed<T: Asset>(events: &mut MessageReader<AssetEvent<T>>, stale: &mut HashSet<u64>) {
    for ev in events.read() {
        if let AssetEvent::Modified { id } | AssetEvent::Removed { id } | AssetEvent::Unused { id } = ev {
            stale.insert(aid(*id));
        }
    }
}

/// Meshes/images changed or removed since last frame: their GPU copies re-upload on demand.
pub(super) fn track_assets(g: Option<NonSendMut<Gles>>, mut meshes: MessageReader<AssetEvent<Mesh>>, mut images: MessageReader<AssetEvent<Image>>) {
    let Some(mut g) = g else { return };
    changed(&mut meshes, &mut g.assets.stale_meshes);
    changed(&mut images, &mut g.assets.stale_textures);
}

impl Gles {
    /// Makes sure the mesh is on the GPU; false if it cannot be drawn.
    pub(super) fn upload_mesh(&mut self, id: AssetId<Mesh>, mesh: &Mesh) -> bool {
        let key = aid(id);
        if self.assets.meshes.contains(&key) && !self.assets.stale_meshes.contains(&key) {
            return true;
        }
        self.assets.stale_meshes.remove(&key);
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else { return false };
        let f3 = |a: MeshVertexAttribute| match mesh.attribute(a) {
            Some(VertexAttributeValues::Float32x3(v)) => Some(v.as_slice()),
            _ => None,
        };
        let f2 = |a: MeshVertexAttribute| match mesh.attribute(a) {
            Some(VertexAttributeValues::Float32x2(v)) => Some(v.as_slice()),
            _ => None,
        };
        let color = match mesh.attribute(Mesh::ATTRIBUTE_COLOR) {
            Some(VertexAttributeValues::Float32x4(v)) => Some(v.as_slice()),
            _ => None,
        };
        let joints = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX) {
            Some(VertexAttributeValues::Uint16x4(v)) => Some(v.as_slice()),
            _ => None,
        };
        // Particle meshes carry their sub-image blend in the weights slot.
        let particle_blend: Option<Vec<[f32; 4]>> = match mesh.attribute(p2fx::ATTRIBUTE_BLEND) {
            Some(VertexAttributeValues::Float32(v)) => Some(v.iter().map(|&b| [b, 0.0, 0.0, 0.0]).collect()),
            _ => None,
        };
        let weights = match mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT) {
            Some(VertexAttributeValues::Float32x4(v)) => Some(v.as_slice()),
            _ => particle_blend.as_deref(),
        };
        let indices: Cow<[u32]> = match mesh.indices() {
            Some(Indices::U32(i)) => Cow::Borrowed(i),
            Some(Indices::U16(i)) => i.iter().map(|&v| v as u32).collect(),
            None => (0..pos.len() as u32).collect(),
        };
        let m = MeshIn { pos, normal: f3(Mesh::ATTRIBUTE_NORMAL), uv0: f2(Mesh::ATTRIBUTE_UV_0), uv1: f2(Mesh::ATTRIBUTE_UV_1), color, joints, weights, indices: &indices };
        match pack_mesh(&m) {
            Some(packed) => {
                self.assets.meshes.insert(key);
                self.link.send(Cmd::Mesh(key, packed));
                true
            }
            None => {
                if self.assets.meshes.remove(&key) {
                    self.link.send(Cmd::RemoveMesh(key));
                }
                false
            }
        }
    }

    /// Makes sure the image is on the GPU; false if its format or data cannot be used.
    pub(super) fn upload_image(&mut self, id: AssetId<Image>, img: &Image) -> bool {
        use bevy::render::render_resource::TextureFormat as F;
        use p2view::scene::PixelFormat as P;
        let key = aid(id);
        if self.assets.textures.contains(&key) && !self.assets.stale_textures.contains(&key) {
            return true;
        }
        self.assets.stale_textures.remove(&key);
        let Some(data) = img.data.as_ref() else { return false };
        let (format, block) = match img.texture_descriptor.format {
            F::Rgba8Unorm | F::Rgba8UnormSrgb => (P::Rgba8, 0usize),
            F::Bc1RgbaUnorm | F::Bc1RgbaUnormSrgb => (P::Bc1, 8),
            F::Bc2RgbaUnorm | F::Bc2RgbaUnormSrgb => (P::Bc2, 16),
            F::Bc3RgbaUnorm | F::Bc3RgbaUnormSrgb => (P::Bc3, 16),
            _ => return false,
        };
        let (width, height) = (img.width(), img.height());
        let mut mips: Vec<Vec<u8>> = Vec::new();
        let mut off = 0usize;
        for level in 0..img.texture_descriptor.mip_level_count.max(1) {
            let (w, h) = ((width >> level).max(1) as usize, (height >> level).max(1) as usize);
            let len = if block == 0 { w * h * 4 } else { w.div_ceil(4) * h.div_ceil(4) * block };
            let Some(mip) = data.get(off..off + len) else { break };
            mips.push(mip.to_vec());
            off += len;
        }
        let repeat = match &img.sampler {
            bevy::image::ImageSampler::Descriptor(d) => !matches!(d.address_mode_u, bevy::image::ImageAddressMode::ClampToEdge),
            _ => true,
        };
        if mips.is_empty() || width == 0 || height == 0 {
            return false;
        }
        self.assets.textures.insert(key);
        self.link.send(Cmd::Texture(key, Texture { width, height, format, mips, repeat }));
        true
    }

    /// Frees GPU copies of assets that no longer exist.
    pub(super) fn drop_dead_assets(&mut self, meshes: &Assets<Mesh>, images: &Assets<Image>) {
        let meshes: HashSet<u64> = meshes.ids().map(aid).collect();
        let images: HashSet<u64> = images.ids().map(aid).collect();
        self.assets.meshes.retain(|k| meshes.contains(k));
        self.assets.textures.retain(|k| images.contains(k));
        self.link.send(Cmd::Retain(meshes, images));
    }
}
