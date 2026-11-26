//! What a camera can see: shared by the draw-list collection and by posing, which skips characters
//! nobody would draw.

use bevy::prelude::*;

/// A camera's frustum, cut short where its fog ends.
#[derive(Clone, Copy)]
pub(crate) struct View {
    pub pos: Vec3,
    /// World-space planes, normals pointing inwards.
    planes: [Vec4; 6],
    fog_end: Option<f32>,
}

impl View {
    pub fn new(view_proj: Mat4, pos: Vec3, fog_end: Option<f32>) -> View {
        let row = |i: usize| Vec4::new(view_proj.x_axis[i], view_proj.y_axis[i], view_proj.z_axis[i], view_proj.w_axis[i]);
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        View { pos, planes: [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2], fog_end }
    }

    /// The view projection of a perspective camera as p2gles draws it.
    pub fn projection(p: &PerspectiveProjection, camera: Mat4) -> Mat4 {
        let near = p.near.max(0.05);
        Mat4::perspective_rh_gl(p.fov, p.aspect_ratio, near, p.far.max(near + 1.0)) * camera.inverse()
    }

    /// Whether any of a bounding sphere is inside the frustum and nearer than the end of the fog.
    pub fn sees(&self, center: Vec3, radius: f32) -> bool {
        self.fog_end.is_none_or(|end| self.pos.distance(center) - radius <= end * 1.05) && self.planes.iter().all(|p| p.truncate().dot(center) + p.w >= -radius * p.truncate().length())
    }
}

pub(crate) fn fog_end(fog: Option<&DistanceFog>) -> Option<f32> {
    match fog?.falloff {
        FogFalloff::Linear { end, .. } => Some(end),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sees_what_is_ahead_and_nearer_than_the_fog() {
        let camera = PerspectiveProjection { fov: 90f32.to_radians(), aspect_ratio: 1.0, near: 0.1, far: 1000.0, ..default() };
        // At the origin, looking down -Z.
        let view = View::new(View::projection(&camera, Mat4::IDENTITY), Vec3::ZERO, Some(100.0));
        assert!(view.sees(Vec3::new(0.0, 0.0, -10.0), 1.0));
        assert!(!view.sees(Vec3::new(0.0, 0.0, 10.0), 1.0), "behind");
        assert!(!view.sees(Vec3::new(30.0, 0.0, -10.0), 1.0), "to the side");
        assert!(view.sees(Vec3::new(11.0, 0.0, -10.0), 2.0), "straddles the side plane");
        assert!(!view.sees(Vec3::new(0.0, 0.0, -200.0), 1.0), "beyond the fog");
        assert!(view.sees(Vec3::new(0.0, 0.0, -106.0), 2.0), "reaches into the fog");
    }
}
