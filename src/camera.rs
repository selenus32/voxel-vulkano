use nalgebra as na;
use na::{
    Isometry3, 
    Matrix4,
    UnitQuaternion, 
    Vector3,
    Perspective3
};

pub struct Camera {
    pub isom: Isometry3<f32>,
    pub fov: f32, // deg
    pub near: f32,
    pub far: f32,
    pub yaw: f32, // deg
    pub pitch: f32, // deg
    pub mouse_sens: f32,
    pub speed: f32,
    pub first_mouse: bool
}

impl Default for Camera {
    fn default() -> Self {
        Self { 
            isom: Isometry3::identity(), 
            fov: 90.0, 
            near: 0.1, 
            far: 100.0, 
            yaw: -90.0, 
            pitch: 0.0, 
            mouse_sens: 0.1,
            speed: 200.0,
            first_mouse: true
        }
    }
}

impl Camera {
    pub fn get_view(&self) -> Matrix4<f32> {
        self.isom.inverse().to_homogeneous()
    }
    pub fn get_proj(&self, width: u32, height: u32) -> Matrix4<f32> {
        Perspective3::new(
            width as f32 / height as f32,
            self.fov.to_radians(),
            self.near,
            self.far
        ).to_homogeneous()
    }
    pub fn update_isometry(&mut self) {
        let yaw = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), pitch);
        self.isom.rotation = rotation
    }
    pub fn update_orientation(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.mouse_sens;
        self.pitch += dy * self.mouse_sens;
        self.pitch = self.pitch.clamp(-89.9,89.9);
        self.update_isometry();
    }
}