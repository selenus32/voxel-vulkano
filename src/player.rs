use nalgebra as na;
use na::{
    Isometry3, 
    Matrix4,
    UnitQuaternion, 
    Vector3,
    Perspective3
};

pub struct Player {
    pub isom: Isometry3<f32>,
    pub fov: f32, // deg
    pub near: f32,
    pub far: f32,
    pub yaw: f32, // deg
    pub pitch: f32, // deg
    pub mouse_sens: f32,
    pub speed: f32,
    pub is_placing_voxel: bool,
    pub is_breaking_voxel: bool,
    pub physics_enabled: bool,
    pub velocity: Vector3<f32>,
    pub collision_offset: Vector3<f32>, // unaffected atm
    pub collision_flags: u32,
    pub is_colliding: bool,
    pub is_on_ground: bool,
    pub is_ceiling_low: bool,
}

impl Default for Player {
    fn default() -> Self {
        Self { 
            isom: Isometry3::identity(), 
            fov: 90.0, 
            near: 0.1, 
            far: 100.0, 
            yaw: -90.0, 
            pitch: 0.0, 
            mouse_sens: 0.1,
            speed: 100.0,
            is_placing_voxel: false,
            is_breaking_voxel: false,
            physics_enabled: false,
            velocity: Vector3::zeros(),
            collision_offset: Vector3::zeros(),
            collision_flags: 0,
            is_colliding: false,
            is_on_ground: false,
            is_ceiling_low: false,
        }
    }
}

impl Player {
    pub fn update_physics(&mut self, dt: f32) {
        let mut t = self.isom.translation.vector;
        if self.isom.translation.vector.y < 0.0 {
            t.y = 0.0;
        }
        if self.physics_enabled {
            if !self.is_on_ground && !self.is_colliding {
                let damping = (-dt * (std::f32::consts::LN_2 / 0.5)).exp();
                self.velocity.x *= damping;
                self.velocity.y *= damping;

                self.velocity.y -= 9.8 * dt * 8.0;
            } else {
                if self.is_colliding {
                    t += self.collision_offset;
                    self.velocity = Vector3::zeros();
                }

                self.velocity.y = 0.0;
                self.velocity.x *= 0.2;
                self.velocity.z *= 0.2;
            }
            
            t += self.velocity * dt;
        } else {
            self.velocity = Vector3::zeros();
        }
        self.isom.translation.vector = t;
    }
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