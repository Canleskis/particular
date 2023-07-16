use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

#[derive(Component, Default)]
pub struct OrbitCamera {
    pub min_distance: f32,
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PostUpdate, camera_controls);
    }
}

pub fn camera_controls(
    query_windows: Query<&Window>,
    input_mouse: Res<Input<MouseButton>>,
    mut scroll_events: EventReader<MouseWheel>,
    mut motion_events: EventReader<MouseMotion>,
    mut query_camera: Query<(&mut Transform, &OrbitCamera)>,
) {
    let Ok((mut transform, orbit)) = query_camera.get_single_mut() else { return };
    let Ok(window) = query_windows.get_single() else { return };

    let window_size = Vec2::new(window.width(), window.height());
    let scroll = scroll_events.iter().map(|ev| ev.y).sum::<f32>();
    let mouse_motion = input_mouse
        .pressed(MouseButton::Right)
        .then(|| motion_events.iter().map(|ev| ev.delta).sum::<Vec2>())
        .unwrap_or_default();

    motion_events.clear();

    let is_moving = mouse_motion.length_squared() != 0.0;
    let is_zooming = scroll != 0.0;

    let mut rotation = transform.rotation;
    if is_moving {
        let input = mouse_motion / window_size * std::f32::consts::PI;

        rotation *= Quat::from_rotation_y(-input.x * 2.0);
        rotation *= Quat::from_rotation_x(-input.y);
    }

    let mut distance = transform.translation.length();
    if is_zooming {
        distance -= distance * scroll * 0.2;
    }

    if is_moving || is_zooming || transform.is_changed() {
        *transform = Transform {
            translation: Mat3::from_quat(rotation).mul_vec3(Vec3::new(
                0.0,
                0.0,
                distance.clamp(orbit.min_distance, 1000.0),
            )),
            rotation,
            ..*transform
        }
    }
}
