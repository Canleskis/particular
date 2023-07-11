use crate::SelectableEntities;

use bevy::{input::mouse::*, prelude::*};

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (set_selected_camera_parent, camera_controls));
    }
}

#[derive(Component)]
pub struct OrbitCamera {
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        OrbitCamera { distance: 100.0 }
    }
}

fn camera_controls(
    query_windows: Query<&Window>,
    input_mouse: Res<Input<MouseButton>>,
    mut scroll_events: EventReader<MouseWheel>,
    mut motion_events: EventReader<MouseMotion>,
    mut query_camera: Query<(&mut OrbitCamera, &mut Transform)>,
    mut init: Local<bool>,
) {
    let Ok((mut orbit, mut transform)) = query_camera.get_single_mut() else { return };
    let Ok(window) = query_windows.get_single() else { return };

    let window_size = Vec2::new(window.width(), window.height());
    let scroll = scroll_events.iter().map(|ev| ev.y).sum::<f32>();
    let mouse_motion = input_mouse
        .pressed(MouseButton::Left)
        .then(|| motion_events.iter().map(|ev| ev.delta).sum::<Vec2>())
        .unwrap_or_default();

    motion_events.clear();

    let is_moving = mouse_motion.length_squared() > 0.0;
    let is_zooming = scroll.abs() > 0.0;

    if is_moving {
        let delta = mouse_motion / window_size * std::f32::consts::PI;

        transform.rotation *= Quat::from_rotation_y(-delta.x * 2.0);
        transform.rotation *= Quat::from_rotation_x(-delta.y);
    }

    if is_zooming {
        orbit.distance -= scroll * orbit.distance * 0.2;
        orbit.distance = orbit.distance.clamp(10.0, 1000.0);
    }

    if is_moving || is_zooming || !*init {
        let rotation_matrix = Mat3::from_quat(transform.rotation);
        transform.translation = rotation_matrix.mul_vec3(Vec3::new(0.0, 0.0, orbit.distance));

        *init = true;
    }
}

fn set_selected_camera_parent(
    mut commands: Commands,
    selectable_entites: Res<SelectableEntities>,
    query_camera: Query<Entity, With<Camera>>,
) {
    if selectable_entites.is_changed() {
        let Ok(camera_entity) = query_camera.get_single() else { return };
        let Some(selected_entity) = selectable_entites.selected() else { return };

        commands.entity(camera_entity).set_parent(selected_entity);
    }
}
