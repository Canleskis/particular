use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
};

#[derive(Resource, Deref, DerefMut, Default)]
pub struct Followed(pub Option<Entity>);

#[derive(Component, Default)]
pub struct CanFollow {
    pub min_camera_distance: f32,
    pub saved_transform: Transform,
}

#[derive(Component, Default)]
pub struct OrbitCamera {
    pub min_distance: f32,
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Followed>()
            .add_systems(
                Update,
                (followed_save_transform, followed_set_parent_camera).chain(),
            )
            .add_systems(PostUpdate, camera_controls);
    }
}

pub fn camera_controls(
    query_windows: Query<&Window>,
    input_mouse: Res<Input<MouseButton>>,
    mut scroll_events: EventReader<MouseWheel>,
    mut motion_events: EventReader<MouseMotion>,
    mut query_camera: Query<(&mut Transform, &OrbitCamera, Ref<Parent>)>,
) {
    let Ok((mut transform, orbit, parent)) = query_camera.get_single_mut() else {
        return;
    };
    let Ok(window) = query_windows.get_single() else {
        return;
    };

    let window_size = Vec2::new(window.width(), window.height());

    let scroll = scroll_events
        .iter()
        .map(|ev| match ev.unit {
            bevy::input::mouse::MouseScrollUnit::Pixel => ev.y * 0.005,
            bevy::input::mouse::MouseScrollUnit::Line => ev.y * 1.0,
        })
        .sum::<f32>();

    let motion = input_mouse
        .pressed(MouseButton::Right)
        .then(|| motion_events.iter().map(|ev| ev.delta).sum::<Vec2>())
        .unwrap_or_default();

    motion_events.clear();

    let is_moving = motion.length_squared() != 0.0;
    let is_zooming = scroll != 0.0;

    let delta = motion / window_size * std::f32::consts::PI;
    let mut rotation = transform.rotation;
    rotation *= Quat::from_rotation_y(-delta.x * 2.0);
    rotation *= Quat::from_rotation_x(-delta.y);

    let mut distance = transform.translation.length();
    distance -= distance * scroll * 0.2;

    if is_moving || is_zooming || parent.is_changed() {
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

fn followed_save_transform(
    followed: Res<Followed>,
    mut query_can_follow: Query<&mut CanFollow>,
    query_camera: Query<&Transform, With<Camera>>,
    mut previous_followed: Local<Option<Entity>>,
) {
    if followed.is_changed() {
        let can_follow = previous_followed.and_then(|e| query_can_follow.get_mut(e).ok());
        let Ok(&transform) = query_camera.get_single() else {
            return;
        };

        if let Some(mut can_follow) = can_follow {
            can_follow.saved_transform = transform;
        }
    }
    *previous_followed = **followed;
}

fn followed_set_parent_camera(
    mut commands: Commands,
    followed: Res<Followed>,
    mut query_camera: Query<(Entity, &mut Transform, &mut OrbitCamera), With<Camera>>,
    query_can_follow: Query<&CanFollow>,
) {
    if !followed.is_changed() {
        return;
    }

    let Some(followed) = **followed else { return };
    let Ok((camera_entity, mut transform, mut orbit)) = query_camera.get_single_mut() else {
        return;
    };
    let Ok(selectable) = query_can_follow.get(followed) else {
        return;
    };

    commands.entity(camera_entity).set_parent(followed);
    orbit.min_distance = selectable.min_camera_distance;
    *transform = selectable.saved_transform;
}
