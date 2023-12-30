use crate::camera::OrbitCamera;
use bevy::prelude::*;

#[derive(Component, Default)]
pub struct Clickable {
    pub radius: f32,
}

#[derive(Event, Deref, DerefMut)]
pub struct ClickEvent(pub Option<Entity>);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct Followed(pub Option<Entity>);

#[derive(Component, Default)]
pub struct CanFollow {
    pub min_camera_distance: f32,
}

pub struct SelectionPlugin;

impl Plugin for SelectionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Followed>()
            .add_event::<ClickEvent>()
            .add_systems(Update, (entity_picker, follow_clicked).chain())
            .add_systems(PostUpdate, sync_followed.before(crate::camera_controls));
    }
}

fn entity_picker(
    mut click_entity_events: EventWriter<ClickEvent>,
    mouse_input: Res<Input<MouseButton>>,
    query_window: Query<&Window>,
    query_camera: Query<(&GlobalTransform, &Camera)>,
    query_can_select: Query<(Entity, &Transform, &Clickable)>,
) {
    if !mouse_input.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = query_window.get_single() else {
        return;
    };
    let Ok((camera_transform, camera)) = query_camera.get_single() else {
        return;
    };

    let clicked_entity = window
        .cursor_position()
        .and_then(|position| camera.viewport_to_world(camera_transform, position))
        .and_then(|ray| {
            query_can_select
                .iter()
                .fold(None, |acc, (entity, transform, clickable)| {
                    let distance = transform.translation - ray.origin;
                    let proj = ray.direction.dot(distance);
                    let mag = distance.length_squared();
                    let radius = clickable.radius + mag.sqrt() / 150.0;
                    let d = (proj * proj + radius * radius >= mag).then_some((entity, mag));

                    acc.filter(|&(_, mag2)| d.is_none() || mag2 < mag).or(d)
                })
                .map(|(entity, _)| entity)
        });

    click_entity_events.send(ClickEvent(clicked_entity));
}

fn follow_clicked(mut click_events: EventReader<ClickEvent>, mut followed: ResMut<Followed>) {
    for &ClickEvent(entity) in click_events.read() {
        if let Some(entity) = entity {
            followed.replace(entity);
        }
    }
}

fn sync_followed(
    followed: Res<Followed>,
    query_can_follow: Query<(&Transform, &CanFollow)>,
    mut query_camera: Query<&mut OrbitCamera>,
) {
    let Some((followed_transform, can_follow)) =
        followed.and_then(|e| query_can_follow.get(e).ok())
    else {
        return;
    };

    for mut orbit in &mut query_camera {
        orbit.min_distance = can_follow.min_camera_distance;
        orbit.focus = followed_transform.translation;
    }
}

fn _show_pickable_zone(
    mut gizmos: Gizmos,
    query_camera: Query<&GlobalTransform, With<Camera>>,
    query_can_select: Query<(&Transform, &Clickable)>,
) {
    let Ok(camera_transform) = query_camera.get_single() else {
        return;
    };

    for (transform, can_select) in &query_can_select {
        let distance = transform.translation - camera_transform.translation();
        let mag = distance.length_squared();
        let radius = can_select.radius + mag.sqrt() / 150.0;
        gizmos.circle(
            transform.translation,
            distance.normalize(),
            radius,
            Color::WHITE,
        );
    }
}
