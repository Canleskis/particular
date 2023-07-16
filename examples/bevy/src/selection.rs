use crate::OrbitCamera;

use bevy::prelude::*;

#[derive(Component, Default)]
pub struct Selectable {
    pub radius: f32,
    pub min_camera_distance: f32,
    pub saved_transform: Transform,
}

#[derive(Component)]
pub struct Selected;

#[derive(Event)]
pub struct ClickEntityEvent(pub Entity);

pub struct SelectionPlugin;

impl Plugin for SelectionPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ClickEntityEvent>()
            .add_systems(PreUpdate, (select_clicked, entity_picker).chain())
            .add_systems(
                Update,
                (deselected_save_transform, selected_make_camera_follow).chain(),
            );
    }
}

fn entity_picker(
    mut click_entity_events: EventWriter<ClickEntityEvent>,
    mouse_input: Res<Input<MouseButton>>,
    query_window: Query<&Window>,
    query_camera: Query<(&GlobalTransform, &Camera)>,
    query_selectable: Query<(Entity, &Transform, &Selectable)>,
) {
    if !mouse_input.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = query_window.get_single() else { return };
    let Ok((camera_transform, camera)) = query_camera.get_single() else { return };

    let clicked_entity = window
        .cursor_position()
        .and_then(|position| camera.viewport_to_world(camera_transform, position))
        .and_then(|ray| {
            query_selectable
                .iter()
                .fold(None, |acc, (entity, transform, selectable)| {
                    let distance = transform.translation - ray.origin;
                    let proj = ray.direction.dot(distance);
                    let mag = distance.length_squared();
                    let radius = selectable.radius + mag.sqrt() / 150.0;
                    let d = (proj * proj + radius * radius >= mag).then_some((entity, mag));

                    acc.filter(|&(_, mag2)| d.is_none() || mag2 < mag).or(d)
                })
                .map(|(entity, _)| entity)
        });

    if let Some(clicked_entity) = clicked_entity {
        click_entity_events.send(ClickEntityEvent(clicked_entity));
    }
}

fn _show_pickable_zone(
    mut gizmos: Gizmos,
    query_camera: Query<&GlobalTransform, With<Camera>>,
    query_selectable: Query<(&Transform, &Selectable)>,
) {
    let Ok(camera_transform) = query_camera.get_single() else { return };

    for (transform, selectable) in &query_selectable {
        let distance = transform.translation - camera_transform.translation();
        let radius = selectable.radius + distance.length() / 200.0;
        gizmos.circle(
            transform.translation,
            distance.normalize(),
            radius,
            Color::WHITE,
        );
    }
}

fn select_clicked(
    mut commands: Commands,
    mut selection_events: EventReader<ClickEntityEvent>,
    query_selected: Query<Entity, With<Selected>>,
) {
    let currently_selected = query_selected.get_single();

    for &ClickEntityEvent(entity) in selection_events.iter() {
        if let Ok(currently_selected) = currently_selected {
            commands.entity(currently_selected).remove::<Selected>();
        }
        commands.entity(entity).insert(Selected);
    }
}

fn deselected_save_transform(
    mut deselected: RemovedComponents<Selected>,
    mut query_camera: Query<&Transform, With<Camera>>,
    mut query_selected: Query<&mut Selectable>,
) {
    let Ok(camera_transform) = query_camera.get_single_mut() else { return };

    for entity in deselected.iter() {
        if let Ok(mut selectable) = query_selected.get_mut(entity) {
            selectable.saved_transform = *camera_transform;
        }
    }
}

fn selected_make_camera_follow(
    mut commands: Commands,
    mut query_camera: Query<(Entity, &mut Transform, &mut OrbitCamera), With<Camera>>,
    mut query_selected: Query<(Entity, &Selectable), Added<Selected>>,
) {
    let Ok((camera_entity, mut camera_transform, mut orbit)) = query_camera.get_single_mut() else { return };
    let Ok((selected_entity, selectable)) = query_selected.get_single_mut() else { return };

    commands.entity(camera_entity).set_parent(selected_entity);

    *camera_transform = selectable.saved_transform;
    orbit.min_distance = selectable.min_camera_distance;
}
