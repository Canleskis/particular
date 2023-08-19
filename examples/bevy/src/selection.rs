use bevy::prelude::*;

#[derive(Component, Default)]
pub struct CanSelect {
    pub radius: f32,
}

#[derive(Component, Default)]
pub struct Selected;

#[derive(Event, Deref, DerefMut)]
pub struct ClickEvent(pub Option<Entity>);

pub struct SelectionPlugin;

impl Plugin for SelectionPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ClickEvent>()
            .add_systems(Update, (entity_picker, select_clicked).chain());
    }
}

fn entity_picker(
    mut click_entity_events: EventWriter<ClickEvent>,
    mouse_input: Res<Input<MouseButton>>,
    query_window: Query<&Window>,
    query_camera: Query<(&GlobalTransform, &Camera)>,
    query_can_select: Query<(Entity, &Transform, &CanSelect)>,
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
            query_can_select
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

    click_entity_events.send(ClickEvent(clicked_entity));
}

fn select_clicked(
    mut commands: Commands,
    mut ctxs: bevy_egui::EguiContexts,
    mut selection_events: EventReader<ClickEvent>,
    query_selected: Query<Entity, With<Selected>>,
) {
    if ctxs.ctx_mut().is_pointer_over_area() {
        return;
    }

    for &ClickEvent(entity) in &mut selection_events {
        for entity in &query_selected {
            commands.entity(entity).remove::<Selected>();
        }

        if let Some(entity) = entity {
            commands.entity(entity).insert(Selected);
        }
    }
}

fn _show_pickable_zone(
    mut gizmos: Gizmos,
    query_camera: Query<&GlobalTransform, With<Camera>>,
    query_can_select: Query<(&Transform, &CanSelect)>,
) {
    let Ok(camera_transform) = query_camera.get_single() else { return };

    for (transform, selectable) in &query_can_select {
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
