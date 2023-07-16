use crate::{PhysicsSettings, PredictionDuration};

use bevy::prelude::*;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui)
            // .add_systems(PostStartup, setup_labels)
            .add_systems(
                Last,
                (
                    update_time_scale_ui,
                    update_prediction_duration_ui,
                    update_position_labels,
                ),
            );
    }
}

#[derive(Component)]
struct UiTimeScale;

#[derive(Component)]
struct UiPredictionDuration;

fn setup_ui(mut commands: Commands) {
    let style = TextStyle {
        font_size: 20.0,
        color: Color::GRAY,
        ..default()
    };

    commands
        .spawn(NodeBundle {
            style: Style {
                flex_direction: FlexDirection::Column,
                margin: UiRect {
                    top: Val::Px(5.0),
                    left: Val::Px(5.0),
                    ..default()
                },
                row_gap: Val::Px(5.0),
                ..default()
            },
            ..default()
        })
        .with_children(|node| {
            node.spawn((
                TextBundle::from_sections([
                    TextSection::new("Time scale: ", style.clone()),
                    TextSection::from_style(style.clone()),
                ]),
                UiTimeScale,
            ));

            node.spawn((
                TextBundle::from_sections([
                    TextSection::new("Prediction duration: ", style.clone()),
                    TextSection::from_style(style.clone()),
                ]),
                UiPredictionDuration,
            ));
        });
}

fn update_time_scale_ui(
    physics: Res<PhysicsSettings>,
    mut query_ui: Query<&mut Text, With<UiTimeScale>>,
) {
    if physics.is_changed() {
        for mut text in &mut query_ui {
            text.sections[1].value = format!("{:.1}", physics.time_scale);
        }
    }
}

fn update_prediction_duration_ui(
    prediction_duration: Res<PredictionDuration>,
    mut query_ui: Query<&mut Text, With<UiPredictionDuration>>,
) {
    if prediction_duration.is_changed() {
        let formatted = humantime::format_duration(std::time::Duration::from_secs(
            prediction_duration.0.as_secs(),
        ));
        let duration_string: String = formatted.to_string().split_inclusive(' ').take(2).collect();

        for mut text in &mut query_ui {
            text.sections[1].value = duration_string.to_string();
        }
    }
}

#[derive(Component)]
pub struct Labelled {
    pub entity: Entity,
    pub offset: Vec2,
}

fn update_position_labels(
    query_camera: Query<(&Camera, &GlobalTransform)>,
    query_labelled: Query<(&Labelled, &GlobalTransform)>,
    mut query_labels: Query<(&mut Style, &Node)>,
) {
    let (camera, camera_transform) = query_camera.single();

    for (label, transform) in &query_labelled {
        let Ok((mut style, node)) = query_labels.get_mut(label.entity) else { continue };

        let rotation_matrix = Mat3::from_quat(camera_transform.to_scale_rotation_translation().1);
        let viewport_position = camera
            .world_to_viewport(
                camera_transform,
                transform.translation() + rotation_matrix.mul_vec3(label.offset.extend(0.0)),
            )
            .map(|position| position - node.size() / 2.0);

        if let Some(viewport_position) = viewport_position {
            style.position_type = PositionType::Absolute;
            style.left = Val::Px(viewport_position.x);
            style.top = Val::Px(viewport_position.y);
            style.display = Display::Flex;
        } else {
            style.display = Display::None;
        }
    }
}
