use crate::{PhysicsSettings, PredictionDuration};

use bevy::prelude::*;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_ui).add_systems(
            Update,
            (update_time_scale_ui, update_prediction_duration_ui),
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
                    TextSection::from_style(style),
                ]),
                UiPredictionDuration,
            ));
        });
}

fn update_time_scale_ui(
    physics: Res<PhysicsSettings>,
    mut query_ui: Query<&mut Text, With<UiTimeScale>>,
) {
    for mut text in &mut query_ui {
        text.sections[1].value = format!("{:.1}", physics.time_scale);
    }
}

fn update_prediction_duration_ui(
    prediction_duration: Res<PredictionDuration>,
    mut query_ui: Query<&mut Text, With<UiPredictionDuration>>,
) {
    let formatted = humantime::format_duration(std::time::Duration::from_secs(
        prediction_duration.0.as_secs(),
    ));
    let duration_string: String = formatted.to_string().split_inclusive(' ').take(2).collect();

    for mut text in &mut query_ui {
        text.sections[1].value = duration_string.to_string();
    }
}
