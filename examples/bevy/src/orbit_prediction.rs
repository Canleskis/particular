use crate::{Mass, PhysicsSettings, Position, Selected, Velocity, COMPUTE_METHOD};

use bevy::prelude::*;
use particular::prelude::*;
use std::time::Duration;

#[derive(Resource, Default, Deref, DerefMut)]
pub struct PredictionDuration(pub Duration);

#[derive(Component, Default)]
pub struct PredictionState {
    pub velocity: Option<Vec3>,
    pub positions: Vec<Vec3>,
    pub reference: Option<Entity>,
    pub color: Color,
}

pub struct OrbitPredictionPlugin;

impl Plugin for OrbitPredictionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GizmoConfig {
            line_width: 1.0,
            ..default()
        })
        .insert_resource(PredictionDuration::default())
        .add_systems(
            PostUpdate,
            (reset_prediction, compute_prediction, draw_prediction).chain(),
        );
    }
}

fn compute_prediction(
    time: Res<Time>,
    input: Res<Input<KeyCode>>,
    physics: Res<PhysicsSettings>,
    query_selected: Query<Entity, Added<Selected>>,
    mut prediction_duration: ResMut<PredictionDuration>,
    mut query: Query<(&Velocity, &Position, &Mass, &mut PredictionState)>,
) {
    if let Ok(selected_entity) = query_selected.get_single() {
        for (_, _, _, mut prediction) in &mut query {
            prediction.reference = Some(selected_entity);
        }
    }

    if input.pressed(KeyCode::F) {
        let steps = (physics.time_scale * time.delta_seconds() * 4000.0).ceil() as usize;
        prediction_duration.0 += Duration::from_secs_f32(steps as f32 * physics.delta_time);

        for (_, _, _, mut prediction) in &mut query {
            prediction.positions.reserve(steps);
        }

        let mut mapped_query: Vec<_> = query
            .iter_mut()
            .map(|(v, p, mass, prediction)| {
                (
                    *prediction.positions.last().unwrap_or(p),
                    prediction.velocity.unwrap_or(**v),
                    mass.0,
                    prediction,
                )
            })
            .collect();

        for _ in 0..steps {
            mapped_query
                .iter()
                .map(|&(position, _, mass, _)| (position, mass))
                .accelerations(COMPUTE_METHOD)
                .zip(&mut mapped_query)
                .for_each(|(acceleration, (position, velocity, _, prediction))| {
                    *velocity += acceleration * physics.delta_time;
                    *position += *velocity * physics.delta_time;

                    prediction.velocity.replace(*velocity);
                    prediction.positions.push(*position);
                });
        }
    }
}

fn draw_prediction(mut gizmos: Gizmos, query: Query<(&Transform, &PredictionState)>) {
    let resolution = 5;
    for (_, prediction) in &query {
        let reference = prediction.reference.and_then(|r| query.get(r).ok());

        if let Some((ref_transform, ref_prediction)) = reference {
            gizmos.linestrip(
                prediction
                    .positions
                    .iter()
                    .zip(&ref_prediction.positions)
                    .step_by(resolution)
                    .map(|(&pred_position, &ref_pred_position)| {
                        pred_position - ref_pred_position + ref_transform.translation
                    }),
                prediction.color,
            )
        } else {
            gizmos.linestrip(
                prediction.positions.iter().step_by(resolution).copied(),
                prediction.color,
            )
        }
    }
}

fn reset_prediction(
    input: Res<Input<KeyCode>>,
    mut prediction_duration: ResMut<PredictionDuration>,
    mut query: Query<&mut PredictionState>,
) {
    if input.just_pressed(KeyCode::R) {
        *prediction_duration = Default::default();

        for mut prediction in &mut query {
            prediction.velocity.take();
            prediction.positions.clear();
        }
    }
}
