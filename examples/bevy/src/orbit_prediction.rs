use crate::{Mass, PhysicsSettings, SelectableEntities, Velocity, COMPUTE_METHOD};

use bevy::prelude::*;
use particular::prelude::*;
use std::time::Duration;

#[derive(Resource, Default, Deref, DerefMut)]
pub struct PredictionDuration(pub Duration);

#[derive(Component, Default)]
pub struct PredictionState {
    pub velocities: Vec<Vec3>,
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
            Update,
            (reset_prediction, compute_prediction, draw_prediction).chain(),
        );
    }
}

fn compute_prediction(
    time: Res<Time>,
    input: Res<Input<KeyCode>>,
    physics: Res<PhysicsSettings>,
    selectable_entites: Res<SelectableEntities>,
    mut prediction_duration: ResMut<PredictionDuration>,
    mut query: Query<(&Velocity, &Transform, &Mass, &mut PredictionState)>,
) {
    if selectable_entites.is_changed() {
        for (_, _, _, mut prediction) in &mut query {
            prediction.reference = selectable_entites.selected();
        }
    }

    if input.pressed(KeyCode::F) {
        let steps = (physics.time_scale * time.delta_seconds() * 4000.0).ceil() as usize;
        prediction_duration.0 += Duration::from_secs_f32(steps as f32 * physics.delta_time);

        for (_, _, _, mut prediction) in &mut query {
            prediction.positions.reserve(steps);
            prediction.velocities.reserve(steps);
        }

        let mut mapped_query: Vec<_> = query
            .iter_mut()
            .map(|(v, t, mass, prediction)| {
                (
                    *prediction.positions.last().unwrap_or(&t.translation),
                    *prediction.velocities.last().unwrap_or(&v.linear),
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

                    prediction.velocities.push(*velocity);
                    prediction.positions.push(*position);
                });
        }
    }
}

fn draw_prediction(mut gizmos: Gizmos, query: Query<(&Transform, &PredictionState)>) {
    let resolution = 5;
    for (_, prediction) in &query {
        let reference_positions = prediction.reference.and_then(|reference| {
            query
                .get(reference)
                .map(|(transform, prediction)| {
                    prediction
                        .positions
                        .iter()
                        .map(|&position| position - transform.translation)
                })
                .ok()
        });

        if let Some(reference_positions) = reference_positions {
            gizmos.linestrip(
                prediction
                    .positions
                    .iter()
                    .zip(reference_positions)
                    .step_by(resolution)
                    .map(|(&position, reference_position)| position - reference_position),
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
            prediction.velocities.clear();
            prediction.positions.clear();
        }
    }
}
