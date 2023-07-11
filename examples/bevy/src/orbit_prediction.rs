use crate::{nbody::COMPUTE_METHOD, physics::Velocity, Mass, SelectableEntities};

use bevy::prelude::*;
use particular::prelude::*;

pub struct OrbitPredictionPlugin;

impl Plugin for OrbitPredictionPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GizmoConfig {
            line_width: 1.0,
            ..default()
        })
        .add_systems(Update, (reset_paths, compute_paths, draw_paths).chain());
    }
}

#[derive(Component, Default)]
pub struct PredictionState {
    pub velocities: Vec<Vec3>,
    pub positions: Vec<Vec3>,
    pub reference: Option<Entity>,
}

fn compute_paths(
    input: Res<Input<KeyCode>>,
    fixed_time: Res<FixedTime>,
    selectable_entites: Res<SelectableEntities>,
    mut query: Query<(&Velocity, &Transform, &Mass, &mut PredictionState)>,
) {
    if selectable_entites.is_changed() {
        for (_, _, _, mut prediction) in &mut query {
            prediction.reference = selectable_entites.selected();
        }
    }

    if input.pressed(KeyCode::F) {
        let dt = fixed_time.period.as_secs_f32();

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

        for _ in 0..10 {
            mapped_query
                .iter()
                .map(|&(position, _, mass, _)| (position, mass))
                .accelerations(COMPUTE_METHOD)
                .zip(&mut mapped_query)
                .for_each(|(acceleration, (position, velocity, _, prediction))| {
                    *velocity += acceleration * dt;
                    *position += *velocity * dt;

                    prediction.velocities.push(*velocity);
                    prediction.positions.push(*position);
                });
        }
    }
}

fn draw_paths(mut gizmos: Gizmos, query: Query<(&Transform, &PredictionState)>) {
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
                    .map(|(&position, reference_position)| position - reference_position),
                Color::WHITE,
            )
        }
    }
}

fn reset_paths(input: Res<Input<KeyCode>>, mut query: Query<&mut PredictionState>) {
    if input.just_pressed(KeyCode::R) {
        for mut prediction in &mut query {
            prediction.velocities.clear();
            prediction.positions.clear();
        }
    }
}
