use crate::{Mass, PhysicsSettings, Position, Velocity, COMPUTE_METHOD};

use bevy::prelude::*;
use particular::prelude::*;

#[derive(Event, Clone, Copy)]
pub struct ComputePredictionEvent {
    pub steps: usize,
}

impl FromWorld for ComputePredictionEvent {
    fn from_world(world: &mut World) -> Self {
        Self {
            steps: world.resource::<PhysicsSettings>().steps_per_second() * 60 * 5,
        }
    }
}

#[derive(Event, Clone, Copy)]
pub struct ResetPredictionEvent;

#[derive(Bundle, Default)]
pub struct PredictionBundle {
    pub state: PredictionState,
    pub draw: PredictionDraw,
}

#[derive(Component, Default)]
pub struct PredictionState {
    pub velocity: Option<Vec3>,
    pub positions: Vec<Vec3>,
}

impl PredictionState {
    pub fn push(&mut self, velocity: Vec3, position: Vec3) {
        self.velocity.replace(velocity);
        self.positions.push(position);
    }

    pub fn reset(&mut self) {
        self.velocity.take();
        self.positions.clear();
    }
}

#[derive(Component)]
pub struct PredictionDraw {
    pub color: Color,
    pub resolution: usize,
    pub steps: Option<usize>,
    pub reference: Option<Entity>,
}

impl Default for PredictionDraw {
    fn default() -> Self {
        Self {
            color: Default::default(),
            resolution: 5,
            steps: Default::default(),
            reference: Default::default(),
        }
    }
}

pub struct OrbitPredictionPlugin;

impl Plugin for OrbitPredictionPlugin {
    fn build(&self, app: &mut App) {
        app.add_event::<ComputePredictionEvent>()
            .add_event::<ResetPredictionEvent>()
            .insert_resource(GizmoConfig {
                line_width: 1.0,
                ..default()
            })
            .add_systems(
                PostUpdate,
                (reset_prediction, compute_prediction, draw_prediction).chain(),
            );
    }
}

fn reset_prediction(
    reset_event: EventReader<ResetPredictionEvent>,
    mut query: Query<&mut PredictionState>,
) {
    if reset_event.is_empty() {
        return;
    }

    query.for_each_mut(|mut state| state.reset());
}

#[cfg(target_arch = "wasm32")]
fn compute_prediction(
    physics: Res<PhysicsSettings>,
    mut compute_event: EventReader<ComputePredictionEvent>,
    mut query: Query<(&Velocity, &Position, &Mass, &mut PredictionState)>,
    mut progress: Local<usize>,
) {
    let mut steps_per_frame = 5000;
    let event = compute_event.iter().next();
    let dt = physics.delta_time;

    if let Some(&ComputePredictionEvent { steps }) = event {
        *progress = steps;

        for (.., mut prediction) in &mut query {
            prediction.positions.reserve(steps);
        }
    }

    if *progress != 0 {
        let mut mapped_query: Vec<_> = query
            .iter_mut()
            .map(|(velocity, position, mass, state)| {
                (
                    state.velocity.unwrap_or(**velocity),
                    *state.positions.last().unwrap_or(position),
                    **mass,
                    state,
                )
            })
            .collect();

        steps_per_frame = steps_per_frame.min(*progress);

        for _ in 0..steps_per_frame {
            mapped_query
                .iter()
                .map(|&(.., position, mass, _)| (position.to_array(), mass))
                .accelerations(&mut COMPUTE_METHOD.clone())
                .map(Vec3::from)
                .zip(&mut mapped_query)
                .for_each(|(acceleration, (velocity, position, .., state))| {
                    *velocity += acceleration * dt;
                    *position += *velocity * dt;

                    state.push(*velocity, *position);
                });
        }

        *progress -= steps_per_frame;
    }
}

#[cfg(not(target_arch = "wasm32"))]
type PredictionReceiver = crossbeam_channel::Receiver<Vec<(Entity, Vec3, Vec3)>>;

#[cfg(not(target_arch = "wasm32"))]
fn compute_prediction(
    physics: Res<PhysicsSettings>,
    mut compute_event: EventReader<ComputePredictionEvent>,
    mut query: Query<(Entity, &Velocity, &Position, &Mass, &mut PredictionState)>,
    mut receiver: Local<Option<PredictionReceiver>>,
) {
    let event = compute_event.iter().next();
    let dt = physics.delta_time;

    if let Some(&ComputePredictionEvent { steps }) = event {
        for (.., mut prediction) in &mut query {
            prediction.positions.reserve(steps);
        }

        let mut mapped_query: Vec<_> = query
            .iter()
            .map(|(entity, velocity, position, mass, state)| {
                (
                    entity,
                    state.velocity.unwrap_or(**velocity),
                    *state.positions.last().unwrap_or(position),
                    **mass,
                )
            })
            .collect();

        let (tx, rx) = crossbeam_channel::bounded(steps);
        *receiver = Some(rx);

        std::thread::spawn(move || {
            for _ in 0..steps {
                tx.send(
                    mapped_query
                        .iter()
                        .map(|&(.., position, mass)| (position.to_array(), mass))
                        .accelerations(&mut COMPUTE_METHOD.clone())
                        .map(Vec3::from)
                        .zip(&mut mapped_query)
                        .map(|(acceleration, (entity, velocity, position, ..))| {
                            (*velocity, *position) =
                                crate::sympletic_euler(acceleration, *velocity, *position, dt);

                            (*entity, *velocity, *position)
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            }
        });
    }

    if let Some(receiver) = &*receiver {
        for received in receiver.try_iter() {
            for (entity, velocity, position) in received {
                if let Ok((.., mut state)) = query.get_mut(entity) {
                    state.push(velocity, position);
                }
            }
        }
    }
}

fn draw_prediction(
    mut gizmos: Gizmos,
    query: Query<(&Transform, &PredictionState, &PredictionDraw)>,
) {
    for (_, state, draw) in query.iter() {
        let steps = draw.steps.unwrap_or(usize::MAX);
        let positions = state.positions.iter().take(steps);

        let reference = draw.reference.and_then(|r| query.get(r).ok());
        if let Some((ref_transform, ref_state, _)) = reference {
            gizmos.linestrip(
                positions
                    .zip(&ref_state.positions)
                    .step_by(draw.resolution)
                    .map(|(&pred_position, &ref_pred_position)| {
                        pred_position - ref_pred_position + ref_transform.translation
                    }),
                draw.color,
            )
        } else {
            gizmos.linestrip(positions.step_by(draw.resolution).copied(), draw.color)
        }
    }
}
