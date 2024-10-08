use crate::{steps_per_second, PhysicsTime, DT};

use bevy::prelude::*;
use std::time::Duration;

use particular::gravity::newtonian;
use particular::prelude::*;

#[derive(Event, Clone, Copy)]
pub struct ComputePredictionEvent {
    pub steps: usize,
}

impl Default for ComputePredictionEvent {
    fn default() -> Self {
        Self {
            steps: steps_per_second() * 60 * 5,
        }
    }
}

#[derive(Event, Clone, Copy)]
pub struct ResetPredictionEvent;

#[derive(Bundle, Default)]
pub struct PredictionBundle {
    pub mass: Mass,
    pub state: PredictionState,
    pub draw: PredictionDraw,
}

#[derive(Component, Default, Deref, DerefMut)]
pub struct Mass(pub f32);

#[derive(Component, Default)]
pub struct PredictionState {
    pub velocities: Vec<Vec3>,
    pub positions: Vec<Vec3>,
}

impl PredictionState {
    pub fn len(&self) -> usize {
        self.positions.len().saturating_sub(1)
    }

    pub fn push(&mut self, velocity: Vec3, position: Vec3) {
        self.velocities.push(velocity);
        self.positions.push(position);
    }

    pub fn evaluate_position(&self, time: Duration) -> Option<Vec3> {
        let index = (time.as_nanos().saturating_sub(1) / DT.as_nanos()) as usize;
        let s = (time - index as u32 * DT).as_secs_f32() / DT.as_secs_f32();
        self.positions
            .get(index)
            .zip(self.positions.get(index + 1))
            .map(|(pos1, pos2)| pos1.lerp(*pos2, s))
    }

    pub fn at(&self, time: Duration) -> Option<(Vec3, Vec3)> {
        let index = (time.as_nanos() / DT.as_nanos()) as usize;

        self.positions
            .get(index)
            .copied()
            .zip(self.velocities.get(index).copied())
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
    mut physics_time: ResMut<PhysicsTime>,
    mut reset_event: EventReader<ResetPredictionEvent>,
    mut query: Query<&mut PredictionState>,
) {
    if reset_event.is_empty() {
        return;
    }
    reset_event.clear();

    for mut state in query.iter_mut() {
        if let Some((current_pos, current_vel)) = state.at(physics_time.current()) {
            state.positions.clear();
            state.positions.push(current_pos);

            state.velocities.clear();
            state.velocities.push(current_vel);
        }
    }
    physics_time.reset();
}

#[cfg(target_arch = "wasm32")]
fn compute_prediction(
    mut compute_event: EventReader<ComputePredictionEvent>,
    mut query: Query<(&Mass, &mut PredictionState)>,
    mut progress: Local<usize>,
) {
    let mut steps_per_frame = 5000;
    let event = compute_event.read().next();

    if let Some(&ComputePredictionEvent { steps }) = event {
        *progress = steps;

        for (.., mut prediction) in &mut query {
            prediction.positions.reserve(steps);
        }
    }

    if *progress != 0 {
        let (mut velocities, mut point_masses): (Vec<_>, Vec<_>) = query
            .iter()
            .map(|(mass, state)| {
                (
                    *state.velocities.last().unwrap(),
                    (*state.positions.last().unwrap(), **mass),
                )
            })
            .unzip();

        steps_per_frame = steps_per_frame.min(*progress);

        for _ in 0..steps_per_frame {
            point_masses
                .brute_force(newtonian::Acceleration::checked())
                .zip(velocities.iter_mut())
                .for_each(|(acceleration, velocity)| {
                    *velocity = crate::sympletic_euler_velocity(acceleration, *velocity, DT);
                });
            point_masses
                .iter_mut()
                .zip(velocities.iter())
                .for_each(|((position, _), velocity)| {
                    *position = crate::sympletic_euler_position(*velocity, *position, DT);
                });

            point_masses
                .iter()
                .zip(velocities.iter())
                .zip(query.iter_mut())
                .for_each(|(((position, _), velocity), (.., mut state))| {
                    state.push(*velocity, *position);
                });
        }

        *progress -= steps_per_frame;
    }
}

#[cfg(not(target_arch = "wasm32"))]
type PredictionReceiver = crossbeam_channel::Receiver<PredictionMessage>;

#[cfg(not(target_arch = "wasm32"))]
enum PredictionMessage {
    Done,
    Send(Vec<(Entity, Vec3, Vec3)>),
}

#[cfg(not(target_arch = "wasm32"))]
fn compute_prediction(
    mut compute_event: EventReader<ComputePredictionEvent>,
    mut query: Query<(Entity, &Mass, &mut PredictionState)>,
    mut receiver: Local<Option<PredictionReceiver>>,
) {
    let event = compute_event.read().next();

    if let Some(&ComputePredictionEvent { steps }) = event {
        for (.., mut prediction) in &mut query {
            prediction.positions.reserve(steps);
        }

        let (entities, (mut velocities, mut point_masses)): (Vec<_>, (Vec<_>, Vec<_>)) = query
            .iter()
            .map(|(entity, mass, state)| {
                (
                    entity,
                    (
                        *state.velocities.last().unwrap(),
                        (*state.positions.last().unwrap(), **mass),
                    ),
                )
            })
            .unzip();

        let (tx, rx) = crossbeam_channel::bounded(steps);
        *receiver = Some(rx);

        std::thread::spawn(move || {
            for _ in 0..steps {
                point_masses
                    .brute_force(newtonian::Acceleration::checked())
                    .zip(velocities.iter_mut())
                    .for_each(|(acceleration, velocity)| {
                        *velocity = crate::sympletic_euler_velocity(acceleration, *velocity, DT);
                    });
                point_masses.iter_mut().zip(velocities.iter()).for_each(
                    |((position, _), velocity)| {
                        *position = crate::sympletic_euler_position(*velocity, *position, DT);
                    },
                );

                tx.send(PredictionMessage::Send(
                    entities
                        .iter()
                        .zip(velocities.iter())
                        .zip(point_masses.iter())
                        .map(|((entity, velocity), (position, _))| (*entity, *velocity, *position))
                        .collect::<Vec<_>>(),
                ))
                .unwrap();
            }

            tx.send(PredictionMessage::Done).unwrap();
        });
    }

    if let Some(receiver_inner) = receiver.clone() {
        for received in receiver_inner.try_iter() {
            if let PredictionMessage::Send(received) = received {
                for (entity, velocity, position) in received {
                    if let Ok((.., mut state)) = query.get_mut(entity) {
                        state.push(velocity, position);
                    }
                }
            } else {
                *receiver = None;
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

pub fn sympletic_euler_velocity(acceleration: Vec3, velocity: Vec3, dt: Duration) -> Vec3 {
    velocity + acceleration * dt.as_secs_f32()
}

pub fn sympletic_euler_position(velocity: Vec3, position: Vec3, dt: Duration) -> Vec3 {
    position + velocity * dt.as_secs_f32()
}
