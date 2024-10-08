use crate::{PredictionState, DT};

use bevy::prelude::*;
use std::time::Duration;

#[derive(Resource)]
pub struct PhysicsTime {
    current: Duration,
    pub paused: bool,
    pub scale: f32,
}

impl Default for PhysicsTime {
    fn default() -> Self {
        Self {
            current: Duration::ZERO,
            paused: false,
            scale: 1.0,
        }
    }
}

impl PhysicsTime {
    pub fn current(&self) -> Duration {
        self.current
    }

    pub fn reset(&mut self) {
        self.current = Duration::ZERO;
    }
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PhysicsTime::default())
            .add_systems(PreUpdate, tick_time)
            .add_systems(Update, move_bodies);
    }
}

fn tick_time(
    mut physics_time: ResMut<PhysicsTime>,
    time: Res<Time>,
    query: Query<&PredictionState>,
) {
    let max_time = query.iter().fold(Duration::ZERO, |max, state| {
        max.max(state.len() as u32 * DT)
    });
    if !physics_time.paused {
        let delta = time.delta().mul_f32(physics_time.scale);
        physics_time.current += delta;
        physics_time.current = physics_time.current.min(max_time);
    }
}

fn move_bodies(
    physics_time: Res<PhysicsTime>,
    mut query: Query<(&PredictionState, &mut Transform)>,
) {
    for (state, mut transform) in &mut query {
        if let Some(position) = state.evaluate_position(physics_time.current) {
            transform.translation = position;
        }
    }
}
