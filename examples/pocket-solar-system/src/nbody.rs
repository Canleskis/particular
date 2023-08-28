use crate::{Acceleration, PhysicsSchedule, PhysicsSet, Position};

use bevy::prelude::*;
use particular::prelude::*;

pub const COMPUTE_METHOD: sequential::BruteForcePairsAlt = sequential::BruteForcePairsAlt;

#[derive(Component, Clone, Copy, Default, Deref, DerefMut)]
pub struct Mass(pub f32);

pub struct ParticularPlugin;

impl Plugin for ParticularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(
            PhysicsSchedule,
            accelerate_particles.in_set(PhysicsSet::First),
        );
    }
}

fn accelerate_particles(mut query: Query<(&mut Acceleration, &Position, &Mass)>) {
    query
        .iter()
        .map(|(.., position, mass)| (**position, **mass))
        .accelerations(COMPUTE_METHOD)
        .zip(&mut query)
        .for_each(|(acceleration, (mut physics_acceleration, ..))| {
            **physics_acceleration = acceleration
        });
}