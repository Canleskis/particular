use crate::{Acceleration, PhysicsSchedule, PhysicsSet, Position};

use bevy::prelude::*;
use particular::gravity::newtonian::AccelerationSoftened;
use particular::prelude::*;

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

pub fn accelerate_particles(mut query: Query<(&mut Acceleration, &Position, &Mass)>) {
    // It is faster to collect the particles into a vector to compute their accelerations than to
    // iterate over the query directly.
    let particles = query
        .iter()
        .map(|(.., position, mass)| (**position, **mass))
        .collect::<Vec<_>>();

    Reordered::new(&particles, |(_, mass)| *mass != 0.0)
        .brute_force(AccelerationSoftened::checked(100.0))
        .zip(&mut query)
        .for_each(|(acceleration, (mut physics_acceleration, ..))| {
            **physics_acceleration += acceleration
        });
}
