use crate::{Acceleration, PhysicsSet, Position};

use bevy::prelude::*;
use particular::prelude::*;

pub const COMPUTE_METHOD: sequential::BruteForcePairsAlt = sequential::BruteForcePairsAlt;

#[derive(Component, Clone, Copy, Default, Deref, DerefMut)]
pub struct Mass(pub f32);

pub struct ParticularPlugin;

impl Plugin for ParticularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(FixedUpdate, accelerate_particles.in_set(PhysicsSet::First));
    }
}

fn accelerate_particles(mut query: Query<(&mut Acceleration, &Position, &Mass)>) {
    query
        .iter()
        .map(|(_, position, mass)| (**position, **mass))
        .accelerations(COMPUTE_METHOD)
        .zip(&mut query)
        .for_each(|(acceleration, (mut acc, _, _))| **acc = acceleration);
}
