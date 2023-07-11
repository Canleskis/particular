use crate::{physics::*, Mass};

use bevy::prelude::*;
use particular::prelude::*;

pub const COMPUTE_METHOD: sequential::BruteForce = sequential::BruteForce;

pub struct ParticularPlugin;

impl Plugin for ParticularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(FixedUpdate, accelerate_particles.in_set(PhysicsSet::First));
    }
}

fn accelerate_particles(mut query: Query<(&mut Acceleration, &Transform, &Mass)>) {
    query
        .iter()
        .map(|(_, transform, mass)| (transform.translation, mass.0))
        .accelerations(COMPUTE_METHOD)
        .zip(&mut query)
        .for_each(|(acceleration, (mut acc, _, _))| acc.linear = acceleration);
}
