use crate::rapier_schedule::PreRapierSchedule;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use particular::gravity::newtonian::Acceleration;
use particular::prelude::*;

pub struct ParticularPlugin;

impl Plugin for ParticularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(PreRapierSchedule, accelerate_particles);
    }
}

fn accelerate_particles(mut query: Query<(&mut Velocity, &Transform, &ReadMassProperties)>) {
    // It is faster to collect the particles into a vector to compute their accelerations than to
    // iterate over the query directly.
    let particles = query
        .iter()
        .map(|(_, transform, mass)| (transform.translation, mass.get().mass))
        .collect::<Vec<_>>();

    particles
        .brute_force_simd::<8>(Acceleration::checked())
        .zip(&mut query)
        .for_each(|(acceleration, (mut velocity, ..))| velocity.linvel += acceleration * crate::DT);
}
