use crate::{rapier_schedule::PreRapierSchedule, DT, G};

use bevy::prelude::*;
use bevy_rapier2d::prelude::Velocity;

use particular::gravity::newtonian::Acceleration;
use particular::prelude::*;

#[cfg(target_arch = "wasm32")]
const COMPUTE_METHOD: particular::sequential::BruteForceSimd<4, Acceleration<true>> =
    particular::sequential::BruteForceSimd(Acceleration::checked());
#[cfg(not(target_arch = "wasm32"))]
const COMPUTE_METHOD: particular::parallel::BruteForceSimd<8, Acceleration<true>> =
    particular::parallel::BruteForceSimd(Acceleration::checked());

pub struct ParticularPlugin;

impl Plugin for ParticularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(PreRapierSchedule, accelerate_rigidbodies);
    }
}

#[derive(Component)]
pub enum PointMass {
    HasGravity { mass: f32 },
    AffectedByGravity,
}

impl PointMass {
    fn mass(&self) -> f32 {
        match *self {
            PointMass::HasGravity { mass } => mass,
            PointMass::AffectedByGravity => 0.0,
        }
    }
}

fn accelerate_rigidbodies(mut query: Query<(&mut Velocity, &GlobalTransform, &PointMass)>) {
    // It is faster to collect the particles into a vector to compute their accelerations than to
    // iterate over the query directly.
    let particles = query
        .iter()
        .map(|(.., transform, mass)| (transform.translation().truncate(), mass.mass() * G))
        .collect::<Vec<_>>();
    let reoredered = Reordered::new(&particles, |(_, mass)| *mass != 0.0);

    particular::Interaction::compute(&mut COMPUTE_METHOD.clone(), &reoredered)
        // Collecting is necessary because the returned iterator is a rayon parallel iterator and
        // Bevy's `Query` does not implement rayon's `IntoParallelIterator` trait.
        .collect::<Vec<_>>()
        .into_iter()
        .zip(&mut query)
        .for_each(|(acceleration, (mut velocity, ..))| {
            velocity.linvel += acceleration * DT;
        })
}
