use crate::{rapier_schedule::PreRapierSchedule, DT, G};

use bevy::prelude::*;
use bevy_rapier2d::prelude::Velocity;
use particular::prelude::*;

#[cfg(target_arch = "wasm32")]
const COMPUTE_METHOD: sequential::BruteForceSIMD<4> = sequential::BruteForceSIMD;
#[cfg(not(target_arch = "wasm32"))]
const COMPUTE_METHOD: parallel::BruteForceSIMD<8> = parallel::BruteForceSIMD;

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
    query
        .iter()
        .map(|(.., transform, mass)| {
            (
                transform.translation().truncate().to_array(),
                mass.mass() * G,
            )
        })
        .accelerations(&mut COMPUTE_METHOD.clone())
        .zip(&mut query)
        .for_each(|(acceleration, (mut velocity, ..))| {
            velocity.linvel += Vec2::from(acceleration) * DT;
        })
}
