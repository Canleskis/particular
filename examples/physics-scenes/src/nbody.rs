use crate::{rapier_schedule::physics_step, DT, G};

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
        app.add_system_set_to_stage(
            CoreStage::PreUpdate,
            SystemSet::new()
                .with_run_criteria(physics_step)
                .with_system(accelerate_rigidbodies),
        );
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
