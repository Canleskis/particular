use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use particular::prelude::*;

use crate::rapier_schedule::PreRapierSchedule;

pub const COMPUTE_METHOD: sequential::BruteForceSIMD<8> = sequential::BruteForceSIMD;

pub struct ParticularPlugin;

impl Plugin for ParticularPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(PreRapierSchedule, accelerate_particles);
    }
}

fn accelerate_particles(mut query: Query<(&mut Velocity, &Transform, &ReadMassProperties)>) {
    query
        .iter()
        .map(|(.., transform, mass)| (transform.translation.to_array(), mass.0.mass))
        .accelerations(&mut COMPUTE_METHOD.clone())
        .map(Vec3::from)
        .zip(&mut query)
        .for_each(|(acceleration, (mut velocity, ..))| velocity.linvel += acceleration * crate::DT);
}
