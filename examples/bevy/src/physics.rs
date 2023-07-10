use bevy::prelude::*;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum PhysicsSet {
    First,
    Last,
}

#[derive(Component, Default, Reflect)]
pub struct Acceleration {
    pub linear: Vec3,
}

#[derive(Component, Default, Reflect)]
pub struct Velocity {
    pub linear: Vec3,
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(FixedUpdate, (PhysicsSet::First, PhysicsSet::Last).chain())
            .add_systems(FixedUpdate, integrate_position.in_set(PhysicsSet::Last))
            .register_type::<Acceleration>()
            .register_type::<Velocity>();
    }
}

fn integrate_position(
    fixed_time: Res<FixedTime>,
    mut query: Query<(&mut Acceleration, &mut Velocity, &mut Transform)>,
) {
    let dt = fixed_time.period.as_secs_f32();
    for (mut acceleration, mut velocity, mut transform) in &mut query {
        velocity.linear += acceleration.linear * dt;
        transform.translation += velocity.linear * dt;

        acceleration.linear = Vec3::ZERO;
    }
}
