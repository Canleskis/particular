use bevy::prelude::*;

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum PhysicsSet {
    First,
    Main,
    Last,
}

#[derive(Resource)]
pub struct PhysicsSettings {
    pub time_scale: f32,
    pub delta_time: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            time_scale: 1.0,
            delta_time: 1.0 / 60.0,
        }
    }
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
        app.configure_sets(
            FixedUpdate,
            (PhysicsSet::First, PhysicsSet::Main, PhysicsSet::Last).chain(),
        )
        .insert_resource(PhysicsSettings {
            time_scale: 1.0,
            ..default()
        })
        .add_systems(PreUpdate, time_scale)
        .add_systems(FixedUpdate, integrate_position.in_set(PhysicsSet::Main))
        .register_type::<Acceleration>()
        .register_type::<Velocity>();
    }
}

fn time_scale(
    time: Res<Time>,
    input: Res<Input<KeyCode>>,
    mut physics: ResMut<PhysicsSettings>,
    mut fixed_time: ResMut<FixedTime>,
) {
    let input = input.pressed(KeyCode::Right) as isize - input.pressed(KeyCode::Left) as isize;

    if input != 0 || physics.is_added() {
        physics.time_scale += physics.time_scale * input as f32 * time.delta_seconds() * 2.0;
        physics.time_scale = physics.time_scale.clamp(0.5, 100.0);
        fixed_time.period =
            std::time::Duration::from_secs_f32(physics.delta_time / physics.time_scale);
    }
}

fn integrate_position(
    physics: Res<PhysicsSettings>,
    mut query: Query<(&mut Acceleration, &mut Velocity, &mut Transform)>,
) {
    for (mut acceleration, mut velocity, mut transform) in &mut query {
        velocity.linear += acceleration.linear * physics.delta_time;
        transform.translation += velocity.linear * physics.delta_time;

        acceleration.linear = Vec3::ZERO;
    }
}
