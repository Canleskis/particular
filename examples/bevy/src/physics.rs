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

#[derive(Component, Clone, Copy, Default, Deref, DerefMut, Reflect)]
pub struct Position(pub Vec3);

#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
pub struct Interpolated {
    previous_position: Option<Vec3>,
}

#[derive(Component, Clone, Copy, Default, Deref, DerefMut, Reflect)]
pub struct Velocity(pub Vec3);

#[derive(Component, Clone, Copy, Default, Deref, DerefMut, Reflect)]
pub struct Acceleration(pub Vec3);

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            FixedUpdate,
            (PhysicsSet::First, PhysicsSet::Main, PhysicsSet::Last).chain(),
        )
        .add_systems(PreUpdate, time_scale)
        .add_systems(
            FixedUpdate,
            (cache_previous_position, integrate_position)
                .chain()
                .in_set(PhysicsSet::Main),
        )
        .add_systems(Update, update_transform)
        .register_type::<Acceleration>()
        .register_type::<Velocity>();
    }
}

fn time_scale(
    time: Res<Time>,
    input: Res<Input<KeyCode>>,
    mut fixed_time: ResMut<FixedTime>,
    mut physics: ResMut<PhysicsSettings>,
) {
    let left_input = input.pressed(KeyCode::Left) as isize;
    let right_input = input.pressed(KeyCode::Right) as isize;
    let input_value = right_input - left_input;

    if input_value != 0 || physics.is_added() {
        physics.time_scale += physics.time_scale * input_value as f32 * time.delta_seconds() * 2.0;
        physics.time_scale = physics.time_scale.clamp(0.05, 100.0);
        fixed_time.period =
            std::time::Duration::from_secs_f32(physics.delta_time / physics.time_scale);
    }
}

fn cache_previous_position(mut query: Query<(&Position, &mut Interpolated)>) {
    for (position, mut interpolated) in &mut query {
        interpolated.previous_position = Some(**position);
    }
}

fn integrate_position(
    physics: Res<PhysicsSettings>,
    mut query: Query<(&mut Acceleration, &mut Velocity, &mut Position)>,
) {
    for (mut acceleration, mut velocity, mut position) in &mut query {
        **velocity += **acceleration * physics.delta_time;
        **position += **velocity * physics.delta_time;

        **acceleration = Vec3::ZERO;
    }
}

/// Interpolates or sets the transform depending on if the entity has an [`Interpolated`] component.
/// Interpolation doesn't behave properly when the time scale is changed with high time steps.
fn update_transform(
    fixed_time: Res<FixedTime>,
    mut query: Query<(&mut Transform, &Position, Option<&Interpolated>)>,
) {
    let s = fixed_time.accumulated().as_secs_f32() / fixed_time.period.as_secs_f32();

    for (mut transform, position, interpolated) in &mut query {
        let new_position = interpolated
            .and_then(|interpolated| interpolated.previous_position)
            .map(|previous_position| previous_position.lerp(**position, s))
            .unwrap_or(**position);

        if transform.translation != new_position {
            transform.translation = new_position;
        }
    }
}
