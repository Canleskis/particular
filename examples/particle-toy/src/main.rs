mod nbody;
use nbody::*;

mod physics;
use physics::*;

use bevy::prelude::*;
use rand::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    #[cfg(not(target_arch = "wasm32"))]
                    resolution: bevy::window::WindowResolution::new(1920.0, 1080.0),
                    prevent_default_event_handling: false,
                    fit_canvas_to_parent: true,
                    canvas: Some("#app".to_owned()),
                    ..default()
                }),
                ..default()
            }),
            PhysicsPlugin,
            ParticularPlugin,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(AmbientLight {
            color: Color::NONE,
            brightness: 0.0,
        })
        .insert_resource(PhysicsSettings::delta_time(1.0 / 60.0))
        .add_event::<InputParticleEvent>()
        .add_systems(PreStartup, setup_camera)
        .add_systems(
            PostStartup,
            setup_scene.after(bevy::render::camera::camera_system::<OrthographicProjection>),
        )
        .add_systems(
            PhysicsSchedule,
            (add_friction, bounds)
                .chain()
                .in_set(PhysicsSet::First)
                .before(accelerate_particles),
        )
        .add_systems(Update, (mouse_event, touch_event, input_particle).chain())
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

fn setup_scene(mut commands: Commands, query: Query<&Camera>) {
    let size = query.single().logical_target_size().unwrap() / 2.0;
    let mut rng = thread_rng();

    for _ in 0..20_000 {
        commands.spawn(ParticleBundle::new(
            Vec2::new(
                rng.gen_range(-size.x..size.x),
                rng.gen_range(-size.y..size.y),
            ),
            Vec2::new(rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0)),
            Mass(0.0),
        ));
    }
}

fn add_friction(mut query: Query<(&mut Acceleration, &Velocity)>) {
    for (mut acceleration, velocity) in &mut query {
        **acceleration -= **velocity * 0.6;
    }
}

fn wrap(value: f32, range: f32) -> f32 {
    if value < -range {
        return range;
    } else if value > range {
        return -range;
    }
    value
}

fn bounds(mut query: Query<&mut Position>, query_camera: Query<&Camera>) {
    let size = query_camera.single().logical_target_size().unwrap() / 2.0;

    for mut position in &mut query {
        position.x = wrap(position.x, size.x);
        position.y = wrap(position.y, size.y);
    }
}

#[derive(Event, Deref, DerefMut, Default)]
struct InputParticleEvent(Vec2);

#[derive(Component)]
struct InputParticle;

fn input_particle(
    mut commands: Commands,
    mut events: EventReader<InputParticleEvent>,
    query_particle: Query<Entity, With<InputParticle>>,
) {
    for entity in query_particle.iter() {
        commands.entity(entity).despawn();
    }

    for event in events.read() {
        commands.spawn((
            ParticleBundle::new(event.0, Vec2::ZERO, Mass(1e7)),
            InputParticle,
        ));
    }
}

fn mouse_event(
    mut events: EventWriter<InputParticleEvent>,
    input: Res<Input<MouseButton>>,
    query_window: Query<&Window>,
    query_camera: Query<(&GlobalTransform, &Camera)>,
) {
    let Ok(window) = query_window.get_single() else {
        return;
    };

    let Ok((camera_transform, camera)) = query_camera.get_single() else {
        return;
    };

    if input.pressed(MouseButton::Left) {
        let position = window
            .cursor_position()
            .and_then(|position| camera.viewport_to_world_2d(camera_transform, position));

        if let Some(position) = position {
            events.send(InputParticleEvent(position));
        }
    }
}

// Doesn't work properly because of winit/bevy issues.
fn touch_event(
    mut events: EventWriter<InputParticleEvent>,
    touches: Res<Touches>,
    query_camera: Query<(&GlobalTransform, &Camera)>,
) {
    let Ok((camera_transform, camera)) = query_camera.get_single() else {
        return;
    };

    for touch in touches.iter() {
        let position = camera.viewport_to_world_2d(camera_transform, touch.position());

        if let Some(position) = position {
            events.send(InputParticleEvent(position));
        }
    }
}

#[derive(Bundle, Default)]
struct ParticleBundle {
    sprite_bundle: SpriteBundle,
    pub acceleration: Acceleration,
    pub velocity: Velocity,
    pub position: Position,
    pub mass: Mass,
}

impl ParticleBundle {
    fn new(position: Vec2, velocity: Vec2, point_mass: Mass) -> Self {
        Self {
            sprite_bundle: SpriteBundle {
                transform: Transform::from_translation(position.extend(0.0)),
                ..default()
            },
            velocity: Velocity(velocity),
            position: Position(position),
            mass: point_mass,
            ..default()
        }
    }
}
