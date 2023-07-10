mod physics;
use physics::*;

mod nbody;
use nbody::*;

use bevy::{
    core_pipeline::bloom::BloomSettings,
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PhysicsPlugin, ParticularPlugin))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(FixedTime::default())
        .add_systems(Startup, setup_scene)
        .add_systems(Update, (switch_parent_camera, zoom_camera))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let light_color = Color::rgb(1.0, 1.0, 0.9);

    let camera = commands
        .spawn((
            Camera3dBundle {
                transform: Transform::from_xyz(0.0, 0.0, 200.0)
                    .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                ..default()
            },
            BloomSettings {
                intensity: 0.15,
                ..default()
            },
        ))
        .id();

    let light = commands
        .spawn(PointLightBundle {
            point_light: PointLight {
                color: light_color,
                intensity: 1E5,
                range: 2E3,
                shadows_enabled: true,
                ..default()
            },
            transform: Transform::from_xyz(0.0, 0.0, 0.0),
            ..default()
        })
        .id();

    let main_position = Vec3::ZERO;
    let main_mu = 5E3;

    let secondary_position = Vec3::new(60.0, 0.0, 0.0);
    let secondary_mu = 100.0;

    let tertiary_position = secondary_position + Vec3::new(0.0, 4.0, 4.0);
    let tertiary_mu = 0.0;

    commands
        .spawn(ParticleBundle {
            pbr_bundle: PbrBundle {
                mesh: meshes.add(
                    shape::UVSphere {
                        radius: 8.0,
                        ..default()
                    }
                    .into(),
                ),
                material: materials.add(StandardMaterial {
                    emissive: light_color * 2.0,
                    ..default()
                }),
                transform: Transform::from_translation(Vec3::ZERO),
                ..default()
            },
            mass: Mass(main_mu),
            ..default()
        })
        .add_child(light)
        .add_child(camera);

    let secondary_velocity =
        circular_orbit_velocity(secondary_position, main_position, main_mu, Vec3::Z);
    commands.spawn(ParticleBundle {
        pbr_bundle: PbrBundle {
            mesh: meshes.add(
                shape::UVSphere {
                    radius: 2.0,
                    ..default()
                }
                .into(),
            ),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(0.0, 0.3, 1.0),
                ..default()
            }),
            transform: Transform::from_translation(secondary_position),
            ..default()
        },
        velocity: Velocity {
            linear: secondary_velocity,
        },
        mass: Mass(secondary_mu),
        ..default()
    });

    let tertiary_velocity = circular_orbit_velocity(
        tertiary_position,
        secondary_position,
        secondary_mu,
        Vec3::new(1.0, 1.0, 0.0),
    ) + secondary_velocity;
    commands.spawn(ParticleBundle {
        pbr_bundle: PbrBundle {
            mesh: meshes.add(
                shape::UVSphere {
                    radius: 0.8,
                    ..default()
                }
                .into(),
            ),
            material: materials.add(StandardMaterial {
                base_color: Color::rgb(1.0, 0.97, 0.91),
                ..default()
            }),
            transform: Transform::from_translation(tertiary_position),
            ..default()
        },
        velocity: Velocity {
            linear: tertiary_velocity,
        },
        mass: Mass(tertiary_mu),
        ..default()
    });
}

fn switch_parent_camera(
    mut commands: Commands,
    input: Res<Input<KeyCode>>,
    query_bodies: Query<Entity, With<Mass>>,
    query_camera: Query<Entity, With<Camera>>,
    mut current: Local<usize>,
) {
    let Ok(camera_entity) = query_camera.get_single() else {
        return
    };

    if input.just_pressed(KeyCode::Space) {
        let mut bodies: Vec<_> = query_bodies.iter().collect();
        bodies.sort();

        *current = (*current + 1) % bodies.len();

        commands.entity(camera_entity).set_parent(bodies[*current]);
    }
}

fn zoom_camera(
    mut scroll_events: EventReader<MouseWheel>,
    mut query_camera: Query<&mut Transform, With<Camera>>,
) {
    let Ok(mut camera_transform) = query_camera.get_single_mut() else {
        return
    };

    let pixels_per_line = 10.;
    let scroll = scroll_events
        .iter()
        .map(|ev| match ev.unit {
            MouseScrollUnit::Pixel => ev.y,
            MouseScrollUnit::Line => ev.y * pixels_per_line,
        })
        .sum::<f32>();

    if scroll == 0.0 {
        return;
    }

    camera_transform.translation.z =
        (camera_transform.translation.z * (1.0 + -scroll * 0.01)).max(10.0);
}

#[derive(Component, Default)]
struct Mass(f32);

#[derive(Bundle, Default)]
struct ParticleBundle {
    pbr_bundle: PbrBundle,
    acceleration: Acceleration,
    velocity: Velocity,
    mass: Mass,
}

fn circular_orbit_velocity(
    orbiting_position: Vec3,
    main_position: Vec3,
    main_mu: f32,
    axis: Vec3,
) -> Vec3 {
    let distance = main_position - orbiting_position;

    distance.cross(axis).normalize() * (main_mu / distance.length()).sqrt()
}
