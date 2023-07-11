mod camera;
use camera::*;

mod physics;
use physics::*;

mod nbody;
use nbody::*;

mod orbit_prediction;
use orbit_prediction::*;

use bevy::{core_pipeline::bloom::BloomSettings, prelude::*};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            CameraPlugin,
            PhysicsPlugin,
            ParticularPlugin,
            OrbitPredictionPlugin,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(FixedTime::default())
        .insert_resource(SelectableEntities::default())
        .add_systems(Startup, setup_scene)
        .add_systems(PostStartup, find_selectable_entities)
        .add_systems(PreUpdate, switch_selected_entity)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let main_color = Color::rgb(1.0, 1.0, 0.9);

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
            OrbitCamera::default(),
            BloomSettings {
                intensity: 0.15,
                ..default()
            },
        ))
        .id();

    let light = commands
        .spawn(PointLightBundle {
            point_light: PointLight {
                color: main_color,
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

    let tertiary_position = secondary_position + Vec3::new(0.0, 4.6, 4.6);
    let tertiary_mu = 0.0;

    commands
        .spawn((
            ParticleBundle {
                pbr_bundle: PbrBundle {
                    mesh: meshes.add(
                        shape::UVSphere {
                            radius: 8.0,
                            ..default()
                        }
                        .into(),
                    ),
                    material: materials.add(StandardMaterial {
                        emissive: main_color * 2.0,
                        ..default()
                    }),
                    transform: Transform::from_translation(Vec3::ZERO),
                    ..default()
                },
                mass: Mass(main_mu),
                ..default()
            },
            PredictionState::default(),
        ))
        .add_child(light)
        .add_child(camera);

    let secondary_velocity =
        circular_orbit_velocity(secondary_position, main_position, main_mu, Vec3::Z);

    commands.spawn((
        ParticleBundle {
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
        },
        PredictionState::default(),
    ));

    let tertiary_velocity = circular_orbit_velocity(
        tertiary_position,
        secondary_position,
        secondary_mu,
        Vec3::new(1.0, 1.0, 0.0),
    ) + secondary_velocity;

    commands.spawn((
        ParticleBundle {
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
        },
        PredictionState::default(),
    ));
}

#[derive(Resource, Default, Deref, DerefMut)]
pub struct SelectableEntities(Vec<Entity>);

impl SelectableEntities {
    pub fn selected(&self) -> Option<Entity> {
        self.first().copied()
    }
}

fn find_selectable_entities(
    mut selectable_entites: ResMut<SelectableEntities>,
    query_bodies: Query<Entity, With<Mass>>,
) {
    let mut bodies: Vec<_> = query_bodies.iter().collect();
    bodies.sort();

    selectable_entites.0 = bodies;
}

fn switch_selected_entity(
    input: Res<Input<KeyCode>>,
    mut selectable_entites: ResMut<SelectableEntities>,
) {
    if input.just_pressed(KeyCode::Space) {
        selectable_entites.rotate_left(1);
    }
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
