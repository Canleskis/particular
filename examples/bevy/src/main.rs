mod camera;
use camera::*;

mod nbody;
use nbody::*;

mod orbit_prediction;
use orbit_prediction::*;

mod physics;
use physics::*;

mod ui;
use ui::*;

use bevy::{core_pipeline::bloom::BloomSettings, prelude::*};

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            CameraPlugin,
            PhysicsPlugin,
            ParticularPlugin,
            OrbitPredictionPlugin,
            UiPlugin,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(SelectableEntities::default())
        .insert_resource(PhysicsSettings::default())
        .add_systems(Startup, setup_scene)
        .add_systems(PostStartup, find_selectable_entities)
        .add_systems(PreUpdate, switch_selected_entity)
        .run();
}

#[derive(Default, Clone)]
struct BodySetting {
    velocity: Vec3,
    position: Vec3,
    mu: f32,
    radius: f32,
    material: StandardMaterial,
}

impl BodySetting {
    fn orbiting(mut self, orbiting: &Self, axis: Vec3) -> Self {
        self.velocity =
            circular_orbit_velocity(orbiting.position, orbiting.mu, self.position, self.mu, axis)
                + orbiting.velocity;

        self
    }
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
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
    ));

    let star_color = Color::rgb(1.0, 1.0, 0.9);
    let star = BodySetting {
        mu: 5E3,
        radius: 8.0,
        material: StandardMaterial {
            base_color: star_color,
            emissive: star_color * 2.0,
            ..default()
        },
        ..default()
    };

    let planet = BodySetting {
        position: Vec3::new(0.0, 60.0, 0.0),
        mu: 100.0,
        radius: 2.0,
        material: StandardMaterial {
            base_color: Color::rgb(0.0, 0.6, 1.0),
            ..default()
        },
        ..default()
    }
    .orbiting(&star, Vec3::Z);

    let moon = BodySetting {
        position: planet.position + Vec3::new(4.5, 0.0, 0.0),
        mu: 1.0,
        radius: 0.6,
        material: StandardMaterial {
            base_color: Color::rgb(0.6, 0.4, 0.1),
            ..default()
        },
        ..default()
    }
    .orbiting(&planet, Vec3::new(0.0, 0.5, -1.0));

    let comet = BodySetting {
        velocity: Vec3::new(3.0, 0.2, 0.4),
        position: Vec3::new(-200.0, 138.0, -18.0),
        mu: 0.0,
        radius: 0.1,
        material: StandardMaterial {
            base_color: Color::rgb(0.3, 0.3, 0.3),
            ..default()
        },
    };

    let bodies = vec![
        ("Star", star),
        ("Planet", planet),
        ("Moon", moon),
        ("Comet", comet),
    ];

    for (name, body) in bodies {
        let light_entity = (body.material.emissive != Color::BLACK).then(|| {
            commands
                .spawn(PointLightBundle {
                    point_light: PointLight {
                        color: body.material.emissive,
                        intensity: 5E4,
                        range: 2E3,
                        shadows_enabled: true,
                        ..default()
                    },
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    ..default()
                })
                .id()
        });

        let mut body_entity = commands.spawn((
            Name::new(name),
            PredictionState {
                color: body.material.base_color,
                ..default()
            },
            ParticleBundle {
                pbr_bundle: PbrBundle {
                    mesh: meshes.add(
                        shape::UVSphere {
                            radius: body.radius,
                            ..default()
                        }
                        .into(),
                    ),
                    material: materials.add(body.material),
                    transform: Transform::from_translation(body.position),
                    ..default()
                },
                mass: Mass(body.mu),
                velocity: Velocity {
                    linear: body.velocity,
                },
                ..default()
            },
        ));

        if let Some(light_entity) = light_entity {
            body_entity.add_child(light_entity);
        }
    }
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
    orbiting_mu: f32,
    main_position: Vec3,
    main_mu: f32,
    axis: Vec3,
) -> Vec3 {
    let distance = main_position - orbiting_position;

    distance.cross(axis).normalize() * ((main_mu + orbiting_mu) / distance.length()).sqrt()
}
