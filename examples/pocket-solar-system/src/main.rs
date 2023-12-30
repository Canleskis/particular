#![allow(clippy::too_many_arguments)]

mod camera;
use camera::*;

mod nbody;
use nbody::*;

mod orbit_prediction;
use orbit_prediction::*;

mod physics;
use physics::*;

mod selection;
use selection::*;

mod ui;
use ui::*;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    #[cfg(not(target_arch = "wasm32"))]
                    resolution: bevy::window::WindowResolution::new(1920.0, 1080.0),
                    fit_canvas_to_parent: true,
                    prevent_default_event_handling: false,
                    canvas: Some("#app".to_owned()),
                    ..default()
                }),
                ..default()
            }),
            CameraPlugin,
            PhysicsPlugin,
            ParticularPlugin,
            OrbitPredictionPlugin,
            SelectionPlugin,
            UiPlugin,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(AmbientLight {
            color: Color::NONE,
            brightness: 0.0,
        })
        .insert_resource(PhysicsSettings::delta_time(1.0 / 60.0))
        .add_systems(Startup, setup_scene)
        .add_systems(First, add_materials)
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut event_writer: EventWriter<ComputePredictionEvent>,
    physics: Res<PhysicsSettings>,
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
        bevy::core_pipeline::bloom::BloomSettings {
            intensity: 0.15,
            ..default()
        },
    ));

    let star_color = Color::rgb(1.0, 1.0, 0.9);
    let star = BodySetting {
        name: "Star",
        velocity: Vec3::new(-0.1826, -0.001, 0.0),
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
        name: "Planet",
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
        name: "Moon",
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
        name: "Comet",
        velocity: Vec3::new(2.8, 0.15, 0.4),
        position: Vec3::new(-200.0, 138.0, -18.0),
        mu: 0.000,
        radius: 0.1,
        material: StandardMaterial {
            base_color: Color::rgb(0.3, 0.3, 0.3),
            ..default()
        },
    };

    let mut star_bundle = BodyBundle::new(star);
    star_bundle.prediction_bundle.draw.steps = Some(0);
    let star = commands.spawn((star_bundle, Selected)).id();

    let mut planet_bundle = BodyBundle::new(planet);
    planet_bundle.prediction_bundle.draw.reference = Some(star);
    let planet = commands.spawn(planet_bundle).id();

    let mut moon_bundle = BodyBundle::new(moon);
    moon_bundle.prediction_bundle.draw.reference = Some(planet);
    commands.spawn(moon_bundle);

    let mut comet_bundle = BodyBundle::new(comet);
    comet_bundle.prediction_bundle.draw.reference = Some(star);
    commands.spawn(comet_bundle);

    commands.insert_resource(Followed(Some(star)));

    event_writer.send(ComputePredictionEvent {
        steps: physics.steps_per_second() * 60 * 5,
    });
}

#[derive(Component, Clone)]
pub struct BodyMaterial {
    pub mesh: Mesh,
    pub material: StandardMaterial,
}

impl Default for BodyMaterial {
    fn default() -> Self {
        Self {
            mesh: shape::Cube { size: 10.0 }.into(),
            material: StandardMaterial::default(),
        }
    }
}

fn add_materials(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    query: Query<(Entity, &BodyMaterial), Added<BodyMaterial>>,
) {
    for (entity, material) in &query {
        let mut cmds = commands.entity(entity);

        let BodyMaterial { mesh, material } = material.clone();
        if material.emissive != Color::BLACK {
            cmds.with_children(|child| {
                child.spawn(PointLightBundle {
                    point_light: PointLight {
                        color: material.emissive,
                        intensity: 5E4,
                        range: 2E3,
                        shadows_enabled: true,
                        ..default()
                    },
                    transform: Transform::from_xyz(0.0, 0.0, 0.0),
                    ..default()
                });
            });
        }

        cmds.insert(PbrBundle {
            mesh: meshes.add(mesh),
            material: materials.add(material),
            ..default()
        });
    }
}

#[derive(Bundle, Default)]
pub struct ParticleBundle {
    pub interolated: Interpolated,
    pub acceleration: Acceleration,
    pub velocity: Velocity,
    pub position: Position,
    pub mass: Mass,
}

#[derive(Bundle, Default)]
pub struct BodyBundle {
    pub name: Name,
    pub labelled: Labelled,
    pub can_select: CanSelect,
    pub can_follow: CanFollow,
    pub body_material: BodyMaterial,
    pub particle_bundle: ParticleBundle,
    pub prediction_bundle: PredictionBundle,
}

#[derive(Default, Clone)]
pub struct BodySetting {
    name: &'static str,
    velocity: Vec3,
    position: Vec3,
    mu: f32,
    radius: f32,
    material: StandardMaterial,
}

impl BodySetting {
    fn orbiting(mut self, orbiting: &Self, axis: Vec3) -> Self {
        let distance = self.position - orbiting.position;

        self.velocity = distance.cross(axis).normalize()
            * ((self.mu + orbiting.mu) / distance.length()).sqrt()
            + orbiting.velocity;

        self
    }
}

impl BodyBundle {
    pub fn new(setting: BodySetting) -> Self {
        Self {
            name: Name::new(setting.name),
            labelled: Labelled {
                style: TextStyle {
                    font_size: 6.0 * (1000.0 * setting.radius).log10(),
                    color: Color::GRAY,
                    ..default()
                },
                offset: Vec2::splat(setting.radius) * 1.1,
            },
            can_select: CanSelect {
                radius: setting.radius,
            },
            can_follow: CanFollow {
                min_camera_distance: setting.radius * 3.0,
            },
            particle_bundle: ParticleBundle {
                mass: Mass(setting.mu),
                velocity: Velocity(setting.velocity),
                position: Position(setting.position),
                ..default()
            },
            prediction_bundle: PredictionBundle {
                draw: PredictionDraw {
                    color: setting.material.base_color,
                    ..default()
                },
                ..default()
            },
            body_material: BodyMaterial {
                mesh: shape::UVSphere {
                    radius: setting.radius,
                    ..default()
                }
                .into(),
                material: setting.material,
            },
        }
    }
}

pub fn format_duration(duration: std::time::Duration, precision: usize) -> String {
    humantime::format_duration(duration)
        .to_string()
        .split_inclusive(' ')
        .take(precision)
        .collect::<String>()
}
