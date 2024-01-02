mod nbody;
mod rapier_schedule;
mod simulation_scene;
mod simulation_scenes;
mod trails;

use std::f32::consts::PI;

use nbody::{ParticularPlugin, PointMass};
use rapier_schedule::CustomRapierSchedule;
use simulation_scene::*;
use simulation_scenes::{DoubleOval, Figure8, Orbits, TernaryOrbit};
use trails::{Trail, TrailsPlugin};

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::{prelude::*, window::PresentMode};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use bevy_pancam::{PanCam, PanCamPlugin};
use bevy_rapier2d::prelude::*;

const G: f32 = 1000.0;
const DT: f32 = 1.0 / 60.0;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: PresentMode::AutoNoVsync,
                    fit_canvas_to_parent: true,
                    canvas: Some("#app".to_owned()),
                    ..default()
                }),
                ..default()
            }),
            RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(1.0)
                .with_default_system_setup(false),
            CustomRapierSchedule,
            FrameTimeDiagnosticsPlugin,
            EguiPlugin,
            PanCamPlugin,
            TrailsPlugin,
            ParticularPlugin,
            SimulationScenePlugin,
        ))
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(
            SceneCollection::new()
                .with_scene::<Empty>()
                .with_scene::<Orbits>()
                .with_scene::<Figure8>()
                .with_scene::<DoubleOval>()
                .with_scene::<TernaryOrbit>(),
        )
        .insert_resource(LoadedScene::new(Orbits::default()))
        .init_resource::<BodyInfo>()
        .add_state::<SimulationState>()
        .add_systems(Startup, (spawn_camera, setup_ui_fps))
        .insert_resource(Time::<Fixed>::from_hz(2.0))
        .add_systems(FixedUpdate, update_ui_fps)
        .add_systems(Update, (place_body, body_info_window, pause_resume))
        .add_systems(OnEnter(SimulationState::Paused), pause_physics)
        .add_systems(OnExit(SimulationState::Paused), resume_physics)
        .run();
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera2dBundle::default(),
        PanCam {
            grab_buttons: vec![MouseButton::Right, MouseButton::Middle],
            ..default()
        },
    ));
}

#[derive(Component)]
struct FpsText;

fn setup_ui_fps(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands
        .spawn(
            TextBundle::from_sections([
                TextSection::new(
                    "FPS: ",
                    TextStyle {
                        font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                        font_size: 20.0,
                        color: Color::GRAY,
                    },
                ),
                TextSection::from_style(TextStyle {
                    font: asset_server.load("fonts/FiraSans-Bold.ttf"),
                    font_size: 20.0,
                    color: Color::GRAY,
                }),
            ])
            .with_style(Style {
                position_type: PositionType::Absolute,
                top: Val::Px(5.0),
                right: Val::Px(10.0),
                ..default()
            }),
        )
        .insert(FpsText);
}

fn update_ui_fps(
    mut query_text: Query<&mut Text, With<FpsText>>,
    diagnostic: Res<DiagnosticsStore>,
) {
    let fps = diagnostic
        .get(FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.average());
    if let Some(fps) = fps {
        for mut text in &mut query_text {
            text.sections[1].value = format!("{fps:.1}");
        }
    }
}

#[derive(States, Default, Debug, Clone, Eq, PartialEq, Hash)]
enum SimulationState {
    #[default]
    Running,
    Paused,
}

fn pause_resume(
    keys: Res<Input<KeyCode>>,
    state: Res<State<SimulationState>>,
    mut next_state: ResMut<NextState<SimulationState>>,
) {
    if keys.just_pressed(KeyCode::Space) {
        match state.get() {
            SimulationState::Running => {
                next_state.set(SimulationState::Paused);
            }
            SimulationState::Paused => {
                next_state.set(SimulationState::Running);
            }
        }
    }
}

fn pause_physics(mut physics: ResMut<RapierConfiguration>) {
    physics.physics_pipeline_active = false;
}

fn resume_physics(mut physics: ResMut<RapierConfiguration>) {
    physics.physics_pipeline_active = true;
}

#[derive(Resource)]
struct BodyInfo {
    mass: f32,
    with_mass: bool,
    with_trail: bool,
}

impl Default for BodyInfo {
    fn default() -> Self {
        Self {
            mass: 20.0,
            with_mass: true,
            with_trail: false,
        }
    }
}

fn body_info_window(
    mut egui_ctx: EguiContexts,
    mut body_info: ResMut<BodyInfo>,
    scene: Res<LoadedScene>,
) {
    egui::Window::new("Body spawner").show(egui_ctx.ctx_mut(), |ui| {
        ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
            if let Some((min_mass, max_mass)) = scene.spawnable().mass_range() {
                ui.add_enabled(
                    body_info.with_mass,
                    egui::Slider::new(&mut body_info.mass, min_mass..=max_mass),
                );

                ui.toggle_value(&mut body_info.with_mass, "Mass");
            }
        });

        ui.checkbox(&mut body_info.with_trail, "Draw trail");
    });
}

#[allow(clippy::too_many_arguments)]
fn place_body(
    mut commands: Commands,
    mut gizmos: Gizmos,
    mut egui_ctx: EguiContexts,
    body_info: Res<BodyInfo>,
    buttons: Res<Input<MouseButton>>,
    asset_server: Res<AssetServer>,
    scene: Res<LoadedScene>,
    query_window: Query<&Window>,
    query_camera: Query<(&GlobalTransform, &Camera)>,
    mut last_click: Local<Option<Vec2>>,
) {
    if egui_ctx.ctx_mut().is_pointer_over_area() {
        return;
    }

    let Ok(window) = query_window.get_single() else {
        return;
    };
    let Ok((camera_transform, camera)) = query_camera.get_single() else {
        return;
    };

    let Some(mouse_pos) = window
        .cursor_position()
        .and_then(|position| camera.viewport_to_world_2d(camera_transform, position))
    else {
        return;
    };

    if buttons.just_pressed(MouseButton::Left) {
        *last_click = Some(mouse_pos);
    }

    if buttons.just_released(MouseButton::Left) {
        if let Some(place_pos) = last_click.take() {
            let spawnable = scene.spawnable();
            let point_mass = if spawnable.is_massive() && body_info.with_mass {
                PointMass::HasGravity {
                    mass: body_info.mass,
                }
            } else {
                PointMass::AffectedByGravity
            };

            let mut entity = commands.entity(scene.entity());

            entity.with_children(|child| {
                let mut entity = child.spawn(BodyBundle::new(
                    place_pos,
                    Velocity::linear(place_pos - mouse_pos),
                    spawnable.density(),
                    point_mass,
                    Color::WHITE,
                    &asset_server,
                ));

                if body_info.with_trail {
                    entity.insert(Trail::new(20.0, 1));
                }
            });
        }
    }

    if let Some(place_pos) = *last_click {
        let scale = (mouse_pos.distance_squared(place_pos).powf(0.04) - 1.0).clamp(0.0, 1.0);
        gizmos.line(
            place_pos.extend(0.0),
            mouse_pos.extend(0.0),
            Color::rgb(scale, 1.0 - scale, 0.0),
        )
    }
}

#[derive(Bundle)]
struct BodyBundle {
    sprite_bundle: SpriteBundle,
    collider: Collider,
    mass_properties: ColliderMassProperties,
    rigidbody: RigidBody,
    velocity: Velocity,
    acceleration: ExternalForce,
    point_mass: PointMass,
    read_mass: ReadMassProperties,
}

impl BodyBundle {
    fn new(
        position: Vec2,
        velocity: Velocity,
        density: f32,
        point_mass: PointMass,
        color: Color,
        asset_server: &Res<AssetServer>,
    ) -> Self {
        let (mass, density) = match point_mass {
            PointMass::HasGravity { mass } => (mass, density),
            PointMass::AffectedByGravity => (1e-20, density * 1e-20),
        };

        let radius = (mass / (density * PI)).sqrt();

        Self {
            sprite_bundle: SpriteBundle {
                transform: Transform::from_translation(position.extend(0.0)),
                texture: asset_server.load("sprites/circle-sprite-300.png"),
                sprite: Sprite {
                    color,
                    custom_size: Some(Vec2::splat(radius * 2.0)),
                    ..default()
                },
                ..default()
            },
            collider: Collider::ball(radius),
            mass_properties: ColliderMassProperties::Density(density),
            rigidbody: RigidBody::Dynamic,
            velocity,
            acceleration: ExternalForce::default(),
            point_mass,
            read_mass: ReadMassProperties::default(),
        }
    }
}
