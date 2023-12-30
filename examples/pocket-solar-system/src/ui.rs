use std::time::Duration;

use crate::{
    format_duration, ComputePredictionEvent, ElapsedPhysicsTime, Followed, PhysicsSettings,
    PhysicsTime, PredictionDraw, PredictionState, ResetPredictionEvent, Selected,
};

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts, EguiPlugin};

#[derive(Component, Default)]
pub struct Labelled {
    pub style: TextStyle,
    pub offset: Vec2,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((EguiPlugin, bevy::diagnostic::FrameTimeDiagnosticsPlugin))
            .add_systems(PostStartup, setup_egui)
            .add_systems(First, spawn_labels)
            .add_systems(
                Update,
                (
                    windows_selection,
                    window_simulation,
                    update_labels_position,
                    update_labels_color,
                ),
            );
    }
}

fn setup_egui(mut ctxs: EguiContexts) {
    ctxs.ctx_mut().set_visuals(egui::Visuals {
        window_fill: egui::Color32::from_rgba_premultiplied(27, 27, 27, 225),
        window_stroke: egui::Stroke::NONE,
        ..egui::Visuals::dark()
    });
}

fn window_simulation(
    mut ctxs: EguiContexts,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
    elapsed_time: Res<ElapsedPhysicsTime>,
    mut physics: ResMut<PhysicsSettings>,
    mut physics_time: ResMut<PhysicsTime>,
    mut compute_event: EventWriter<ComputePredictionEvent>,
    mut reset_event: EventWriter<ResetPredictionEvent>,
    prediction_state: Query<&PredictionState>,
    mut event: Local<ComputePredictionEvent>,
) {
    let dt = physics.delta_time;
    let prediction_duration = Duration::from_secs_f32(
        prediction_state
            .iter()
            .next()
            .map(|state| state.positions.len() as f32 * dt)
            .unwrap_or(0.0),
    );

    egui::Window::new("Simulation settings")
        .default_width(255.0)
        .resizable(false)
        .anchor(egui::Align2::RIGHT_TOP, [0.0, 0.0])
        .show(ctxs.ctx_mut(), |ui| {
            if let Some(fps) = diagnostics.get(bevy::diagnostic::FrameTimeDiagnosticsPlugin::FPS) {
                if let Some(value) = fps.smoothed() {
                    ui.horizontal(|ui| {
                        ui.label("FPS:");
                        ui.label(format!("{value:.2}"));
                    });
                }
            }

            ui.horizontal(|ui| {
                ui.label("Time elapsed:");
                ui.label(format_duration(**elapsed_time, 3));
            });

            ui.horizontal(|ui| {
                ui.label("Time scale:");
                ui.add(egui::Slider::new(&mut physics.time_scale, 0.05..=100.0).logarithmic(true));
            });

            ui.checkbox(&mut physics_time.paused, "Paused");

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Total prediction duration:");
                ui.label(format_duration(prediction_duration, 2));
            });

            ui.horizontal(|ui| {
                ui.label("Predict for:");
                ui.add(
                    egui::Slider::new_duration(
                        &mut event.steps,
                        physics.steps_per_second()..=physics.steps_per_second() * 3600 * 5,
                        dt,
                        2,
                    )
                    .logarithmic(true),
                );
            });

            ui.horizontal(|ui| {
                if ui.button("Start").clicked() {
                    compute_event.send(*event)
                }

                if ui.button("Reset").clicked() {
                    reset_event.send(ResetPredictionEvent)
                }
            });
        });
}

fn windows_selection(
    mut ctxs: EguiContexts,
    physics: Res<PhysicsSettings>,
    mut followed: ResMut<Followed>,
    mut query_prediction: Query<(&PredictionState, &mut PredictionDraw)>,
    query_selection: Query<(Option<Entity>, &Name, bevy::ecs::query::Has<Selected>)>,
) {
    let dt = physics.delta_time;

    for (entity, selected_name, is_selected) in &query_selection {
        if !is_selected {
            continue;
        }

        egui::Window::new(selected_name.to_string())
            .default_width(245.0)
            .resizable(false)
            .collapsible(false)
            .anchor(egui::Align2::LEFT_TOP, [0.0, 0.0])
            .show(ctxs.ctx_mut(), |ui| {
                ui.heading("Camera");

                if ui.button("Follow").clicked() {
                    **followed = entity;
                }
                ui.separator();

                ui.heading("Prediction");

                let prediction = entity.and_then(|e| query_prediction.get_mut(e).ok());
                if let Some((state, mut draw)) = prediction {
                    let none = (None, &Name::new("None"), false);
                    let reference = draw
                        .reference
                        .and_then(|e| query_selection.get(e).ok())
                        .unwrap_or(none);

                    egui::ComboBox::from_label("Reference")
                        .selected_text(reference.1.to_string())
                        .show_ui(ui, |ui| {
                            [none]
                                .into_iter()
                                .chain(query_selection.iter().filter(|(s, ..)| *s != entity))
                                .for_each(|(entity, name, ..)| {
                                    ui.selectable_value(
                                        &mut draw.reference,
                                        entity,
                                        name.to_string(),
                                    );
                                });
                        });

                    if ui.checkbox(&mut draw.steps.is_none(), "Draw all").changed() {
                        if draw.steps.is_none() {
                            draw.steps.replace(state.positions.len());
                        } else {
                            draw.steps.take();
                        }
                    }

                    ui.horizontal(|ui| {
                        ui.label("Resolution:");
                        ui.add(egui::Slider::new(&mut draw.resolution, 1..=50));
                    });

                    ui.add_enabled_ui(draw.steps.is_some(), |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Draw for:");
                            ui.add(egui::Slider::new_duration(
                                draw.steps.as_mut().unwrap_or(&mut state.positions.len()),
                                0..=state.positions.len(),
                                dt,
                                2,
                            ));
                        });
                    });
                }
            });
    }
}

#[derive(Component, Deref, DerefMut)]
struct LabelEntity(Entity);

fn spawn_labels(
    mut commands: Commands,
    query_labelled: Query<(Entity, &Name, &Labelled), Added<Labelled>>,
) {
    for (entity, name, labelled) in &query_labelled {
        let id = commands
            .spawn(TextBundle::from_section(
                name.to_string(),
                labelled.style.clone(),
            ))
            .id();

        commands.entity(entity).insert(LabelEntity(id));
    }
}

fn update_labels_position(
    query_camera: Query<(&Camera, &GlobalTransform)>,
    query_labelled: Query<(&LabelEntity, &Labelled, &GlobalTransform)>,
    mut query_labels: Query<(&mut Style, &Node)>,
) {
    let (camera, camera_transform) = query_camera.single();

    for (entity, label, transform) in &query_labelled {
        let Ok((mut style, node)) = query_labels.get_mut(**entity) else {
            continue;
        };

        let rotation_matrix = Mat3::from_quat(camera_transform.to_scale_rotation_translation().1);
        let viewport_position = camera
            .world_to_viewport(
                camera_transform,
                transform.translation() + rotation_matrix.mul_vec3(label.offset.extend(0.0)),
            )
            .map(|position| position - node.size() / 2.0);

        if let Some(viewport_position) = viewport_position {
            style.position_type = PositionType::Absolute;
            style.left = Val::Px(viewport_position.x);
            style.top = Val::Px(viewport_position.y);
            style.display = Display::Flex;
        } else {
            style.display = Display::None;
        }
    }
}

fn update_labels_color(
    mut query_labels: Query<&mut Text>,
    query_labelled: Query<&LabelEntity>,
    selected: Query<Entity, Added<Selected>>,
    mut deselected: RemovedComponents<Selected>,
) {
    let mut set_label_color = |entity, color| {
        if let Ok(mut text) = query_labelled
            .get(entity)
            .and_then(|e| query_labels.get_mut(**e))
        {
            text.sections[0].style.color = color;
        }
    };

    for entity in deselected.read() {
        set_label_color(entity, Color::GRAY);
    }

    for entity in selected.iter() {
        set_label_color(entity, Color::rgb(0.75, 0.0, 0.0));
    }
}

trait DurationSlider<'a> {
    fn new_duration<Num: egui::emath::Numeric>(
        value: &'a mut Num,
        range: std::ops::RangeInclusive<Num>,
        delta: f32,
        precision: usize,
    ) -> egui::Slider<'a> {
        egui::Slider::new(value, range).custom_formatter(move |s, _| {
            format_duration(Duration::from_secs_f32(s as f32 * delta), precision)
        })
    }
}

impl DurationSlider<'_> for egui::Slider<'_> {}
