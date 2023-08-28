use crate::LoadedScene;
use bevy::{
    ecs::{
        change_detection::DetectChanges,
        schedule::ShouldRun,
        system::{Commands, Local, Res, ResMut},
    },
    hierarchy::DespawnRecursiveExt,
    prelude::AssetServer,
    scene::SceneBundle,
};
use bevy_egui::egui;
use bevy_prototype_debug_lines::DebugLines;

use super::SceneCollection;

// Cannot use this as run criteria as changes done to `LoadedScene` in systems with this run criteria are also detected.
pub fn _scene_changed(scene: Res<LoadedScene>) -> ShouldRun {
    if scene.is_changed() {
        ShouldRun::Yes
    } else {
        ShouldRun::No
    }
}

pub fn scene_cleanup_and_reload(
    mut commands: Commands,
    mut lines: ResMut<DebugLines>,
    mut scene: ResMut<LoadedScene>,
    asset_server: Res<AssetServer>,
) {
    if scene.is_changed() {
        *lines = DebugLines::default();

        let entity_commands = if let Some(entity) = scene.get_entity() {
            let mut commands = commands.entity(entity);
            commands.despawn_descendants();
            commands
        } else {
            let commands = commands.spawn(SceneBundle::default());
            scene.spawned(commands.id());
            commands
        };

        scene.instance(entity_commands, asset_server);
    }
}

pub fn show_ui(
    mut egui_ctx: ResMut<bevy_egui::EguiContext>,
    mut scenes: ResMut<SceneCollection>,
    mut scene: ResMut<LoadedScene>,
    mut current: Local<Option<usize>>,
) {
    if let Some(selected) = current.as_mut() {
        egui::Window::new("Simulation").show(egui_ctx.ctx_mut(), |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                egui::ComboBox::from_label("")
                    .show_index(ui, selected, scenes.len(), |i| scenes[i].to_string());

                if ui.button("New").clicked() {
                    let selected_scene = scenes[*selected].clone();
                    scene.load(selected_scene);
                }
            });

            scenes[*selected].show_ui(ui);
        });
    } else {
        *current = scenes.iter().position(|s| s == scene.loaded());
    }
}
