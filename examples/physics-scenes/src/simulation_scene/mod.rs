mod loaded_scene;
mod scene_data;
mod spawnable;

pub use loaded_scene::LoadedScene;
pub use scene_data::{Empty, SceneData, SimulationScene};
pub use spawnable::Spawnable;

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

#[derive(Resource, Deref, DerefMut)]
pub struct SceneCollection(pub Vec<SimulationScene>);

impl SceneCollection {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn _with<S>(mut self, scene: S) -> Self
    where
        S: SceneData + Send + Sync + 'static,
    {
        self.push(Box::new(scene));
        self
    }

    pub fn _add<S>(&mut self, scene: S)
    where
        S: SceneData + Send + Sync + 'static,
    {
        self.push(Box::new(scene));
    }

    pub fn with_scene<S>(mut self) -> Self
    where
        S: SceneData + Default + Send + Sync + 'static,
    {
        self.push(Box::<S>::default());
        self
    }
}

pub struct SimulationScenePlugin;

impl Plugin for SimulationScenePlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(LoadedScene::new(Empty {}))
            .add_systems(PreUpdate, scene_cleanup_and_reload)
            .add_systems(Update, show_ui);
    }
}

pub fn scene_cleanup_and_reload(
    mut commands: Commands,
    mut scene: ResMut<LoadedScene>,
    asset_server: Res<AssetServer>,
) {
    if scene.is_changed() {
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
    mut egui_ctx: EguiContexts,
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
