use std::fmt::Display;

use bevy::{
    ecs::system::{EntityCommands, Res},
    prelude::AssetServer,
};
use bevy_egui::egui::Ui;

use super::Spawnable;

pub type SimulationScene = Box<dyn SceneData + Send + Sync>;

pub trait SceneDataClone {
    fn clone_box(&self) -> SimulationScene;
}

impl<T: 'static + SceneData + Send + Sync + Clone> SceneDataClone for T {
    fn clone_box(&self) -> SimulationScene {
        Box::new(self.clone())
    }
}

pub trait SceneData: SceneDataClone + Display {
    fn instance(&self, scene_commands: EntityCommands, asset_server: Res<AssetServer>);

    fn show_ui(&mut self, ui: &mut Ui);

    fn spawnable(&self) -> Spawnable;
}

impl Clone for SimulationScene {
    fn clone(&self) -> SimulationScene {
        self.clone_box()
    }
}

impl PartialEq for SimulationScene {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

#[derive(Clone, Default)]
pub struct Empty;

impl Display for Empty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Empty")
    }
}

impl SceneData for Empty {
    fn instance(&self, _: EntityCommands, _: Res<AssetServer>) {}

    fn show_ui(&mut self, _: &mut Ui) {}

    fn spawnable(&self) -> Spawnable {
        Spawnable::Massive {
            min_mass: 1.0,
            max_mass: 100.0,
            density: 0.1,
        }
    }
}
