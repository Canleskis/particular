mod loaded_scene;
mod scene_data;
mod spawnable;
mod systems;

use std::ops::{Deref, DerefMut};

pub use loaded_scene::LoadedScene;
pub use scene_data::{Empty, SceneData, SimulationScene};
pub use spawnable::Spawnable;

use bevy::{
    app::{App, CoreStage, Plugin},
    prelude::Resource,
};

#[derive(Resource)]
pub struct SceneCollection(pub Vec<SimulationScene>);

impl Deref for SceneCollection {
    type Target = Vec<SimulationScene>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SceneCollection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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
            .add_system_to_stage(CoreStage::PreUpdate, systems::scene_cleanup_and_reload)
            .add_system(systems::show_ui);
    }
}
