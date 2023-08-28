use crate::{SceneData, SimulationScene};
use bevy::{
    ecs::{
        entity::Entity,
        system::{EntityCommands, Res},
    },
    prelude::{AssetServer, Resource},
};

use crate::Spawnable;

#[derive(Resource)]
pub struct LoadedScene {
    scene: SimulationScene,
    entity: Option<Entity>,
}

impl LoadedScene {
    pub fn new<S>(scene: S) -> Self
    where
        S: SceneData + Send + Sync + 'static,
    {
        Self {
            scene: Box::new(scene),
            entity: None,
        }
    }

    pub fn load(&mut self, scene: SimulationScene) {
        self.scene = scene;
    }

    pub fn loaded(&mut self) -> &SimulationScene {
        &self.scene
    }

    pub fn spawned(&mut self, entity: Entity) {
        self.entity.get_or_insert(entity);
    }

    pub fn _despawned(&mut self) {
        self.entity = None;
    }

    pub fn entity(&self) -> Entity {
        self.entity
            .unwrap_or_else(|| panic!("No entity for {}", self.scene))
    }

    pub fn get_entity(&self) -> Option<Entity> {
        self.entity
    }

    pub fn instance(&self, scene_commands: EntityCommands, asset_server: Res<AssetServer>) {
        self.scene.instance(scene_commands, asset_server)
    }

    pub fn spawnable(&self) -> Spawnable {
        self.scene.spawnable()
    }
}
