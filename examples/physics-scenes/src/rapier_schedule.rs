use bevy::{ecs::schedule::ShouldRun, prelude::*};
use bevy_rapier2d::prelude::*;

use crate::DT;

#[derive(Resource)]
pub struct StepCount(pub usize);

pub fn time_sync(
    time: Res<Time>,
    mut step_count: ResMut<StepCount>,
    config: Res<RapierConfiguration>,
    mut sim_to_render_time: ResMut<SimulationToRenderTime>,
) {
    if !config.physics_pipeline_active {
        return;
    }

    let TimestepMode::Fixed { dt, .. } = config.timestep_mode else {
        return;
    };

    if sim_to_render_time.diff > dt {
        sim_to_render_time.diff = 0.0;
        step_count.0 += 1;
    }
    sim_to_render_time.diff += time.delta_seconds();
}

pub fn physics_step(
    config: Res<RapierConfiguration>,
    sim_to_render_time: Res<SimulationToRenderTime>,
) -> ShouldRun {
    let TimestepMode::Fixed { dt, .. } = config.timestep_mode else {
        return ShouldRun::Yes;
    };

    if config.physics_pipeline_active && sim_to_render_time.diff > dt {
        ShouldRun::Yes
    } else {
        ShouldRun::No
    }
}

pub struct CustomRapierSchedule;

impl Plugin for CustomRapierSchedule {
    fn build(&self, app: &mut App) {
        app.add_system_to_stage(CoreStage::First, time_sync);
        app.add_stage_after(
            CoreStage::Update,
            PhysicsStages::SyncBackend,
            SystemStage::parallel()
                .with_run_criteria(physics_step)
                .with_system_set(RapierPhysicsPlugin::<()>::get_systems(
                    PhysicsStages::SyncBackend,
                )),
        );
        app.add_stage_after(
            PhysicsStages::SyncBackend,
            PhysicsStages::StepSimulation,
            SystemStage::parallel()
                .with_run_criteria(physics_step)
                .with_system_set(RapierPhysicsPlugin::<()>::get_systems(
                    PhysicsStages::StepSimulation,
                )),
        );
        app.add_stage_after(
            PhysicsStages::StepSimulation,
            PhysicsStages::Writeback,
            SystemStage::parallel()
                .with_run_criteria(physics_step)
                .with_system_set(RapierPhysicsPlugin::<()>::get_systems(
                    PhysicsStages::Writeback,
                )),
        );

        // NOTE: we run sync_removals at the end of the frame, too, in order to make sure we don’t miss any `RemovedComponents`.
        app.add_stage_before(
            CoreStage::Last,
            PhysicsStages::DetectDespawn,
            SystemStage::parallel().with_system_set(RapierPhysicsPlugin::<()>::get_systems(
                PhysicsStages::DetectDespawn,
            )),
        );

        app.insert_resource(StepCount(0))
            .insert_resource(RapierConfiguration {
                gravity: Vec2::ZERO,
                timestep_mode: TimestepMode::Fixed {
                    dt: DT,
                    substeps: 1,
                },
                ..default()
            });
    }
}
