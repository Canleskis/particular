use bevy::{ecs::schedule::ScheduleLabel, prelude::*};
use bevy_rapier2d::prelude::*;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct SyncRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct StepRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PostRapierSchedule;

fn run_physics_schedule(world: &mut World) {
    let config = *world.resource::<RapierConfiguration>();
    let TimestepMode::Fixed { dt, .. } = world.resource::<RapierConfiguration>().timestep_mode
    else {
        return;
    };

    if !config.physics_pipeline_active {
        return;
    }

    let delta = world.resource::<Time>().delta_seconds();
    let diff = &mut world.resource_mut::<SimulationToRenderTime>().diff;

    *diff += delta;

    let is_physics_step = *diff >= dt;

    if is_physics_step {
        *diff = 0.0;

        world.run_schedule(PreRapierSchedule);
    }

    world.run_schedule(SyncRapierSchedule);

    if is_physics_step {
        world.run_schedule(StepRapierSchedule);
        world.run_schedule(PostRapierSchedule);
    }
}

pub struct CustomRapierSchedule;

impl Plugin for CustomRapierSchedule {
    fn build(&self, app: &mut App) {
        app.add_systems(PreUpdate, run_physics_schedule);

        let pre_schedule = Schedule::new(PreRapierSchedule);

        let mut sync_schedule = Schedule::new(SyncRapierSchedule);
        sync_schedule.configure_sets(PhysicsSet::SyncBackend);
        sync_schedule.add_systems(
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::SyncBackend)
                .in_set(PhysicsSet::SyncBackend),
        );

        let mut step_schedule = Schedule::new(StepRapierSchedule);
        step_schedule.configure_sets((PhysicsSet::StepSimulation, PhysicsSet::Writeback).chain());
        step_schedule.add_systems((
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::StepSimulation)
                .in_set(PhysicsSet::StepSimulation),
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::Writeback)
                .in_set(PhysicsSet::Writeback),
        ));

        let post_schedule = Schedule::new(PostRapierSchedule);

        app.add_schedule(pre_schedule)
            .add_schedule(sync_schedule)
            .add_schedule(step_schedule)
            .add_schedule(post_schedule);

        app.insert_resource(RapierConfiguration {
            gravity: Vec2::ZERO,
            timestep_mode: TimestepMode::Fixed {
                dt: crate::DT,
                substeps: 1,
            },
            ..default()
        });
    }
}
