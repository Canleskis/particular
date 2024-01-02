use bevy::{ecs::schedule::ScheduleLabel, prelude::*};
use bevy_rapier2d::prelude::*;

#[derive(Default, Resource)]
pub struct RealWorldTick(pub f32);

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct SyncRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct StepRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PostRapierSchedule;

pub fn time_sync(
    time: Res<Time>,
    config: Res<RapierConfiguration>,
    mut physics_time: ResMut<RealWorldTick>,
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
    }

    physics_time.0 = time.delta_seconds().min(dt);
    sim_to_render_time.diff += time.delta_seconds();
}

pub fn physics_step(world: &mut World) {
    let config = world.resource::<RapierConfiguration>();
    let sim_to_render_time = world.resource::<SimulationToRenderTime>();

    let TimestepMode::Fixed { dt, .. } = config.timestep_mode else {
        return;
    };

    let is_physics_step = config.physics_pipeline_active && sim_to_render_time.diff > dt;

    if is_physics_step {
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
        app.init_resource::<RealWorldTick>()
            .add_systems(First, time_sync)
            .add_systems(PreUpdate, physics_step);

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
