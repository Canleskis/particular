#[derive(Default, Resource)]
pub struct RealWorldTick(pub f32);

use bevy::{ecs::schedule::ScheduleLabel, prelude::*};
use bevy_rapier3d::prelude::*;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PreRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct SyncRapierSchedule;

#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
struct StepRapierSchedule;

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
    }
}

pub struct CustomRapierSchedule;

impl Plugin for CustomRapierSchedule {
    fn build(&self, app: &mut App) {
        app.init_resource::<RealWorldTick>()
            .add_systems(First, time_sync)
            .add_systems(PreUpdate, physics_step);

        let mut pre_schedule = Schedule::new();
        pre_schedule.add_systems(apply_accelerations);

        let mut sync_schedule = Schedule::new();
        sync_schedule
            .configure_sets((PhysicsSet::SyncBackend, PhysicsSet::SyncBackendFlush).chain());
        sync_schedule.add_systems((
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::SyncBackend)
                .in_set(PhysicsSet::SyncBackend),
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::SyncBackendFlush)
                .in_set(PhysicsSet::SyncBackendFlush),
        ));

        let mut step_schedule = Schedule::new();
        step_schedule.configure_sets((PhysicsSet::StepSimulation, PhysicsSet::Writeback).chain());
        step_schedule.add_systems((
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::StepSimulation)
                .in_set(PhysicsSet::StepSimulation),
            RapierPhysicsPlugin::<()>::get_systems(PhysicsSet::Writeback)
                .in_set(PhysicsSet::Writeback),
        ));

        app.add_schedule(PreRapierSchedule, pre_schedule)
            .add_schedule(SyncRapierSchedule, sync_schedule)
            .add_schedule(StepRapierSchedule, step_schedule);

        app.insert_resource(RapierConfiguration {
            gravity: Vec3::ZERO,
            timestep_mode: TimestepMode::Fixed {
                dt: crate::DT,
                substeps: 1,
            },
            ..default()
        });
    }
}

#[derive(Component, Debug, Default)]
pub struct Acceleration {
    pub linear: Vec3,
    pub angular: f32,
}

fn apply_accelerations(
    config: Res<RapierConfiguration>,
    mut query: Query<(&mut Acceleration, &mut Velocity)>,
) {
    let TimestepMode::Fixed { dt, .. } = config.timestep_mode else {
        return;
    };

    for (mut acceleration, mut velocity) in &mut query {
        velocity.linvel += acceleration.linear * dt;
        velocity.angvel += acceleration.angular * dt;

        acceleration.linear = Default::default();
        acceleration.angular = Default::default();
    }
}
