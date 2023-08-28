use bevy::prelude::*;
use bevy_inspector_egui::Inspectable;
use bevy_prototype_debug_lines::{DebugLines, DebugLinesPlugin};
use bevy_rapier2d::prelude::{RapierConfiguration, TimestepMode};

use crate::rapier_schedule::physics_step;

pub struct TrailsPlugin;

impl Plugin for TrailsPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(DebugLinesPlugin::default())
            .add_system_set_to_stage(
                CoreStage::PostUpdate,
                SystemSet::new()
                    .with_run_criteria(physics_step)
                    .with_system(draw_trails),
            );
    }
}

#[derive(Component, Inspectable)]
pub struct Trail {
    pub length: f32,
    pub resolution: usize,
    cached: Option<(Vec3, usize)>,
}

impl Trail {
    pub fn new(length: f32, resolution: usize) -> Self {
        Self {
            length,
            resolution,
            cached: None,
        }
    }
}

fn draw_trails(
    config: Res<RapierConfiguration>,
    mut lines: ResMut<DebugLines>,
    mut query: Query<(&GlobalTransform, &mut Trail)>,
) {
    let TimestepMode::Fixed { dt, .. } = config.timestep_mode else {
        return;
    };
    for (transform, mut trail) in query.iter_mut() {
        let (resolution, length) = (trail.resolution, trail.length);
        if let Some((last_position, last_iteration)) = &mut trail.cached {
            if *last_iteration == resolution {
                lines.line_colored(*last_position, transform.translation(), length, Color::RED);
                *last_position = transform.translation();
                *last_iteration = 0;
            } else {
                lines.line_colored(*last_position, transform.translation(), dt, Color::RED);
                *last_iteration += 1;
            }
        } else {
            trail.cached = Some((transform.translation(), 0));
        }
    }
}
