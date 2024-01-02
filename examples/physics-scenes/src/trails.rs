use std::collections::VecDeque;

use bevy::prelude::*;

use crate::rapier_schedule::PostRapierSchedule;

pub struct TrailsPlugin;

impl Plugin for TrailsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GizmoConfig {
            line_width: 1.0,
            ..default()
        })
        .add_systems(PostRapierSchedule, cache_trails)
        .add_systems(Last, draw_trails);
    }
}

#[derive(Component)]
pub struct Trail {
    pub length: f32,
    pub resolution: usize,
    cached: VecDeque<Vec2>,
}

impl Trail {
    pub fn new(length: f32, resolution: usize) -> Self {
        Self {
            length,
            resolution,
            cached: VecDeque::new(),
        }
    }
}

fn cache_trails(mut query: Query<(&GlobalTransform, &mut Trail)>) {
    for (transform, mut trail) in query.iter_mut() {
        trail.cached.push_back(transform.translation().truncate());
        if trail.cached.len() == (trail.length / crate::DT) as usize {
            trail.cached.pop_front();
        }
    }
}

fn draw_trails(mut gizmos: Gizmos, query: Query<&Trail>) {
    for trail in &query {
        gizmos.linestrip_2d(
            trail.cached.iter().step_by(trail.resolution).copied(),
            Color::RED,
        );
    }
}
