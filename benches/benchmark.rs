use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};

use glam::Vec2;
use particular::prelude::*;

#[particle(2)]
pub struct Body {
    position: Vec2,
    mu: f32,
}

fn get_acceleration(mut set: ParticleSet<Body>) {
    _ = set.result();
}

fn random_bodies(i: usize) -> ParticleSet<Body> {
    let mut rng = thread_rng();
    let mut gen = |range| rng.gen_range(range);
    let mut particle_set = ParticleSet::new();
    for _ in 0..i {
        let body = Body {
            position: Vec2::splat(gen(0.0..10000.0)),
            mu: gen(0.0..100.0),
        };
        particle_set.add(body);
    }
    particle_set
}

fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(not(feature = "parallel"))]
    let mut group = c.benchmark_group("Particular single-threaded");
    #[cfg(feature = "parallel")]
    let mut group = c.benchmark_group("Particular multi-threaded");

    for i in (500..=5000).step_by(500) {
        group.bench_with_input(BenchmarkId::new("Body count", i), &i, |b, i| {
            b.iter(|| get_acceleration(black_box(random_bodies(*i))))
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
