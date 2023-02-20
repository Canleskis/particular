use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{thread_rng, Rng};

use glam::Vec2 as Vector;
use particular::prelude::*;

#[derive(Particle)]
pub struct Body {
    position: Vector,
    mu: f32,
}

type Bodies = ParticleSet<Body>;

fn get_acceleration(set: &Bodies, cm: &mut impl ComputeMethod<glam::Vec3A, f32>) {
    _ = set.accelerations(cm).collect::<Vec<_>>();
}

fn random_bodies(i: usize) -> Bodies {
    let mut rng = thread_rng();
    let mut gen = |range| rng.gen_range(range);

    let mut particle_set = Bodies::new();

    for _ in 0..i {
        let body = Body {
            position: Vector::splat(gen(0.0..10000.0)),
            mu: gen(0.0..100.0),
        };
        particle_set.add(body);
    }

    particle_set
}

fn criterion_benchmark(c: &mut Criterion) {
    #[cfg(not(feature = "parallel"))]
    #[cfg(not(feature = "gpu"))]
    let (mut group, mut cm) = { (c.benchmark_group("CPU ST"), sequential::BruteForce) };

    #[cfg(feature = "parallel")]
    #[cfg(not(feature = "gpu"))]
    let (mut group, mut cm) = { (c.benchmark_group("CPU MT"), parallel::BruteForce) };

    #[cfg(feature = "gpu")]
    let (mut group, mut cm) = { (c.benchmark_group("GPU"), gpu::BruteForce::default()) };

    for i in (500..=50000).step_by(500) {
        let bb = black_box(random_bodies(i));

        group.bench_function(BenchmarkId::new("Body count", i), |b| {
            b.iter(|| get_acceleration(&bb, &mut cm))
        });
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
