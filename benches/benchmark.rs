use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
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
    let mut group = c.benchmark_group("Particular");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for i in (2..=16).map(|i| 2_usize.pow(i)) {
        let bodies = &random_bodies(i);

        #[cfg(feature = "gpu")]
        {
            let mut cm = gpu::BruteForce::default();
            group.bench_with_input(BenchmarkId::new("GPU", i), &bodies, |b, input| {
                b.iter(|| get_acceleration(input, &mut cm))
            });
        }

        #[cfg(feature = "parallel")]
        {
            let mut cm = parallel::BruteForce;
            group.bench_with_input(BenchmarkId::new("CPU MT", i), &bodies, |b, input| {
                b.iter(|| get_acceleration(input, &mut cm))
            });
        }

        {
            let mut cm = sequential::BruteForce;
            group.bench_with_input(BenchmarkId::new("CPU ST", i), &bodies, |b, input| {
                b.iter(|| get_acceleration(input, &mut cm))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
