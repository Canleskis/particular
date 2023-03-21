use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use rand::{thread_rng, Rng};

use glam::Vec3 as Vector;
use particular::prelude::*;

struct Dummy;

impl ComputeMethod<Vector, f32> for Dummy {
    fn compute(&mut self, particles: &[(Vector, f32)]) -> Vec<Vector> {
        vec![Vector::ZERO; particles.len()]
    }
}

#[derive(Particle, Clone)]
pub struct Body {
    position: Vector,
    mu: f32,
}

fn get_acceleration(bodies: &[Body], cm: &mut impl ComputeMethod<glam::Vec3A, f32>) {
    let _ = bodies.iter().accelerations(cm).collect::<Vec<_>>();
}

fn random_bodies(i: usize) -> Vec<Body> {
    let mut rng = thread_rng();
    let mut gen = |range| rng.gen_range(range);

    (0..i)
        .map(|_| {
            let position = Vector::splat(gen(0.0..10000.0));
            let mu = gen(0.1..100.0);

            Body { position, mu }
        })
        .collect()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Particular");
    group
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic))
        .warm_up_time(std::time::Duration::from_secs(1))
        .sample_size(15);

    for i in (2..=16).map(|i| 2_usize.pow(i)) {
        let bodies = random_bodies(i);

        #[cfg(feature = "gpu")]
        {
            let mut cm = gpu::BruteForce::default();
            group.bench_with_input(
                BenchmarkId::new("gpu::BruteForce", i),
                &bodies,
                |b, input| b.iter(|| get_acceleration(input, &mut cm)),
            );
        }

        #[cfg(feature = "parallel")]
        {
            let mut cm = parallel::BruteForce;
            group.bench_with_input(
                BenchmarkId::new("parallel::BruteForce", i),
                &bodies,
                |b, input| b.iter(|| get_acceleration(input, &mut cm)),
            );

            let mut cm = parallel::BarnesHut { theta: 1.0 };
            group.bench_with_input(
                BenchmarkId::new("parallel::BarnesHut", i),
                &bodies,
                |b, input| b.iter(|| get_acceleration(input, &mut cm)),
            );
        }

        {
            let mut cm = sequential::BruteForce;
            group.bench_with_input(
                BenchmarkId::new("sequential::BruteForce", i),
                &bodies,
                |b, input| b.iter(|| get_acceleration(input, &mut cm)),
            );

            let mut cm = sequential::BarnesHut { theta: 1.0 };
            group.bench_with_input(
                BenchmarkId::new("sequential::BarnesHut", i),
                &bodies,
                |b, input| b.iter(|| get_acceleration(input, &mut cm)),
            );
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
