use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use rand::{thread_rng, Rng};

use glam::Vec3 as Vector;
use particular::prelude::*;

#[derive(Particle, Clone)]
pub struct Body {
    position: Vector,
    mu: f32,
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
        .sample_size(50);

    for i in (2..=16).map(|i| 2_usize.pow(i)) {
        let bodies = random_bodies(i);

        #[cfg(feature = "gpu")]
        {
            let mut cm = gpu::BruteForce::new_init(i, i);
            group.bench_with_input(
                BenchmarkId::new("gpu::BruteForce", i),
                &bodies,
                |b, input| b.iter(|| input.iter().accelerations(&mut cm).collect::<Vec<_>>()),
            );
        }

        #[cfg(feature = "parallel")]
        {
            let cm = parallel::BruteForce;
            group.bench_with_input(
                BenchmarkId::new("parallel::BruteForce", i),
                &bodies,
                |b, input| b.iter(|| input.iter().accelerations(cm).collect::<Vec<_>>()),
            );

            let cm = parallel::BarnesHut { theta: 1.0 };
            group.bench_with_input(
                BenchmarkId::new("parallel::BarnesHut", i),
                &bodies,
                |b, input| b.iter(|| input.iter().accelerations(cm).collect::<Vec<_>>()),
            );
        }

        {
            let cm = sequential::BruteForce;
            group.bench_with_input(
                BenchmarkId::new("sequential::BruteForce", i),
                &bodies,
                |b, input| b.iter(|| input.iter().accelerations(cm).collect::<Vec<_>>()),
            );

            let cm = sequential::BarnesHut { theta: 1.0 };
            group.bench_with_input(
                BenchmarkId::new("sequential::BarnesHut", i),
                &bodies,
                |b, input| b.iter(|| input.iter().accelerations(cm).collect::<Vec<_>>()),
            );
        }
    }

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
