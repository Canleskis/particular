use criterion::{AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration};

use glam::Vec3 as Vector;
use particular::prelude::*;
use rand::prelude::*;

#[derive(Debug, Default, Particle, Clone, Copy)]
pub struct Body {
    position: Vector,
    mu: f32,
}

fn gen_range_vector<const N: usize, V>(rng: &mut StdRng, range: std::ops::Range<f32>) -> V
where
    V: From<[f32; N]>,
{
    [0.0; N].map(|_| rng.gen_range(range.clone())).into()
}

pub fn random_bodies(mut rng: StdRng, i: usize) -> Vec<Body> {
    (0..i)
        .map(|_| {
            let position = gen_range_vector(&mut rng, -5000.0..5000.0);
            let mu = rng.gen_range(0.1..100.0);

            Body { position, mu }
        })
        .collect()
}

fn bench_compute_method<M, S, C>(bodies: &[Body], group: &mut BenchmarkGroup<'_, M>, mut cm: C)
where
    M: criterion::measurement::Measurement,
    S: compute_method::Storage<PointMass<Body>>,
    for<'a> &'a mut C: compute_method::ComputeMethod<S, Vector>,
{
    let name = std::any::type_name::<C>().trim_start_matches("particular::algorithms::");
    let bench_name = name.find('<').map_or(name, |index| &name[..index]);

    group.bench_function(BenchmarkId::new(bench_name, bodies.len()), |b| {
        b.iter(|| bodies.iter().accelerations(&mut cm).collect::<Vec<_>>())
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Particular");
    group
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic))
        .warm_up_time(std::time::Duration::from_secs(1))
        .sample_size(50);

    let particle_count_iterator = (0..16).map(|i| 2usize.pow(i));

    for i in particle_count_iterator {
        let bodies = random_bodies(StdRng::seed_from_u64(1808), i);

        #[cfg(feature = "gpu")]
        {
            bench_compute_method(&bodies, &mut group, gpu::BruteForce::new_init(i, i));
        }

        #[cfg(feature = "parallel")]
        {
            bench_compute_method(&bodies, &mut group, parallel::BruteForce);
            bench_compute_method(&bodies, &mut group, parallel::BruteForceSIMD);
            bench_compute_method(&bodies, &mut group, parallel::BarnesHut { theta: 1.0 });
        }

        {
            bench_compute_method(&bodies, &mut group, sequential::BruteForce);
            bench_compute_method(&bodies, &mut group, sequential::BruteForceAlt);
            bench_compute_method(&bodies, &mut group, sequential::BruteForceSIMD);
            bench_compute_method(&bodies, &mut group, sequential::BarnesHut { theta: 1.0 });
        }
    }

    group.finish();
}

criterion::criterion_group!(benches, criterion_benchmark);
criterion::criterion_main!(benches);
