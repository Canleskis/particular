use std::time::Duration;

use criterion::{AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration};

use particular::prelude::*;
use rand::prelude::*;

type Scalar = f32;
type Vector = glam::Vec3;

#[derive(Debug, Default, Particle, Clone, Copy)]
pub struct Body {
    position: Vector,
    mu: Scalar,
}

fn gen_range_vector<const N: usize, V>(rng: &mut StdRng, range: std::ops::Range<Scalar>) -> V
where
    V: From<[Scalar; N]>,
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

fn bench_compute_method<M, S, C>(
    bodies: &[Body],
    group: &mut BenchmarkGroup<'_, M>,
    mut cm: C,
    suffix: &str,
) where
    M: criterion::measurement::Measurement,
    S: Storage<ParticlePointMass<Body>>,
    for<'a> &'a mut C: ComputeMethod<S, Vector>,
{
    let name = std::any::type_name::<C>().trim_start_matches("particular::algorithms::");
    let bench_name = name
        .find('<')
        .map_or(name, |index| &name[..index])
        .to_owned()
        + suffix;

    group.bench_function(BenchmarkId::new(bench_name, bodies.len()), |b| {
        b.iter(|| bodies.iter().accelerations(&mut cm).collect::<Vec<_>>())
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut gr = c.benchmark_group("Particular");
    gr.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic))
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(Duration::from_secs(4))
        .sample_size(50);

    let particle_count_iterator = (1..16).map(|i| 2usize.pow(i));
    let thetas = [0.2, 0.8];

    for i in particle_count_iterator {
        let b = random_bodies(StdRng::seed_from_u64(1808), i);

        #[cfg(feature = "gpu")]
        {
            bench_compute_method(&b, &mut gr, gpu::BruteForce::new_init(i, i), "");
        }

        #[cfg(feature = "parallel")]
        {
            bench_compute_method(&b, &mut gr, parallel::BruteForce, "");
            bench_compute_method(&b, &mut gr, parallel::BruteForceSIMD, "");
            for theta in thetas {
                let suffix = format!("::{theta}");
                bench_compute_method(&b, &mut gr, parallel::BarnesHut { theta }, &suffix);
            }
        }

        {
            bench_compute_method(&b, &mut gr, sequential::BruteForce, "");
            bench_compute_method(&b, &mut gr, sequential::BruteForcePairs, "");
            bench_compute_method(&b, &mut gr, sequential::BruteForcePairsAlt, "");
            bench_compute_method(&b, &mut gr, sequential::BruteForceSIMD, "");
            for theta in thetas {
                let suffix = format!("::{theta}");
                bench_compute_method(&b, &mut gr, sequential::BarnesHut { theta }, &suffix);
            }
        }
    }

    gr.finish();
}

criterion::criterion_group!(benches, criterion_benchmark);
criterion::criterion_main!(benches);
