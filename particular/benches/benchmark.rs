use criterion::{AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration};

use particular::prelude::*;
use rand::prelude::*;

type Scalar = f32;
type Vector = ultraviolet::Vec3;
type PointMass = particular::point_mass::PointMass<Vector, Scalar>;

fn gen_range_vector<const N: usize, V>(rng: &mut StdRng, range: std::ops::Range<Scalar>) -> V
where
    V: From<[Scalar; N]>,
{
    [0.0; N].map(|_| rng.gen_range(range.clone())).into()
}

pub fn random_massive_bodies(mut rng: StdRng, i: usize) -> Vec<PointMass> {
    (0..i)
        .map(|_| {
            let position = gen_range_vector(&mut rng, -5000.0..5000.0);
            let mass = rng.gen_range(0.1..100.0);

            PointMass { position, mass }
        })
        .collect()
}

pub fn random_massless_bodies(mut rng: StdRng, i: usize) -> Vec<PointMass> {
    (0..i)
        .map(|_| {
            let position = gen_range_vector(&mut rng, -5000.0..5000.0);
            let mass = 0.0;

            PointMass { position, mass }
        })
        .collect()
}

pub fn random_mix_bodies(mut rng: StdRng, i: usize) -> Vec<PointMass> {
    (0..i)
        .map(|_| {
            let position = gen_range_vector(&mut rng, -5000.0..5000.0);
            let mass = rng.gen_range(-300.0_f32..100.0).max(0.0);

            PointMass { position, mass }
        })
        .collect()
}

fn bench_compute_method<C>(
    bodies: &[PointMass],
    mut cm: C,
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    suffix: &str,
) where
    for<'a> C: ComputeMethod<&'a [PointMass]>,
{
    let name = std::any::type_name::<C>().trim_start_matches("particular::algorithms::");
    let bench_name = name
        .find('<')
        .map_or(name, |index| &name[..index])
        .to_owned()
        + suffix;

    group.bench_function(BenchmarkId::new(bench_name, bodies.len()), |b| {
        b.iter(|| cm.compute(bodies))
    });
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut g = c.benchmark_group("Particular");
    g.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic))
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(4))
        .sample_size(50);

    let particle_count_iterator = (1..21).map(|i| 2usize.pow(i));
    let thetas = [0.3, 0.7];

    for i in particle_count_iterator {
        let b = random_massive_bodies(StdRng::seed_from_u64(1808), i);

        #[cfg(feature = "gpu")]
        {
            let (device, queue) = &pollster::block_on(particular::gpu::setup_wgpu());
            let gpu_data = &mut gpu::GpuData::new_init(device, b.len(), b.len());
            let gpu_brute_force = gpu::BruteForce {
                gpu_data,
                device,
                queue,
            };
            bench_compute_method(&b, gpu_brute_force, &mut g, "");
        }

        #[cfg(feature = "parallel")]
        {
            bench_compute_method(&b, parallel::BruteForceScalar, &mut g, "");
            bench_compute_method(&b, parallel::BruteForceSIMD::<8>, &mut g, "");
            for theta in thetas {
                let suffix = format!("::{theta}");
                bench_compute_method(&b, parallel::BarnesHut { theta }, &mut g, &suffix);
            }
        }

        {
            bench_compute_method(&b, sequential::BruteForceScalar, &mut g, "");
            bench_compute_method(&b, sequential::BruteForcePairs, &mut g, "");
            bench_compute_method(&b, sequential::BruteForceSIMD::<8>, &mut g, "");
            for theta in thetas {
                let suffix = format!("::{theta}");
                bench_compute_method(&b, sequential::BarnesHut { theta }, &mut g, &suffix);
            }
        }
    }

    g.finish();
}

criterion::criterion_group!(benches, criterion_benchmark);
criterion::criterion_main!(benches);
