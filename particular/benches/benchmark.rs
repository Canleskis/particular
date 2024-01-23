use criterion::{AxisScale, BenchmarkGroup, BenchmarkId, Criterion, PlotConfiguration};

use particular::prelude::*;
use rand::prelude::*;

type Scalar = f32;
type Vector = ultraviolet::Vec3;
type PointMass = particular::storage::PointMass<Vector, Scalar>;

fn gen_range_vector<const N: usize, V>(rng: &mut StdRng, range: std::ops::Range<Scalar>) -> V
where
    V: From<[Scalar; N]>,
{
    [0.0; N].map(|_| rng.gen_range(range.clone())).into()
}

pub fn random_bodies(rng: &mut StdRng, i: usize, ratio_massive: f32) -> Vec<PointMass> {
    let massive_bodies = (i as f32 * ratio_massive).round() as usize;

    (0..i)
        .map(|i| {
            let position = gen_range_vector(rng, -5e3..5e3);
            let mass = if i < massive_bodies {
                rng.gen_range(1e-1..1e3)
            } else {
                0.0
            };

            PointMass { position, mass }
        })
        .collect()
}

#[inline]
fn bench_cm<C, S>(
    bodies: S,
    len: usize,
    mut cm: C,
    group: &mut BenchmarkGroup<'_, criterion::measurement::WallTime>,
    trim_generic: bool,
    suffix: &str,
) where
    S: Copy,
    C: ComputeMethod<S>,
{
    let trimmed_start =
        std::any::type_name::<C>().trim_start_matches("particular::compute_method::");

    let trimmed = if trim_generic {
        trimmed_start.split('<').next().unwrap_or_default()
    } else {
        trimmed_start
    }
    .to_owned();

    group.bench_function(BenchmarkId::new(trimmed + suffix, len), |bencher| {
        bencher.iter(|| cm.compute(bodies))
    });
}

// On wasm there are only 128-bit registers.
#[cfg(target_arch = "wasm32")]
const WIDTH: usize = 128;
#[cfg(not(target_arch = "wasm32"))]
const WIDTH: usize = 256;

const LANES: usize = WIDTH / (8 * std::mem::size_of::<Scalar>());

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Particular");
    group
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic))
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(1))
        .sample_size(15);

    let particle_count_iterator = (1..17).map(|i| 2usize.pow(i));
    let thetas = [0.3, 0.7];

    let g = &mut group;
    for i in particle_count_iterator {
        let b = random_bodies(&mut StdRng::seed_from_u64(1808), i, 1.0);
        let len = b.len();

        #[cfg(feature = "gpu")]
        {
            let (device, queue) = &pollster::block_on(particular::gpu::setup_wgpu());
            let gpu_data = &mut gpu::GpuData::new_init(device, b.len(), b.len());
            let gpu_brute_force = gpu::BruteForce {
                gpu_data,
                device,
                queue,
            };
            bench_cm(&*b, len, gpu_brute_force, g, true, "");
        }

        #[cfg(feature = "parallel")]
        {
            bench_cm(&*b, len, parallel::BruteForceScalar, g, true, "");
            bench_cm(&*b, len, parallel::BruteForceSIMD::<LANES>, g, false, "");
            for theta in thetas {
                let suffix = &format!("::{theta}");
                bench_cm(&*b, len, parallel::BarnesHut { theta }, g, true, suffix);
            }
        }

        {
            bench_cm(&*b, len, sequential::BruteForceScalar, g, true, "");
            bench_cm(&*b, len, sequential::BruteForcePairs, g, true, "");
            bench_cm(&*b, len, sequential::BruteForceSIMD::<LANES>, g, false, "");
            for theta in thetas {
                let suffix = &format!("::{theta}");
                bench_cm(&*b, len, sequential::BarnesHut { theta }, g, true, suffix);
            }
        }
    }

    group.finish();
}

criterion::criterion_group!(benches, criterion_benchmark);
criterion::criterion_main!(benches);
