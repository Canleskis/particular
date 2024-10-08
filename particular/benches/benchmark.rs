use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration};

use particular::prelude::*;
use rand::prelude::*;

type Scalar = f32;
type Vector = glam::Vec3;

fn gen_range_vector<const N: usize, V>(rng: &mut StdRng, range: std::ops::Range<Scalar>) -> V
where
    V: From<[Scalar; N]>,
{
    [0.0; N].map(|_| rng.gen_range(range.clone())).into()
}

#[derive(Clone, Copy, Debug, Position, Mass)]
struct Body {
    position: Vector,
    mu: Scalar,
}

fn random_bodies(rng: &mut StdRng, n: usize, ratio_massive: f32) -> Vec<Body> {
    let massive_bodies = (n as f32 * ratio_massive).round() as usize;

    (0..n)
        .map(|i| {
            let position = gen_range_vector(rng, -5e3..5e3);
            let mu = if i < massive_bodies {
                rng.gen_range(1e3..1e9)
            } else {
                0.0
            };

            Body { position, mu }
        })
        .collect()
}

/// Helper function to get default values for [`wgpu::Device`] and
/// [`wgpu::Queue`].
#[cfg(feature = "gpu")]
async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty() | wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 4,
                    ..Default::default()
                },
            },
            None,
        )
        .await
        .unwrap()
}

// On wasm there are only 128-bit registers.
#[cfg(target_arch = "wasm32")]
const WIDTH: usize = 128;
#[cfg(not(target_arch = "wasm32"))]
const WIDTH: usize = 128;

const L: usize = WIDTH / (std::mem::size_of::<Scalar>() * 8);

use particular::gravity::newtonian::Acceleration;
macro_rules! bench {
    ($group: tt, $velocities: ident, $bodies: ident, $algorithm: ident$(::<$generic: tt>)?($($args: expr),*), $suffix: expr) => {
        $group.bench_function(BenchmarkId::new(format!("{}{}", stringify!($algorithm), $suffix), $bodies.len()), |bencher| {
            bencher.iter(|| $bodies.$algorithm::<$($generic)?>($($args,)* Acceleration::checked()).zip(&mut $velocities).for_each(|(a, v)| *v += a));
        });
    };
    ($group: tt, $velocities: ident, $bodies: ident, $algorithm: ident$(::<$generic: tt>)?($($args: expr),*)) => {
        bench!($group, $velocities, $bodies, $algorithm$(::<$generic>)?($($args),*), "");
    };
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Particular");
    group
        .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic))
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(1))
        .sample_size(15);

    let particle_count_iterator = (1..17).map(|i| 2usize.pow(i));
    let thetas = [0.3, 0.7];

    for n in particle_count_iterator {
        let mut rng = StdRng::seed_from_u64(1808);
        let b = random_bodies(&mut rng, n, 1.0);
        let mut v = b
            .iter()
            .map(|_| gen_range_vector(&mut rng, -100.0..100.0))
            .collect::<Vec<Vector>>();

        #[cfg(feature = "gpu")]
        {
            let (device, queue) = &pollster::block_on(setup_wgpu());
            let resources = &mut GpuResources::new(MemoryStrategy::Shared(64));
            bench!(group, v, b, gpu_brute_force(resources, device, queue));
        }

        #[cfg(feature = "parallel")]
        {
            bench!(group, v, b, par_brute_force());
            bench!(group, v, b, par_brute_force_simd::<L>(), format!("<{L}>"));
            for theta in thetas {
                bench!(group, v, b, par_barnes_hut(theta), format!("({theta})"));
            }
        }

        {
            bench!(group, v, b, brute_force());
            bench!(group, v, b, brute_force_simd::<L>(), format!("<{L}>"));
            bench!(group, v, b, brute_force_pairs());
            for theta in thetas {
                bench!(group, v, b, barnes_hut(theta), format!("({theta})"));
            }
        }
    }

    group.finish();
}

criterion::criterion_group!(benches, criterion_benchmark);
criterion::criterion_main!(benches);
