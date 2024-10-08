mod acceleration;
mod acceleration_softened;

pub use acceleration::Acceleration;
pub use acceleration_softened::AccelerationSoftened;

use crate::gravity::{GravitationalField, Mass, Position};

/// Trait for converting a particle to a SIMD one that can be used to compute accelerations.
pub trait ToSimd<const L: usize>: Sized {
    /// The SIMD type that `Self` can be made into.
    type Simd;

    /// Creates a SIMD value with all lanes set to `self`.
    fn lanes_splat(&self) -> Self::Simd;

    /// Creates a SIMD value with its lanes set to the given values in the array.
    fn lanes_array(array: [Self; L]) -> Self::Simd;

    /// Creates a SIMD value with its lanes set to the given values in the slice. If the slice is
    /// smaller than the number of lanes, the remaining lanes should be set to a default value.
    fn lanes_slice(slice: &[Self]) -> Self::Simd;
}

/// Trait for particles that can be collected into a tree used to compute accelerations.
///
/// This trait provides a method to compute the centre of mass of a group of particles.
pub trait TreeData {
    /// The type of the stored data in the tree used to compute the acceleration.
    type Data;

    /// Returns the centre of mass of the given particles to be stored in the tree.
    fn centre_of_mass<I: ExactSizeIterator<Item = Self> + Clone>(particles: I) -> Self::Data;
}

/// Trait for computing the acceleration a particle exerts at a given position.
pub trait AccelerationAt {
    /// The type of the position vector.
    type Vector;

    /// The type of the softening parameter.
    type Softening;

    /// The type of the acceleration.
    type Output;

    /// Computes the acceleration the particle exerts at the given position.
    fn acceleration_at<const CHECKED: bool>(
        &self,
        at: &Self::Vector,
        softening: &Self::Softening,
    ) -> Self::Output;
}

/// Trait for computing the accelerations two particles exert on each other.
pub trait AccelerationPaired {
    /// The type of the softening parameter.
    type Softening;

    /// The type of the acceleration.
    type Output;

    /// Computes the accelerations the two particles exert on each other.
    fn acceleration_paired(
        &self,
        other: &Self,
        softening: &Self::Softening,
    ) -> (Self::Output, Self::Output);
}

/// Trait for computing the acceleration of particles exerted by other particles on the GPU.
#[cfg(feature = "gpu")]
pub trait AccelerationGPU<const CHEKED: bool> {
    /// The type of the position vector.
    type Vector;

    /// The type of the position vector on the GPU.
    type GPUVector: bytemuck::Pod;

    /// The type of the affecting particle on the GPU.
    type GPUAffecting: bytemuck::Pod;

    /// The type of the acceleration on the GPU.
    type GPUOutput: bytemuck::Pod;

    /// The type of the acceleration.
    type Output;

    /// The source code of the shader that computes the acceleration.
    const SOURCE: &'static str;

    /// The source code of the shader that computes the acceleration with softening.
    const SOURCE_SOFTENED: &'static str;

    /// Converts a CPU position vector to a GPU position vector.
    fn to_gpu_position(position: &Self::Vector) -> Self::GPUVector;

    /// Converts the CPU particle to a GPU particle.
    fn to_gpu_particle(&self) -> Self::GPUAffecting;

    /// Converts the GPU acceleration to a CPU acceleration.
    fn to_cpu_output(output: &Self::GPUOutput) -> Self::Output;
}

impl<const L: usize, P, V, S> ToSimd<L> for P
where
    P: Position<Vector = V> + Mass<Scalar = S>,
    GravitationalField<V, S>: ToSimd<L> + Default,
{
    type Simd = <GravitationalField<V, S> as ToSimd<L>>::Simd;

    #[inline]
    fn lanes_splat(&self) -> Self::Simd {
        GravitationalField::from(self).lanes_splat()
    }

    #[inline]
    fn lanes_array(array: [Self; L]) -> Self::Simd {
        GravitationalField::lanes_array(array.each_ref().map(GravitationalField::from))
    }

    #[inline]
    fn lanes_slice(slice: &[Self]) -> Self::Simd {
        GravitationalField::lanes_array(std::array::from_fn(|i| {
            slice
                .get(i)
                .map(GravitationalField::from)
                .unwrap_or_default()
        }))
    }
}

impl<P, V, S> TreeData for P
where
    P: Position<Vector = V> + Mass<Scalar = S>,
    GravitationalField<V, S>: TreeData,
{
    type Data = <GravitationalField<V, S> as TreeData>::Data;

    #[inline]
    fn centre_of_mass<I: ExactSizeIterator<Item = Self> + Clone>(particles: I) -> Self::Data {
        GravitationalField::centre_of_mass(particles.map(|p| GravitationalField::from(&p)))
    }
}

impl<P, V, S> AccelerationAt for P
where
    P: Position<Vector = V> + Mass<Scalar = S>,
    GravitationalField<V, S>: AccelerationAt,
{
    type Vector = <GravitationalField<V, S> as AccelerationAt>::Vector;

    type Softening = <GravitationalField<V, S> as AccelerationAt>::Softening;

    type Output = <GravitationalField<V, S> as AccelerationAt>::Output;

    #[inline]
    fn acceleration_at<const CHECKED: bool>(
        &self,
        at: &Self::Vector,
        softening: &Self::Softening,
    ) -> Self::Output {
        GravitationalField::from(self).acceleration_at::<CHECKED>(at, softening)
    }
}

impl<P, V, S> AccelerationPaired for P
where
    P: Position<Vector = V> + Mass<Scalar = S>,
    GravitationalField<V, S>: AccelerationPaired,
{
    type Softening = <GravitationalField<V, S> as AccelerationPaired>::Softening;

    type Output = <GravitationalField<V, S> as AccelerationPaired>::Output;

    #[inline]
    fn acceleration_paired(
        &self,
        other: &Self,
        softening: &Self::Softening,
    ) -> (Self::Output, Self::Output) {
        GravitationalField::from(self)
            .acceleration_paired(&GravitationalField::from(other), softening)
    }
}

#[cfg(feature = "gpu")]
impl<const CHECKED: bool, P, V, S> AccelerationGPU<CHECKED> for P
where
    P: Position<Vector = V> + Mass<Scalar = S>,
    GravitationalField<V, S>: AccelerationGPU<CHECKED>,
{
    type Vector = <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::Vector;

    type GPUVector = <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::GPUVector;

    type GPUAffecting = <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::GPUAffecting;

    type GPUOutput = <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::GPUOutput;

    type Output = <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::Output;

    const SOURCE: &'static str = <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::SOURCE;

    const SOURCE_SOFTENED: &'static str =
        <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::SOURCE_SOFTENED;

    #[inline]
    fn to_gpu_position(position: &Self::Vector) -> Self::GPUVector {
        <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::to_gpu_position(position)
    }

    #[inline]
    fn to_gpu_particle(&self) -> Self::GPUAffecting {
        GravitationalField::from(self).to_gpu_particle()
    }

    #[inline]
    fn to_cpu_output(output: &Self::GPUOutput) -> Self::Output {
        <GravitationalField<V, S> as AccelerationGPU<CHECKED>>::to_cpu_output(output)
    }
}

#[cfg(test)]
mod tests {
    #[doc(hidden)]
    #[macro_export]
    macro_rules! acceleration_error {
        ($cm: expr, $epsilon: expr, $vector: ty, $scalar: ty, $splat: expr, $div: expr) => {
            let massive = vec![
                $crate::gravity::GravitationalField::new($splat(0.0), 20.0),
                $crate::gravity::GravitationalField::new($splat(1.0), 30.0),
                $crate::gravity::GravitationalField::new($splat(-3.0), 40.0),
            ];
            let particles = vec![
                $crate::gravity::GravitationalField::new($splat(10.0), 0.0),
                massive[0],
                massive[1],
                massive[2],
                $crate::gravity::GravitationalField::new($splat(30.0), 0.0),
                $crate::gravity::GravitationalField::new($splat(-45.0), 0.0),
            ];

            let is_affecting_fn = $crate::gravity::GravitationalField::is_affecting;
            let reordered = $crate::storage::Reordered::new(&particles, is_affecting_fn);

            let computed = $cm.compute(&reordered);
            dbg!(&computed);

            computed
                .zip(&particles)
                .for_each(|(computed, &point_mass1)| {
                    let mut acceleration = <$vector>::default();

                    for &point_mass2 in massive.iter() {
                        let dir = point_mass2.position - point_mass1.position;
                        let mag_2 = $crate::gravity::Norm::norm_squared(dir);

                        if mag_2 != Default::default() {
                            acceleration += dir * point_mass2.m * mag_2.recip().sqrt() / mag_2;
                        }
                    }

                    dbg!(point_mass1);
                    dbg!(acceleration);
                    dbg!(computed);

                    let error = $crate::gravity::Norm::norm_squared(
                        ($splat(1.0) - $div(computed, acceleration)),
                    )
                    .sqrt();
                    dbg!(error);

                    assert!(error <= $epsilon);
                })
        };
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! circular_orbit {
        ($cm: expr, $orbit_count: expr, $epsilon: expr, $vector: ty, $scalar: ident) => {
            const DT: $scalar = 1.0 / 60.0;

            fn specific_orbital_energy(radius: $scalar, mu1: $scalar, mu2: $scalar) -> $scalar {
                -(mu1 + mu2) / (radius + radius)
            }

            fn orbital_period(radius: $scalar, mu: $scalar) -> $scalar {
                std::$scalar::consts::TAU * ((radius * radius * radius) / mu).sqrt()
            }

            let main = (
                $crate::gravity::GravitationalField::new(<$vector>::default(), 1E6),
                <$vector>::default(),
            );
            let mut orbiting = (
                $crate::gravity::GravitationalField::<$vector, $scalar>::default(),
                <$vector>::default(),
            );
            orbiting.0.position.x = 100.0;
            orbiting.1.y = 100.0;

            let mut particles = [main.0, orbiting.0];
            let mut velocities = [main.1, orbiting.1];

            let main = &particles[0];
            let orbiting = &particles[1];

            let distance_before =
                $crate::gravity::Norm::norm_squared(main.position - orbiting.position).sqrt();
            let energy_before = specific_orbital_energy(distance_before, main.m, orbiting.m);

            let period = orbital_period(distance_before, main.m);
            // Steps to complete one full orbit.
            let steps = (period / DT).round() as usize;

            for _ in 0..steps * $orbit_count {
                let accelerations = $cm.compute(particles.as_slice());

                accelerations
                    .zip(&mut velocities)
                    .for_each(|(acceleration, velocity)| {
                        *velocity += acceleration * DT;
                    });

                for (point_mass, velocity) in particles.iter_mut().zip(velocities.iter()) {
                    point_mass.position += *velocity * DT;
                }
            }

            let main = &particles[0];
            let orbiting = &particles[1];

            let distance_after =
                $crate::gravity::Norm::norm_squared(main.position - orbiting.position).sqrt();
            let energy_after = specific_orbital_energy(distance_after, main.m, orbiting.m);

            let error_distance = (1.0 - distance_before / distance_after).abs();
            dbg!(error_distance);
            assert!(error_distance < $epsilon);

            let error_energy = (1.0 - energy_before / energy_after).abs();
            dbg!(error_energy);
            assert!(error_energy.abs() < $epsilon);
        };
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! tests_algorithms {
        ($vector: path, $scalar: ident, $splat: ident, $div: expr, [$($lane: literal),*], $name: ident on cpu) => {
            mod $name {
                use super::*;
                $crate::tests_algorithms_sequential!($vector, $scalar, $splat, $div, [$($lane),*]);
                #[cfg(feature = "parallel")]
                $crate::tests_algorithms_parallel!($vector, $scalar, $splat, $div, [$($lane),*]);
            }
        };
        ($vector: path, $scalar: ident, $splat: ident, $div: expr, [$($lane: literal),*], $name: ident on cpu and gpu) => {
            mod $name {
                use super::*;
                $crate::tests_algorithms_sequential!($vector, $scalar, $splat, $div, [$($lane),*]);
                #[cfg(feature = "parallel")]
                $crate::tests_algorithms_parallel!($vector, $scalar, $splat, $div, [$($lane),*]);
                #[cfg(feature = "gpu")]
                $crate::tests_algorithms_gpu!($vector, $scalar, $splat, $div);
            }
        };
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! tests_algorithms_sequential {
        ($vector: path, $scalar: ident, $splat: ident, $div: expr, [$($lane: literal),*]) => {
            mod sequential {
                use super::*;
                use $crate::gravity::newtonian::Acceleration;
                use $crate::sequential::{BarnesHut, BruteForce, BruteForcePairs};
                use $crate::Interaction;

                #[test]
                fn brute_force() {
                    let mut c = BruteForce(Acceleration::checked());
                    $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                }

                $(
                    paste::paste! {
                        #[test]
                        fn [<brute_force_simd_$lane>]() {
                            use $crate::sequential::BruteForceSimd;
                            let mut c = BruteForceSimd::<$lane, _>(Acceleration::checked());
                            $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                            $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                        }
                    }
                )*

                #[test]
                fn brute_force_pairs() {
                    let mut c = BruteForcePairs(Acceleration::checked());
                    $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                }

                #[test]
                fn barnes_hut() {
                    let mut c = BarnesHut::new(0.0, Acceleration::checked());
                    $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                }

                #[test]
                fn barnes_hut_05() {
                    let mut c = BarnesHut::new(0.5, Acceleration::checked());
                    $crate::acceleration_error!(c, 5e-1, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-1, $vector, $scalar);
                }
            }
        };
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! tests_algorithms_parallel {
        ($vector: path, $scalar: ident, $splat: ident, $div: expr, [$($lane: literal),*]) => {
            mod parallel {
                use super::*;
                use rayon::prelude::*;
                use $crate::gravity::newtonian::Acceleration;
                use $crate::parallel::{BarnesHut, BruteForce};
                use $crate::Interaction;

                #[test]
                fn brute_force() {
                    let mut c = BruteForce(Acceleration::checked());
                    $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                }

                $(
                    paste::paste! {
                        #[test]
                        fn [<brute_force_simd_$lane>]() {
                            use $crate::parallel::BruteForceSimd;
                            let mut c = BruteForceSimd::<$lane, _>(Acceleration::checked());
                            $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                            $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                        }
                    }
                )*

                #[test]
                fn barnes_hut() {
                    let mut c = BarnesHut::new(0.0, Acceleration::checked());
                    $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-2, $vector, $scalar);
                }

                #[test]
                fn barnes_hut_05() {
                    let mut c = BarnesHut::new(0.5, Acceleration::checked());
                    $crate::acceleration_error!(c, 5e-1, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 1_000, 1e-1, $vector, $scalar);
                }
            }
        };
    }

    #[doc(hidden)]
    #[macro_export]
    macro_rules! tests_algorithms_gpu {
        ($vector: path, $scalar: ident, $splat: ident, $div: expr) => {
            mod gpu {
                use super::*;
                use $crate::gpu::{BruteForce, GpuResources, MemoryStrategy};
                use $crate::gravity::newtonian::Acceleration;
                use $crate::Interaction;

                // Defaults
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
                                required_features: wgpu::Features::empty()
                                    | wgpu::Features::PUSH_CONSTANTS,
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

                #[test]
                fn brute_force() {
                    let (device, queue) = &pollster::block_on(setup_wgpu());
                    let resources = &mut GpuResources::new(MemoryStrategy::Shared(64));
                    let mut c = BruteForce::new(resources, device, queue, Acceleration::checked());
                    $crate::acceleration_error!(c, 1e-2, $vector, $scalar, <$vector>::$splat, $div);
                    $crate::circular_orbit!(c, 100, 1e-2, $vector, $scalar);
                }
            }
        };
    }
}
