#[cfg(feature = "gpu")]
mod wgpu_data;

/// Tree, bounding box and BarnesHut implementation details.
pub mod tree;

/// Internal and SIMD representation of vectors used for expensive computations.
///
/// These traits and their methods are used by the built-in [`ComputeMethods`](crate::compute_method::ComputeMethod)
/// to abstract common vector operations and their conversion from and into arbitrary vectors.
pub mod vector;

/// Storage implementations used by built-in [`ComputeMethods`](crate::compute_method::ComputeMethod).
pub mod storage;

/// Compute methods that use the GPU.
#[cfg(feature = "gpu")]
pub mod gpu;

/// Compute methods that use multiple CPU threads.
#[cfg(feature = "parallel")]
pub mod parallel;

/// Compute methods that use one CPU thread.
pub mod sequential;

/// Built-in [`ComputeMethods`](crate::compute_method::ComputeMethod).
pub mod compute_methods {
    #[cfg(feature = "gpu")]
    pub use super::gpu;

    #[cfg(feature = "parallel")]
    pub use super::parallel;

    pub use super::sequential;
}

pub use storage::*;
pub use vector::*;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::compute_method::{ComputeMethod, Storage};
    use glam::Vec3A;

    pub fn acceleration_computation<S, C>(mut cm: C, epsilon: f32)
    where
        S: Storage<PointMass<Vec3A, f32>>,
        for<'a> &'a mut C: ComputeMethod<S, Vec3A>,
    {
        let massive = vec![
            PointMass::new(Vec3A::splat(0.0), 20.0),
            PointMass::new(Vec3A::splat(1.0), 30.0),
            PointMass::new(Vec3A::splat(-3.0), 40.0),
        ];

        let particles = vec![
            PointMass::new(Vec3A::splat(3.0), 0.0),
            massive[0],
            massive[1],
            massive[2],
            PointMass::new(Vec3A::splat(5.0), 0.0),
            PointMass::new(Vec3A::splat(-5.0), 0.0),
        ];

        let storage = S::store(particles.clone().into_iter());
        let computed = cm.compute(&storage);

        for (&point_mass1, computed) in particles.iter().zip(computed) {
            let mut acceleration = Vec3A::ZERO;

            for &point_mass2 in massive.iter() {
                let dir = point_mass2.position - point_mass1.position;
                let mag_2 = dir.length_squared();

                if mag_2 != 0.0 {
                    acceleration += dir * point_mass2.mass * mag_2.recip().sqrt() / mag_2;
                }
            }

            dbg!(acceleration);
            dbg!(computed);
            assert!((acceleration).abs_diff_eq(computed, epsilon))
        }
    }

    pub fn circular_orbit_stability<S, C>(mut cm: C, orbit_count: usize, epsilon: f32)
    where
        S: Storage<PointMass<Vec3A, f32>>,
        for<'a> &'a mut C: ComputeMethod<S, Vec3A>,
    {
        const DT: f32 = 1.0 / 60.0;

        fn specific_orbital_energy(radius: f32, mu1: f32, mu2: f32) -> f32 {
            let mus = mu1 + mu2;
            -mus / (2.0 * radius)
        }

        fn orbital_period(radius: f32, main_mass: f32) -> f32 {
            2.0 * std::f32::consts::PI * ((radius * radius * radius) / main_mass).sqrt()
        }

        let mut main = (
            Vec3A::new(0.0, 0.0, 0.0),
            PointMass::new(Vec3A::new(0.0, 0.0, 0.0), 1E6),
        );
        let mut orbiting = (
            Vec3A::new(0.0, 100.0, 0.0),
            PointMass::new(Vec3A::new(100.0, 0.0, 0.0), 0.0),
        );

        let radius = main.1.position.distance(orbiting.1.position);
        let energy_before = specific_orbital_energy(radius, main.1.mass, orbiting.1.mass);
        let distance_before = radius;
        let period = orbital_period(radius, main.1.mass);

        // Steps to complete one full orbit.
        let steps = (period / DT).round() as usize;

        let mut particles = vec![&mut main, &mut orbiting];
        for _ in 0..steps * orbit_count {
            let accelerations = cm.compute(&S::store(particles.iter().map(|(_, pm)| *pm)));
            for ((velocity, point_mass), acceleration) in particles.iter_mut().zip(accelerations) {
                *velocity += acceleration * DT;
                point_mass.position += *velocity * DT;
            }
        }

        let radius = main.1.position.distance(orbiting.1.position);
        let energy_after = specific_orbital_energy(radius, main.1.mass, orbiting.1.mass);
        let distance_after = radius;

        let error_energy = 1.0 - energy_before / energy_after;
        dbg!(error_energy);
        assert!(error_energy.abs() < epsilon);

        let error_distance = 1.0 - distance_before / distance_after;
        dbg!(error_distance);
        assert!(error_distance.abs() < epsilon)
    }
}
