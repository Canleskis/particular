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

    pub fn acceleration_computation<S, C>(cm: C, epsilon: f32)
    where
        C: ComputeMethod<S, Vec3A>,
        S: Storage<PointMass<Vec3A, f32>>,
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
}
