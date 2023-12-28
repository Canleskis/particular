#[cfg(feature = "gpu")]
mod gpu_data;

/// Trait abstractions for generic vectors and associated floating-point numbers.
pub mod math;
/// Representation of the position and mass of an object in N-dimensional space and the operations to compute gravitational forces.
pub mod point_mass;
/// Storages used by built-in [`ComputeMethods`](crate::compute_method::ComputeMethod).
pub mod storage;
/// Tree, bounding box and BarnesHut implementation details.
pub mod tree;

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

pub use point_mass::*;
pub use storage::*;

#[cfg(test)]
pub(crate) mod tests {
    use crate::{algorithms::PointMass, compute_method::ComputeMethod};
    use ultraviolet::Vec3;

    pub fn acceleration_error<C>(mut cm: C, epsilon: f32)
    where
        for<'a> C: ComputeMethod<&'a [PointMass<Vec3, f32>], Output = Vec<Vec3>>,
    {
        let massive = [
            PointMass::new(Vec3::broadcast(0.0), 20.0),
            PointMass::new(Vec3::broadcast(1.0), 30.0),
            PointMass::new(Vec3::broadcast(-3.0), 40.0),
        ];

        let particles = [
            PointMass::new(Vec3::broadcast(3.0), 0.0),
            massive[0],
            massive[1],
            massive[2],
            PointMass::new(Vec3::broadcast(5.0), 0.0),
            PointMass::new(Vec3::broadcast(-5.0), 0.0),
        ];

        let computed = cm.compute(&particles);

        for (&point_mass1, computed) in particles.iter().zip(computed) {
            let mut acceleration = Vec3::broadcast(0.0);

            for &point_mass2 in massive.iter() {
                let dir = point_mass2.position - point_mass1.position;
                let mag_2 = dir.mag_sq();

                if mag_2 != 0.0 {
                    acceleration += dir * point_mass2.mass * mag_2.recip().sqrt() / mag_2;
                }
            }

            dbg!(acceleration);
            dbg!(computed);

            let error = (Vec3::one() - computed / acceleration).abs();
            dbg!(error);
            assert!(error.x <= epsilon);
            assert!(error.y <= epsilon);
            assert!(error.z <= epsilon);
        }
    }

    pub fn circular_orbit_stability<C>(mut cm: C, orbit_count: usize, epsilon: f32)
    where
        for<'a> C: ComputeMethod<&'a [PointMass<Vec3, f32>], Output = Vec<Vec3>>,
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
            Vec3::new(0.0, 0.0, 0.0),
            PointMass::new(Vec3::new(0.0, 0.0, 0.0), 1E6),
        );
        let mut orbiting = (
            Vec3::new(0.0, 100.0, 0.0),
            PointMass::new(Vec3::new(100.0, 0.0, 0.0), 0.0),
        );

        let distance_before = (main.1.position - orbiting.1.position).mag();
        let energy_before = specific_orbital_energy(distance_before, main.1.mass, orbiting.1.mass);

        let period = orbital_period(distance_before, main.1.mass);
        // Steps to complete one full orbit.
        let steps = (period / DT).round() as usize;

        let mut particles = [&mut main, &mut orbiting];
        for _ in 0..steps * orbit_count {
            let set: Vec<_> = particles.iter().map(|(.., pm)| *pm).collect();
            let accelerations = cm.compute(&set);

            for ((velocity, point_mass), acceleration) in particles.iter_mut().zip(accelerations) {
                *velocity += acceleration * DT;
                point_mass.position += *velocity * DT;
            }
        }

        let distance_after = (main.1.position - orbiting.1.position).mag();
        let energy_after = specific_orbital_energy(distance_after, main.1.mass, orbiting.1.mass);

        let error_energy = (1.0 - energy_before / energy_after).abs();
        dbg!(error_energy);
        assert!(error_energy.abs() < epsilon);

        let error_distance = (1.0 - distance_before / distance_after).abs();
        dbg!(error_distance);
        assert!(error_distance < epsilon)
    }
}
