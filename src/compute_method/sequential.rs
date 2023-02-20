use crate::compute_method::ComputeMethod;

use glam::Vec3A;

/// A brute-force [`ComputeMethod`] using the CPU.
pub struct BruteForce;

impl ComputeMethod<Vec3A, f32> for BruteForce {
    #[inline]
    fn compute(&mut self, massive: Vec<(Vec3A, f32)>, massless: Vec<(Vec3A, f32)>) -> Vec<Vec3A> {
        let massive_len = massive.len();

        let particles = &[massive, massless].concat()[..];
        let len = particles.len();

        let mut accelerations = vec![Vec3A::ZERO; len];

        for i in 0..massive_len {
            let (pos1, mu1) = particles[i];
            let mut acceleration = Vec3A::ZERO;

            for j in (i + 1)..len {
                let (pos2, mu2) = particles[j];

                let dir = pos2 - pos1;
                let mag_2 = dir.length_squared();

                let f = dir / (mag_2 * mag_2.sqrt());

                acceleration += f * mu2;
                accelerations[j] -= f * mu1;
            }

            accelerations[i] += acceleration;
        }

        accelerations
    }
}

#[cfg(test)]
mod tests {
    use crate::compute_method::{sequential, tests};

    #[test]
    fn brute_force() {
        tests::acceleration_computation(sequential::BruteForce);
    }
}
