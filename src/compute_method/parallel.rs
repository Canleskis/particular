use crate::compute_method::ComputeMethod;

use glam::Vec3A;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A brute-force [`ComputeMethod`] using the CPU with [rayon](https://github.com/rayon-rs/rayon).
pub struct BruteForce;

impl ComputeMethod<Vec3A, f32> for BruteForce {
    #[inline]
    fn compute(&mut self, massive: Vec<(Vec3A, f32)>, massless: Vec<(Vec3A, f32)>) -> Vec<Vec3A> {
        let massive_len = massive.len();

        let particles = &[massive, massless].concat()[..];
        let len = particles.len();

        (0..len)
            .into_par_iter()
            .map(|i| {
                let mut acceleration = Vec3A::ZERO;

                for j in 0..massive_len {
                    if i != j {
                        let dir = particles[j].0 - particles[i].0;
                        let mag_2 = dir.length_squared();

                        acceleration += dir * particles[j].1 / (mag_2 * mag_2.sqrt())
                    }
                }

                acceleration
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::compute_method::{parallel, tests};

    #[test]
    fn brute_force() {
        tests::acceleration_computation(parallel::BruteForce);
    }
}
