use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use crate::{compute_method::ComputeMethod, vector::Normed};

/// A brute-force [`ComputeMethod`] using the CPU with [rayon](https://github.com/rayon-rs/rayon).
pub struct BruteForce;

impl<V, S> ComputeMethod<V, S> for BruteForce
where
    V: Copy
        + Send
        + Sync
        + Default
        + AddAssign
        + SubAssign
        + Sub<Output = V>
        + Mul<S, Output = V>
        + Div<S, Output = V>
        + Normed<Output = S>,
    S: Copy + Sync + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, massive: Vec<(V, S)>, massless: Vec<(V, S)>) -> Vec<V> {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let massive_len = massive.len();

        let particles = &[massive, massless].concat()[..];
        let len = particles.len();

        (0..len)
            .into_par_iter()
            .map(|i| {
                let mut acceleration = V::default();

                for j in 0..massive_len {
                    if i != j {
                        let dir = particles[j].0 - particles[i].0;
                        let mag_2 = dir.length_squared();

                        acceleration += dir * particles[j].1 / (mag_2 * V::sqrt(mag_2))
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
