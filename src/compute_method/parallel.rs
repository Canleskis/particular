use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use crate::compute_method::Computable;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

/// A brute-force [`ComputeMethod`] using the CPU with [rayon](https://github.com/rayon-rs/rayon).
pub struct BruteForce;

impl<V, U> Computable<V, U> for BruteForce
where
    V: Copy
        + Send
        + Sync
        + Default
        + Sub<Output = V>
        + Mul<U, Output = V>
        + Div<U, Output = V>
        + AddAssign
        + SubAssign,
    U: Copy + Sync + Mul<Output = U>,
{
    #[inline]
    fn compute<F, G>(
        massive: Vec<(V, U)>,
        massless: Vec<(V, U)>,
        length_squared: F,
        sqrt: G,
    ) -> Vec<V>
    where
        F: Fn(V) -> U + Sync,
        G: Fn(U) -> U + Sync,
    {
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
                        let mag_2 = length_squared(dir);

                        acceleration += dir * particles[j].1 / (mag_2 * sqrt(mag_2))
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
