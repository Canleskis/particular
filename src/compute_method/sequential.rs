use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use crate::{compute_method::Computable};

/// A brute-force [`ComputeMethod`] using the CPU.
pub struct BruteForce;

impl<V, U> Computable<V, U> for BruteForce
where
    V: Clone
        + Copy
        + Default
        + Sub<Output = V>
        + Mul<U, Output = V>
        + Div<U, Output = V>
        + AddAssign
        + SubAssign,
    U: Clone + Copy + Mul<Output = U>,
{
    #[inline]
    fn compute<F, G>(
        massive: Vec<(V, U)>,
        massless: Vec<(V, U)>,
        length_squared: F,
        sqrt: G,
    ) -> Vec<V>
    where
        F: Fn(V) -> U,
        G: Fn(U) -> U,
    {
        let massive_len = massive.len();

        let particles = &[massive, massless].concat()[..];
        let len = particles.len();

        let mut accelerations = vec![V::default(); len];

        for i in 0..massive_len {
            let (pos1, mu1) = particles[i];
            let mut acceleration = V::default();

            for j in (i + 1)..len {
                let (pos2, mu2) = particles[j];

                let dir = pos2 - pos1;
                let mag_2 = length_squared(dir);

                let f = dir / (mag_2 * sqrt(mag_2));

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
