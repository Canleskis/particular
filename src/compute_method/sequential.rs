use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use crate::{compute_method::ComputeMethod, vector::Normed};

/// A brute-force [`ComputeMethod`] using the CPU.
pub struct BruteForce;

impl<V, S> ComputeMethod<V, S> for BruteForce
where
    V: Copy
        + Default
        + AddAssign
        + SubAssign
        + Sub<Output = V>
        + Mul<S, Output = V>
        + Div<S, Output = V>
        + Normed<Output = S>,
    S: Copy + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, massive: Vec<(V, S)>, massless: Vec<(V, S)>) -> Vec<V> {
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
                let mag_2 = dir.length_squared();

                let f = dir / (mag_2 * V::sqrt(mag_2));

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
