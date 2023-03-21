use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use crate::vector::Normed;

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BruteForce;

impl<T, S> super::ComputeMethod<T, S> for BruteForce
where
    T: Copy
        + Default
        + AddAssign
        + SubAssign
        + Sub<Output = T>
        + Mul<S, Output = T>
        + Div<S, Output = T>
        + Normed<Output = S>,
    S: Copy + Default + PartialEq + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        let len = particles.len();

        let (mut massive, mut massless) = (Vec::with_capacity(len), Vec::with_capacity(len));

        for &(position, mu) in particles.iter() {
            if mu != S::default() {
                massive.push((position, mu));
            } else {
                massless.push((position, mu));
            }
        }

        let massive_len = massive.len();

        let concat = &[massive, massless].concat()[..];
        let len = concat.len();

        let mut accelerations = vec![T::default(); len];

        for i in 0..massive_len {
            let (pos1, mu1) = concat[i];
            let mut acceleration = T::default();

            for j in (i + 1)..len {
                let (pos2, mu2) = concat[j];

                let dir = pos2 - pos1;
                let mag_2 = dir.length_squared();

                let f = dir / (mag_2 * T::sqrt(mag_2));

                acceleration += f * mu2;
                accelerations[j] -= f * mu1;
            }

            accelerations[i] += acceleration;
        }

        let (mut massive_acc, mut massless_acc) = {
            let remainder = accelerations.split_off(massive_len);

            (accelerations.into_iter(), remainder.into_iter())
        };

        particles
            .iter()
            .filter_map(|(_, mu)| {
                if *mu != S::default() {
                    massive_acc.next()
                } else {
                    massless_acc.next()
                }
            })
            .collect()
    }
}

use super::tree::{
    acceleration::TreeAcceleration,
    bbox::{BoundingBox, BoundingBoxExtend},
    Tree, TreeBuilder, TreeData,
};

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, O> super::ComputeMethod<T, S> for BarnesHut<S>
where
    T: Copy + Default,
    S: Copy + Default + PartialEq,
    (T, S): Copy + TreeData,
    Tree<O, (T, S)>: TreeBuilder<BoundingBox<T>, (T, S)> + TreeAcceleration<T, S>,
    BoundingBox<T>: BoundingBoxExtend<Vector = T, Orthant = O>,
{
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        let massive: Vec<_> = particles
            .iter()
            .copied()
            .filter(|(_, mu)| *mu != S::default())
            .collect();

        let bbox = BoundingBox::containing(massive.iter().map(|p| p.0));
        let tree = Tree::build(massive, bbox);

        particles
            .iter()
            .map(|&(position, _)| tree.acceleration_at(position, Some(0), self.theta))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(BruteForce);
    }

    #[test]
    fn barnes_hut() {
        tests::acceleration_computation(BarnesHut { theta: 0.0 });
    }
}
