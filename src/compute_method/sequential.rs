use crate::vector::{InternalVector, Scalar};

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BruteForce;

impl<T, S> super::ComputeMethod<T, S> for BruteForce
where
    S: Scalar,
    T: InternalVector<Scalar = S>,
{
    #[inline]
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        let (massive, massless): (Vec<_>, Vec<_>) =
            particles.iter().partition(|(_, mu)| *mu != S::default());

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

                let f = dir / (mag_2 * mag_2.sqrt());

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
    barnes_hut::BarnesHutTree,
    bbox::{BoundingBox, BoundingBoxDivide, Orthant},
    Tree,
};

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`](super::ComputeMethod) using the CPU.
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, const DIM: usize, const N: usize> super::ComputeMethod<T, S> for BarnesHut<S>
where
    S: Scalar,
    T: InternalVector<Scalar = S, Array = [S; DIM]>,
    BoundingBox<T::Array>: BoundingBoxDivide<(T, S), Output = (Orthant<N>, S)>,
{
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        let mut tree = Tree::default();

        let massive: Vec<_> = particles
            .iter()
            .filter(|(_, mu)| *mu != S::default())
            .copied()
            .collect();

        let bbox = BoundingBox::containing(massive.iter().map(|p| p.0.into()));
        let root = tree.build_node(massive, bbox);

        particles
            .iter()
            .map(|&(position, _)| tree.acceleration_at(root, position, self.theta))
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
