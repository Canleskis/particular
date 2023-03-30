use std::ops::{Add, Div, Mul, Sub};

use crate::vector::Normed;

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the CPU with [rayon](https://github.com/rayon-rs/rayon).
pub struct BruteForce;

impl<T, S> super::ComputeMethod<T, S> for BruteForce
where
    T: Copy
        + Default
        + Send
        + Sync
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<S, Output = T>
        + Div<S, Output = T>
        + Normed<Output = S>,
    S: Copy + Default + Sync + PartialEq + Mul<Output = S>,
{
    #[inline]
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let massive: Vec<_> = particles
            .iter()
            .filter(|(_, mu)| *mu != S::default())
            .collect();

        particles
            .par_iter()
            .map(|&(position1, _)| {
                massive
                    .iter()
                    .fold(T::default(), |acceleration, &&(position2, mass2)| {
                        let dir = position2 - position1;
                        let mag_2 = dir.length_squared();

                        let grav_acc = if mag_2 != S::default() {
                            dir * mass2 / (mag_2 * T::sqrt(mag_2))
                        } else {
                            dir
                        };

                        acceleration + grav_acc
                    })
            })
            .collect()
    }
}

use super::tree::{
    acceleration::TreeAcceleration,
    bbox::{BoundingBox, BoundingBoxExtend},
    Tree, TreeBuilder, TreeData,
};

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`](super::ComputeMethod) using the CPU with [rayon](https://github.com/rayon-rs/rayon) for the force computation.
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, O> super::ComputeMethod<T, S> for BarnesHut<S>
where
    O: Sync,
    T: Copy + Default + Send + Sync,
    S: Copy + Default + Sync + PartialEq,
    (T, S): Copy + Sync + TreeData<Output = (T, S)>,
    Tree<O, (T, S)>: TreeBuilder<BoundingBox<T>, (T, S)> + TreeAcceleration<T, S>,
    BoundingBox<T>: BoundingBoxExtend<Vector = T, Orthant = O>,
{
    fn compute(&mut self, particles: &[(T, S)]) -> Vec<T> {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let mut tree = Tree::default();

        let massive: Vec<_> = particles
            .iter()
            .filter(|(_, mu)| *mu != S::default())
            .copied()
            .collect();

        let bbox = BoundingBox::containing(massive.iter().map(|p| p.0));
        let root = tree.build_node(massive, bbox);

        particles
            .par_iter()
            .map(|&(position, _)| tree.acceleration_at(position, root, self.theta))
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
