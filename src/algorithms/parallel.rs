use crate::{
    algorithms::{
        internal, simd,
        tree::{BarnesHutTree, BoundingBox, SubDivide, Tree},
        MassiveAffected, MassiveAffectedSIMD, PointMass,
    },
    compute_method::ComputeMethod,
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Brute-force [`ComputeMethod`] using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon).
#[derive(Default, Clone, Copy)]
pub struct BruteForce;

impl<T, S, V> ComputeMethod<MassiveAffected<T, S>, V> for BruteForce
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::IntoVectorArray<T::Array, Vector = T> + Send,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: MassiveAffected<T, S>) -> Self::Output {
        storage
            .affected
            .par_iter()
            .map(|p1| {
                storage
                    .massive
                    .iter()
                    .fold(T::default(), |acceleration, p2| {
                        let dir = p2.position - p1.position;
                        let mag_2 = dir.length_squared();
                        let grav_acc = if mag_2 != S::default() {
                            dir * p2.mass / (mag_2 * mag_2.sqrt())
                        } else {
                            dir
                        };

                        acceleration + grav_acc
                    })
            })
            .map(V::from_internal)
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon) and explicit SIMD instructions using [ultraviolet](https://github.com/fu5ha/ultraviolet).
#[derive(Default, Clone, Copy)]
pub struct BruteForceSIMD;

impl<const LANES: usize, T, S, V> ComputeMethod<MassiveAffectedSIMD<LANES, T, S>, V>
    for BruteForceSIMD
where
    S: simd::Scalar<LANES>,
    T: simd::Vector<LANES, Scalar = S>,
    V: simd::IntoVectorElement<T::Element, Vector = T> + Send,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: MassiveAffectedSIMD<LANES, T, S>) -> Self::Output {
        storage
            .affected
            .par_iter()
            .map(|p1| {
                let p1 = PointMass::new(T::splat(p1.position), S::splat(p1.mass));
                storage.massive.iter().fold(T::default(), |acc, p2| {
                    let dir = p2.position - p1.position;
                    let mag_2 = dir.length_squared();
                    let grav_acc = dir * p2.mass * (mag_2.recip_sqrt() * mag_2.recip());

                    acc + grav_acc.nan_to_zero()
                })
            })
            .map(V::from_after_reduce)
            .collect()
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`] using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon) for the force computation.
#[derive(Default, Clone, Copy)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, const DIM: usize, const N: usize, V> ComputeMethod<MassiveAffected<T, S>, V>
    for BarnesHut<S>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S, Array = [S; DIM]>,
    V: internal::IntoVectorArray<T::Array, Vector = T> + Send,
    BoundingBox<T::Array>: SubDivide<Divison = [BoundingBox<T::Array>; N]>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: MassiveAffected<T, S>) -> Self::Output {
        let mut tree = Tree::new();
        let bbox = BoundingBox::square_with(storage.massive.iter().map(|p| p.position.into()));
        let root = tree.build_node(&storage.massive, bbox);

        storage
            .affected
            .par_iter()
            .map(|p| V::from_internal(tree.acceleration_at(root, p.position, self.theta)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(BruteForce, 1e-2);
    }

    #[test]
    fn brute_force_simd() {
        tests::acceleration_computation(BruteForceSIMD, 1e-2);
    }

    #[test]
    fn barnes_hut() {
        tests::acceleration_computation(BarnesHut { theta: 0.0 }, 1e-2);
    }
}
