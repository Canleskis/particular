use crate::{
    algorithms::{
        internal, simd, tree::BarnesHutTree, MassiveAffectedInternal, MassiveAffectedSIMD,
        PointMass, TreeAffectedInternal,
    },
    compute_method::ComputeMethod,
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Brute-force [`ComputeMethod`] using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon).
#[derive(Default, Clone, Copy)]
pub struct BruteForce;

impl<const DIM: usize, S, V, T> ComputeMethod<MassiveAffectedInternal<DIM, S, V>, V> for BruteForce
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::ConvertInternalVector<DIM, S, Vector = T> + Send,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &MassiveAffectedInternal<DIM, S, V>) -> Self::Output {
        let storage = &storage.0;
        storage
            .affected
            .par_iter()
            .map(|p| V::from_internal(p.total_acceleration_internal(&storage.massive)))
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
    fn compute(self, storage: &MassiveAffectedSIMD<LANES, T, S>) -> Self::Output {
        storage
            .affected
            .par_iter()
            .map(|p| {
                V::from_reduced(
                    PointMass::new(T::splat(p.position), S::splat(p.mass))
                        .total_acceleration_simd(&storage.massive)
                        .reduce_add(),
                )
            })
            .collect()
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`] using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon) for the force computation.
#[derive(Default, Clone, Copy)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<const N: usize, const DIM: usize, S, V, T> ComputeMethod<TreeAffectedInternal<N, DIM, S, V>, V>
    for BarnesHut<S>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::ConvertInternalVector<DIM, S, Vector = T> + Send,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &TreeAffectedInternal<N, DIM, S, V>) -> Self::Output {
        let TreeAffectedInternal {
            tree,
            root,
            affected,
        } = storage;

        affected
            .par_iter()
            .map(|p| V::from_internal(tree.acceleration_at(*root, p.position, self.theta)))
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
        tests::circular_orbit_stability(BruteForce, 1_000, 1e-2);
    }

    #[test]
    fn brute_force_simd() {
        tests::acceleration_computation(BruteForceSIMD, 1e-2);
        tests::circular_orbit_stability(BruteForceSIMD, 1_000, 1e-2);
    }

    #[test]
    fn barnes_hut() {
        tests::acceleration_computation(BarnesHut { theta: 0.0 }, 1e-2);
        tests::circular_orbit_stability(BarnesHut { theta: 0.0 }, 1_000, 1e-2);
    }
}
