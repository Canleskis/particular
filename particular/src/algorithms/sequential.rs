use crate::{
    algorithms::{
        internal, simd, tree::BarnesHutAcceleration, vector, MassiveAffectedInternal,
        MassiveAffectedSIMD, ParticleSetInternal, PointMass, TreeAffectedInternal,
    },
    compute_method::ComputeMethod,
};

/// Brute-force [`ComputeMethod`] using the CPU.
#[derive(Default, Clone, Copy)]
pub struct BruteForce;

impl<const D: usize, S, V> ComputeMethod<MassiveAffectedInternal<D, S, V>, V> for BruteForce
where
    S: internal::Scalar,
    V: internal::ConvertInternal<D, S>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &MassiveAffectedInternal<D, S, V>) -> Self::Output {
        let storage = &storage.0;
        storage
            .affected
            .iter()
            .map(|p| V::from_internal(p.total_acceleration_internal(&storage.massive)))
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU.
///
/// Typically faster than [`BruteForce`] because it computes the acceleration over the combination of pairs of particles instead of all the pairs.
///
/// For small numbers of particles however, prefer [`BruteForce`] or [`BruteForcePairsAlt`].
#[derive(Default, Clone, Copy)]
pub struct BruteForcePairs;

impl<const D: usize, S, V> ComputeMethod<MassiveAffectedInternal<D, S, V>, V> for BruteForcePairs
where
    S: internal::Scalar,
    V: internal::ConvertInternal<D, S>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &MassiveAffectedInternal<D, S, V>) -> Self::Output {
        let storage = &storage.0;
        let massive_len = storage.massive.len();
        let affected_len = storage.affected.len();

        let particles: Vec<_> = storage
            .massive
            .iter()
            .chain(storage.affected.iter().filter(|p| p.is_massless()))
            .copied()
            .collect();

        let accelerations = BruteForcePairsCore::new(
            vec![vector::Zero::ZERO; affected_len],
            massive_len,
            affected_len,
        )
        .compute(&particles);

        let (mut massive_acc, mut massless_acc) = {
            let (massive_acc, massless_acc) = accelerations.split_at(massive_len);

            (massive_acc.iter().copied(), massless_acc.iter().copied())
        };

        storage
            .affected
            .iter()
            .filter_map(|p| {
                if p.is_massless() {
                    massless_acc.next()
                } else {
                    massive_acc.next()
                }
            })
            .map(V::from_internal)
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU.
///
/// Faster than [`BruteForcePairs`] except when most particles are massless (they are considered massive for this computation).
#[derive(Default, Clone, Copy)]
pub struct BruteForcePairsAlt;

impl<const D: usize, S, V> ComputeMethod<ParticleSetInternal<D, S, V>, V> for BruteForcePairsAlt
where
    S: internal::Scalar,
    V: internal::ConvertInternal<D, S>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &ParticleSetInternal<D, S, V>) -> Self::Output {
        let len = storage.0.len();

        BruteForcePairsCore::new(vec![vector::Zero::ZERO; len], len, len)
            .compute(&storage.0)
            .into_iter()
            .map(V::from_internal)
            .collect()
    }
}

/// Low-level brute-force [`ComputeMethod`] using the CPU.
/// Useful if you are looking for the smallest overhead (most performance) with small numbers of particles (fixed or not).
///
/// Can only be used by calling [`compute`](ComputeMethod::compute) directly and with vectors implementing [`internal::Vector`].
/// If `massive_len` != `affected_len`, all the massive particles should be contained before the affected particles in the given storage.
#[derive(Default, Clone, Copy)]
pub struct BruteForcePairsCore<A> {
    /// Initial value of the output accelerations.
    pub accelerations: A,
    /// Number of particles with mass for the computation.
    pub massive_len: usize,
    /// Number of affected particles for the computation.
    pub affected_len: usize,
}

impl<A> BruteForcePairsCore<A> {
    /// Creates a new [`BruteForcePairsCore`] with the given massive and affected number of particles.
    #[inline]
    pub fn new(accelerations: A, massive_len: usize, affected_len: usize) -> Self {
        Self {
            accelerations,
            massive_len,
            affected_len,
        }
    }
}

impl<I, T, S, A> ComputeMethod<I, T> for BruteForcePairsCore<A>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    I: std::ops::Index<usize, Output = PointMass<T, S>>,
    A: std::ops::IndexMut<usize, Output = T> + IntoIterator<Item = T>,
{
    type Output = A;

    #[inline]
    fn compute(self, particles: &I) -> Self::Output {
        let Self {
            mut accelerations,
            massive_len,
            affected_len,
        } = self;

        for i in 0..massive_len {
            let p1 = particles[i];
            let mut acceleration = T::ZERO;

            for j in (i + 1)..affected_len {
                let p2 = particles[j];

                let dir = p2.position - p1.position;
                let mag_2 = dir.length_squared();
                let f = dir / (mag_2 * mag_2.sqrt());

                acceleration += f * p2.mass;
                accelerations[j] -= f * p1.mass;
            }

            accelerations[i] += acceleration;
        }

        accelerations
    }
}

/// Brute-force [`ComputeMethod`] using the CPU and explicit SIMD instructions using [ultraviolet](https://github.com/fu5ha/ultraviolet).
#[derive(Default, Clone, Copy)]
pub struct BruteForceSIMD;

impl<const L: usize, const D: usize, S, V> ComputeMethod<MassiveAffectedSIMD<L, D, S, V>, V>
    for BruteForceSIMD
where
    S: Copy,
    V: simd::ConvertSIMD<L, D, S>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &MassiveAffectedSIMD<L, D, S, V>) -> Self::Output {
        storage
            .0
            .affected
            .iter()
            .map(|p| {
                V::from(simd::ReduceAdd::reduce_add(
                    PointMass::new(simd::SIMD::splat(p.position), simd::SIMD::splat(p.mass))
                        .total_acceleration_simd(&storage.0.massive),
                ))
            })
            .collect()
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`] using the CPU.
#[derive(Default, Clone, Copy)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<const N: usize, const D: usize, S, V> ComputeMethod<TreeAffectedInternal<N, D, S, V>, V>
    for BarnesHut<S>
where
    S: internal::Scalar,
    V: internal::ConvertInternal<D, S>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &TreeAffectedInternal<N, D, S, V>) -> Self::Output {
        let TreeAffectedInternal {
            tree,
            root,
            affected,
        } = storage;

        affected
            .iter()
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
    fn brute_force_pairs() {
        tests::acceleration_computation(BruteForcePairs, 1e-2);
        tests::circular_orbit_stability(BruteForcePairs, 1_000, 1e-2);
    }

    #[test]
    fn brute_force_pairs_alt() {
        tests::acceleration_computation(BruteForcePairsAlt, 1e-2);
        tests::circular_orbit_stability(BruteForcePairsAlt, 1_000, 1e-2);
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
