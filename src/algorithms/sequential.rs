use crate::{
    algorithms::{
        internal, simd,
        tree::{BarnesHutTree, BoundingBox, SubDivide, Tree},
        MassiveAffected, MassiveAffectedSIMD, ParticleSet, PointMass,
    },
    compute_method::ComputeMethod,
};

/// Brute-force [`ComputeMethod`] using the CPU.
#[derive(Default, Clone, Copy)]
pub struct BruteForce;

impl<T, S, V> ComputeMethod<MassiveAffected<T, S>, V> for BruteForce
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::IntoVectorArray<T::Array, Vector = T>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: MassiveAffected<T, S>) -> Self::Output {
        storage
            .affected
            .iter()
            .map(|p1| {
                storage
                    .massive
                    .iter()
                    .fold(T::default(), |acceleration, &p2| {
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

/// Brute-force [`ComputeMethod`] using the CPU.
///
/// Faster than [`BruteForce`] because it computes the acceleration over the combination of pairs of particles instead of all the pairs.
///
/// For small numbers of particles however, prefer [`BruteForce`] or [`BruteForceCombinationsAlt`].
#[derive(Default, Clone, Copy)]
pub struct BruteForceCombinations;

impl<T, S, V> ComputeMethod<MassiveAffected<T, S>, V> for BruteForceCombinations
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::IntoVectorArray<T::Array, Vector = T>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: MassiveAffected<T, S>) -> Self::Output {
        let massive_len = storage.massive.len();
        let affected_len = storage.affected.len();

        let massless: Vec<_> = storage
            .affected
            .iter()
            .filter(|p| p.is_massless())
            .copied()
            .collect();

        let mut accelerations = vec![T::default(); affected_len];
        brute_force_combinations(
            &[storage.massive, massless].concat(),
            affected_len,
            massive_len,
            &mut accelerations,
        );

        let (mut massive_acc, mut massless_acc) = {
            let remainder = accelerations.split_off(massive_len);

            (accelerations.into_iter(), remainder.into_iter())
        };

        storage
            .affected
            .iter()
            .filter_map(|p| {
                if p.is_massive() {
                    massive_acc.next()
                } else {
                    massless_acc.next()
                }
            })
            .map(V::from_internal)
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU.
///
/// Typically faster than [`BruteForceCombinations`] except when most particles are massless (they are considered massive for this computation).
#[derive(Default, Clone, Copy)]
pub struct BruteForceCombinationsAlt;

impl<T, S, V> ComputeMethod<ParticleSet<T, S>, V> for BruteForceCombinationsAlt
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::IntoVectorArray<T::Array, Vector = T>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: ParticleSet<T, S>) -> Self::Output {
        let len = storage.0.len();
        let mut accelerations = vec![T::default(); len];
        brute_force_combinations(&storage.0, len, len, &mut accelerations);

        accelerations.into_iter().map(V::from_internal).collect()
    }
}

#[inline]
fn brute_force_combinations<T, S, A>(
    particles: &[PointMass<T, S>],
    affected_len: usize,
    massive_len: usize,
    accelerations: &mut A,
) where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    A: std::ops::IndexMut<usize, Output = T>,
{
    for i in 0..massive_len {
        let p1 = particles[i];
        let mut acceleration = T::default();

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
}

/// Brute-force [`ComputeMethod`] using the CPU and explicit SIMD instructions using [ultraviolet](https://github.com/fu5ha/ultraviolet).
#[derive(Default, Clone, Copy)]
pub struct BruteForceSIMD;

impl<const LANES: usize, T, S, V> ComputeMethod<MassiveAffectedSIMD<LANES, T, S>, V>
    for BruteForceSIMD
where
    S: simd::Scalar<LANES>,
    T: simd::Vector<LANES, Scalar = S>,
    V: simd::IntoVectorElement<T::Element, Vector = T>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: MassiveAffectedSIMD<LANES, T, S>) -> Self::Output {
        storage
            .affected
            .iter()
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

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`] using the CPU.
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
    V: internal::IntoVectorArray<T::Array, Vector = T>,
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
            .iter()
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
    fn brute_force_combinations() {
        tests::acceleration_computation(BruteForceCombinations, 1e-2);
    }

    #[test]
    fn brute_force_combinations_alt() {
        tests::acceleration_computation(BruteForceCombinationsAlt, 1e-2);
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
