use crate::compute_method::{
    math::{Float, FloatVector, InfToZero, Reduce, SIMDElement, Zero},
    storage::{
        ParticleOrdered, ParticleReordered, ParticleSliceSystem, ParticleTreeSystem, PointMass,
    },
    ComputeMethod,
};

/// Brute-force [`ComputeMethod`] using the CPU and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceScalar;

impl<V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceScalar
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        system
            .affected
            .iter()
            .map(|p1| {
                system.massive.iter().fold(V::ZERO, |acceleration, p2| {
                    acceleration + p1.acceleration_scalar::<true>(p2)
                })
            })
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU and simd vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSIMD<const L: usize>;

impl<const L: usize, V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceSIMD<L>
where
    V: SIMDElement<L> + Copy + Zero,
    S: SIMDElement<L> + Copy + Zero,
    V::SIMD: FloatVector<Float = S::SIMD> + Copy + Reduce,
    S::SIMD: Float + Copy + InfToZero,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        let simd: Vec<_> = PointMass::slice_to_lanes(system.massive).collect();
        system
            .affected
            .iter()
            .map(|p1| {
                let p1 = PointMass::splat_lane(p1.position, p1.mass);
                simd.iter().fold(V::SIMD::ZERO, |acceleration, p2| {
                    acceleration + p1.acceleration_simd::<true>(p2)
                })
            })
            .map(Reduce::reduce_sum)
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU and scalar vectors.
///
/// Typically faster than [`BruteForceScalar`] because it computes the acceleration over the
/// combination of pairs of particles instead of all the pairs.
#[derive(Clone, Copy, Default)]
pub struct BruteForcePairs;

impl BruteForcePairs {
    #[inline]
    fn accelerations_pairs<T, S>(particles: &[PointMass<T, S>], massive_len: usize) -> Vec<T>
    where
        S: Copy + Float,
        T: Copy + FloatVector<Float = S>,
    {
        let len = particles.len();
        let mut accelerations = vec![Zero::ZERO; len];

        for i in 0..massive_len {
            let p1 = particles[i];
            let mut acceleration = Zero::ZERO;

            for j in (i + 1)..len {
                let p2 = particles[j];
                let force_dir = p1.force_mul_mass_scalar::<false>(p2.position, S::ONE);

                acceleration += force_dir * p2.mass;
                accelerations[j] -= force_dir * p1.mass;
            }

            accelerations[i] += acceleration;
        }

        accelerations
    }
}

impl<V, S> ComputeMethod<&[PointMass<V, S>]> for BruteForcePairs
where
    V: Copy + FloatVector<Float = S>,
    S: Copy + Float,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: &[PointMass<V, S>]) -> Self::Output {
        Self::accelerations_pairs(storage, storage.len())
    }
}

impl<V, S> ComputeMethod<&ParticleOrdered<V, S>> for BruteForcePairs
where
    V: Copy + FloatVector<Float = S>,
    S: Copy + Float,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: &ParticleOrdered<V, S>) -> Self::Output {
        Self::accelerations_pairs(storage.particles(), storage.massive_len())
    }
}

impl<V, S> ComputeMethod<ParticleReordered<'_, V, S>> for BruteForcePairs
where
    V: Copy + FloatVector<Float = S>,
    S: Copy + Float,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, reordered: ParticleReordered<V, S>) -> Self::Output {
        let accelerations = self.compute(reordered.ordered());
        let (mut massive_accelerations, mut massless_accelerations) = {
            let (a, b) = accelerations.split_at(reordered.massive_len());
            (a.iter(), b.iter())
        };

        reordered
            .unordered
            .iter()
            .filter_map(|p| {
                if p.is_massless() {
                    massless_accelerations.next()
                } else {
                    massive_accelerations.next()
                }
                .copied()
            })
            .collect()
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`]
/// using the CPU and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as
    /// [`BruteForceScalar`].
    pub theta: S,
}

impl<const X: usize, const D: usize, V, S> ComputeMethod<ParticleTreeSystem<'_, X, D, V, S>>
    for BarnesHut<S>
where
    V: Copy + FloatVector<Float = S>,
    S: Copy + Float + PartialOrd,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleTreeSystem<X, D, V, S>) -> Self::Output {
        let tree = system.massive;
        system
            .affected
            .iter()
            .map(|p| p.acceleration_tree(tree.get(), tree.root(), self.theta))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force_scalar() {
        tests::acceleration_error(BruteForceScalar, 1e-2);
        tests::circular_orbit_stability(BruteForceScalar, 1_000, 1e-2);
    }

    #[test]
    fn brute_force_simd() {
        tests::acceleration_error(BruteForceSIMD::<8>, 1e-2);
        tests::circular_orbit_stability(BruteForceSIMD::<8>, 1_000, 1e-2);
    }

    #[test]
    fn brute_force_pairs() {
        tests::acceleration_error(BruteForcePairs, 1e-2);
        tests::circular_orbit_stability(BruteForcePairs, 1_000, 1e-2);
    }

    #[test]
    fn barnes_hut() {
        tests::acceleration_error(BarnesHut { theta: 0.0 }, 1e-2);
        tests::circular_orbit_stability(BarnesHut { theta: 0.0 }, 1_000, 1e-2);
    }

    #[test]
    fn barnes_hut_05() {
        tests::acceleration_error(BarnesHut { theta: 0.5 }, 1e-1);
        tests::circular_orbit_stability(BarnesHut { theta: 0.5 }, 1_000, 1e-1);
    }
}
