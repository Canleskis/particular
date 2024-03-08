use crate::compute_method::{
    math::{BitAnd, CmpNe, Float, FloatVector, Reduce, SIMDElement, Zero, SIMD},
    storage::{
        ParticleOrdered, ParticleReordered, ParticleSliceSystem, ParticleTreeSystem, PointMass,
    },
    ComputeMethod,
};

/// Brute-force [`ComputeMethod`] using the CPU and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSoftenedScalar<S> {
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceSoftenedScalar<S>
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
                    acceleration + p1.force_scalar::<true>(p2.position, p2.mass, self.softening)
                })
            })
            .collect()
    }
}

/// Same as [`BruteForceSoftenedScalar`], but with no softening.
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
                    acceleration + p1.force_scalar::<true>(p2.position, p2.mass, S::ZERO)
                })
            })
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU and simd vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSoftenedSIMD<const L: usize, S> {
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<const L: usize, V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>>
    for BruteForceSoftenedSIMD<L, S>
where
    V: SIMDElement<L> + Zero + Copy,
    S: SIMDElement<L> + Float + Copy,
    V::SIMD: FloatVector<Float = S::SIMD> + Reduce + Copy,
    S::SIMD: Float + BitAnd<Output = S::SIMD> + CmpNe<Output = S::SIMD> + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        let simd_massive: Vec<_> = PointMass::slice_to_lanes(system.massive).collect();
        let simd_softening = S::SIMD::splat(self.softening);
        system
            .affected
            .iter()
            .map(|p1| {
                let p1 = PointMass::splat_lane(p1.position, p1.mass);
                simd_massive.iter().fold(V::SIMD::ZERO, |acceleration, p2| {
                    acceleration + p1.force_simd::<true>(p2.position, p2.mass, simd_softening)
                })
            })
            .map(Reduce::reduce_sum)
            .collect()
    }
}

/// Same as [`BruteForceSoftenedSIMD`], but with no softening.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSIMD<const L: usize>;

impl<const L: usize, V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceSIMD<L>
where
    V: SIMDElement<L> + Zero + Copy,
    S: SIMDElement<L> + Float + Copy,
    V::SIMD: FloatVector<Float = S::SIMD> + Reduce + Copy,
    S::SIMD: Float + BitAnd<Output = S::SIMD> + CmpNe<Output = S::SIMD> + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        let simd_massive: Vec<_> = PointMass::slice_to_lanes(system.massive).collect();
        system
            .affected
            .iter()
            .map(|p1| {
                let p1 = PointMass::splat_lane(p1.position, p1.mass);
                simd_massive.iter().fold(V::SIMD::ZERO, |acceleration, p2| {
                    acceleration + p1.force_simd::<true>(p2.position, p2.mass, S::SIMD::ZERO)
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
pub struct BruteForcePairsSoftened<S> {
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<S> BruteForcePairsSoftened<S> {
    #[inline]
    fn accelerations_pairs<T>(&self, particles: &[PointMass<T, S>], massive_len: usize) -> Vec<T>
    where
        S: Float + Copy,
        T: FloatVector<Float = S> + Copy,
    {
        let len = particles.len();
        let mut accelerations = vec![Zero::ZERO; len];

        for i in 0..massive_len {
            let p1 = particles[i];
            let mut acceleration = Zero::ZERO;

            for j in (i + 1)..len {
                let p2 = particles[j];
                let force_dir = p1.force_scalar::<false>(p2.position, S::ONE, self.softening);

                acceleration += force_dir * p2.mass;
                accelerations[j] -= force_dir * p1.mass;
            }

            accelerations[i] += acceleration;
        }

        accelerations
    }
}

impl<V, S> ComputeMethod<&[PointMass<V, S>]> for BruteForcePairsSoftened<S>
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: &[PointMass<V, S>]) -> Self::Output {
        self.accelerations_pairs(storage, storage.len())
    }
}

impl<V, S> ComputeMethod<&ParticleOrdered<V, S>> for BruteForcePairsSoftened<S>
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: &ParticleOrdered<V, S>) -> Self::Output {
        self.accelerations_pairs(storage.particles(), storage.massive_len())
    }
}

impl<V, S> ComputeMethod<ParticleReordered<'_, V, S>> for BruteForcePairsSoftened<S>
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
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

/// Same as [`BruteForcePairsSoftened`], but with no softening.
#[derive(Clone, Copy, Default)]
pub struct BruteForcePairs;

// We use the same implementationq as `BruteForcePairsSoftened` here because the compiler is able
// to optimize the softening away, unlike for the other compute methods.

impl<V, S> ComputeMethod<&[PointMass<V, S>]> for BruteForcePairs
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: &[PointMass<V, S>]) -> Self::Output {
        BruteForcePairsSoftened { softening: S::ZERO }.compute(storage)
    }
}

impl<V, S> ComputeMethod<&ParticleOrdered<V, S>> for BruteForcePairs
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: &ParticleOrdered<V, S>) -> Self::Output {
        BruteForcePairsSoftened { softening: S::ZERO }.compute(storage)
    }
}

impl<V, S> ComputeMethod<ParticleReordered<'_, V, S>> for BruteForcePairs
where
    V: FloatVector<Float = S> + Copy,
    S: Float + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, storage: ParticleReordered<V, S>) -> Self::Output {
        BruteForcePairsSoftened { softening: S::ZERO }.compute(storage)
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`]
/// using the CPU and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BarnesHutSoftened<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as
    /// [`BruteForceScalar`].
    pub theta: S,
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<const X: usize, const D: usize, V, S> ComputeMethod<ParticleTreeSystem<'_, X, D, V, S>>
    for BarnesHutSoftened<S>
where
    V: FloatVector<Float = S> + Copy,
    S: Float + PartialOrd + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleTreeSystem<X, D, V, S>) -> Self::Output {
        let tree = system.massive;
        system
            .affected
            .iter()
            .map(|p| p.acceleration_tree(tree.get(), tree.root(), self.theta, self.softening))
            .collect()
    }
}

/// Same as [`BarnesHutSoftened`], but with no softening.
#[derive(Clone, Copy, Default)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as
    /// [`BruteForceScalar`].
    pub theta: S,
}

impl<const X: usize, const D: usize, V, S> ComputeMethod<ParticleTreeSystem<'_, X, D, V, S>>
    for BarnesHut<S>
where
    V: FloatVector<Float = S> + Copy,
    S: Float + PartialOrd + Copy,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleTreeSystem<X, D, V, S>) -> Self::Output {
        let tree = system.massive;
        system
            .affected
            .iter()
            .map(|p| p.acceleration_tree(tree.get(), tree.root(), self.theta, S::ZERO))
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
