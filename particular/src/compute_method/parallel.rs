use crate::compute_method::{
    math::{BitAnd, CmpNe, Float, FloatVector, Reduce, SIMDElement, Zero, SIMD},
    storage::{ParticleSliceSystem, ParticleTreeSystem, PointMass},
    ComputeMethod,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Brute-force [`ComputeMethod`] using the CPU in parallel with
/// [rayon](https://github.com/rayon-rs/rayon) and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSoftenedScalar<S> {
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceSoftenedScalar<S>
where
    V: FloatVector<Float = S> + Copy + Send + Sync,
    S: Float + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        system
            .affected
            .par_iter()
            .map(|p1| {
                system.massive.iter().fold(V::ZERO, |acceleration, p2| {
                    acceleration + p1.force_scalar::<true>(p2.position, p2.mass, self.softening)
                })
            })
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU in parallel with
/// [rayon](https://github.com/rayon-rs/rayon) and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceScalar;

impl<V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceScalar
where
    V: FloatVector<Float = S> + Copy + Send + Sync,
    S: Float + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        BruteForceSoftenedScalar { softening: S::ZERO }.compute(system)
    }
}

/// Brute-force [`ComputeMethod`] using the CPU in parallel with
/// [rayon](https://github.com/rayon-rs/rayon) and simd vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSoftenedSIMD<const L: usize, S> {
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<const L: usize, V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>>
    for BruteForceSoftenedSIMD<L, S>
where
    V: SIMDElement<L> + Zero + Copy + Send + Sync,
    S: SIMDElement<L> + Float + Copy + Sync,
    V::SIMD: FloatVector<Float = S::SIMD> + Reduce + Copy + Send + Sync,
    S::SIMD: Float + BitAnd<Output = S::SIMD> + CmpNe<Output = S::SIMD> + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        let simd_massive: Vec<_> = PointMass::slice_to_lanes(system.massive).collect();
        let simd_softening = S::SIMD::splat(self.softening);
        system
            .affected
            .par_iter()
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

/// Brute-force [`ComputeMethod`] using the CPU in parallel with
/// [rayon](https://github.com/rayon-rs/rayon) and simd vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSIMD<const L: usize>;

impl<const L: usize, V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceSIMD<L>
where
    V: SIMDElement<L> + Zero + Copy + Send + Sync,
    S: SIMDElement<L> + Float + Copy + Sync,
    V::SIMD: FloatVector<Float = S::SIMD> + Reduce + Copy + Send + Sync,
    S::SIMD: Float + BitAnd<Output = S::SIMD> + CmpNe<Output = S::SIMD> + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        BruteForceSoftenedSIMD { softening: S::ZERO }.compute(system)
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`]
/// using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon) for the force
/// computation and scalar vectors.
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
    V: FloatVector<Float = S> + Copy + Send + Sync,
    S: Float + PartialOrd + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleTreeSystem<X, D, V, S>) -> Self::Output {
        let tree = system.massive;
        system
            .affected
            .par_iter()
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
    V: FloatVector<Float = S> + Copy + Send + Sync,
    S: Float + PartialOrd + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleTreeSystem<X, D, V, S>) -> Self::Output {
        BarnesHutSoftened {
            theta: self.theta,
            softening: S::ZERO,
        }
        .compute(system)
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
