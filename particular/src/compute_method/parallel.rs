use crate::compute_method::{
    math::{Float, FloatVector, InfToZero, ReduceAdd, SIMDElement, Zero},
    storage::{ParticleSliceSystem, ParticleTreeSystem, PointMass},
    ComputeMethod,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

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
    fn compute(&mut self, system: ParticleSliceSystem<'_, V, S>) -> Self::Output {
        system
            .affected
            .par_iter()
            .map(|p1| {
                system.massive.iter().fold(V::ZERO, |acceleration, p2| {
                    acceleration + p1.acceleration_scalar::<true>(p2)
                })
            })
            .collect()
    }
}

/// Brute-force [`ComputeMethod`] using the CPU in parallel with
/// [rayon](https://github.com/rayon-rs/rayon) and simd vectors.
#[derive(Clone, Copy, Default)]
pub struct BruteForceSIMD<const L: usize>;

impl<const L: usize, V, S> ComputeMethod<ParticleSliceSystem<'_, V, S>> for BruteForceSIMD<L>
where
    V: SIMDElement<L> + Copy + Zero + Send + Sync,
    S: SIMDElement<L> + Copy + Zero + Sync,
    V::SIMD: FloatVector<Float = S::SIMD> + ReduceAdd + Copy + Send + Sync,
    S::SIMD: Float + InfToZero + Copy + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<'_, V, S>) -> Self::Output {
        let simd: Vec<_> = PointMass::slice_to_lanes(system.massive).collect();
        system
            .affected
            .par_iter()
            .map(|p1| {
                let p1 = PointMass::splat_lane(p1.position, p1.mass);
                simd.iter().fold(V::SIMD::ZERO, |acceleration, p2| {
                    acceleration + p1.acceleration_simd::<true>(p2)
                })
            })
            .map(ReduceAdd::reduce_add)
            .collect()
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`]
/// using the CPU in parallel with [rayon](https://github.com/rayon-rs/rayon) for the force
/// computation and scalar vectors.
#[derive(Clone, Copy, Default)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as
    /// [`BruteForceScalar`].
    pub theta: S,
}

impl<const X: usize, const D: usize, V, S> ComputeMethod<ParticleTreeSystem<'_, X, D, V, S>>
    for BarnesHut<S>
where
    V: Copy + FloatVector<Float = S> + Send + Sync,
    S: Copy + Float + PartialOrd + Sync,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(&mut self, system: ParticleTreeSystem<'_, X, D, V, S>) -> Self::Output {
        let tree = system.massive;
        system
            .affected
            .par_iter()
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
