use crate::{
    algorithms::{
        tree::{BarnesHutTree, BoundingBox, BoundingBoxDivide, Orthant, Tree},
        FromMassive, FromMassiveSIMD, InternalVector, IntoInternalVector, IntoSIMDElement,
        ParticleSet, PointMass, SIMDScalar, SIMDVector, Scalar,
    },
    compute_method::ComputeMethod,
};

/// A brute-force [`ComputeMethod`](ComputeMethod) using the CPU.
#[derive(Default, Clone, Copy)]
pub struct BruteForce;

impl<T, S, V> ComputeMethod<ParticleSet<T, S>, V> for BruteForce
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
{
    type Output = Box<dyn Iterator<Item = V>>;

    #[inline]
    fn compute(self, storage: ParticleSet<T, S>) -> Self::Output {
        let (massive, massless) = storage
            .particles
            .iter()
            .partition::<Vec<PointMass<_, _>>, _>(|p| p.is_massive());

        let massive_len = massive.len();

        let concat = &[massive, massless].concat()[..];
        let len = concat.len();

        let mut accelerations = vec![T::default(); len];

        for i in 0..massive_len {
            let p1 = concat[i];
            let mut acceleration = T::default();

            for j in (i + 1)..len {
                let p2 = concat[j];

                let dir = p2.position - p1.position;
                let mag_2 = dir.length_squared();

                let f = dir / (mag_2 * mag_2.sqrt());

                acceleration += f * p2.mass;
                accelerations[j] -= f * p1.mass;
            }

            accelerations[i] += acceleration;
        }

        let (mut massive_acc, mut massless_acc) = {
            let remainder = accelerations.split_off(massive_len);

            (accelerations.into_iter(), remainder.into_iter())
        };

        Box::new(
            storage
                .particles
                .into_iter()
                .filter_map(move |p| {
                    if p.is_massive() {
                        massive_acc.next()
                    } else {
                        massless_acc.next()
                    }
                })
                .map(V::from_internal),
        )
    }
}

/// A brute-force [`ComputeMethod`](ComputeMethod) using the CPU.
///
/// This differs from [`BruteForce`] by not iterating over the combinations of pair of particles, making it slower.
#[derive(Default, Clone, Copy)]
pub struct BruteForceAlt;

impl<T, S, V> ComputeMethod<FromMassive<T, S>, V> for BruteForceAlt
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
{
    type Output = Box<dyn Iterator<Item = V>>;

    #[inline]
    fn compute(self, storage: FromMassive<T, S>) -> Self::Output {
        Box::new(
            storage
                .affected
                .into_iter()
                .map(move |p1| {
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
                .map(V::from_internal),
        )
    }
}

/// A brute-force [`ComputeMethod`](ComputeMethod) using the CPU and explicit SIMD instructions using [ultraviolet](https://github.com/fu5ha/ultraviolet).
#[derive(Default, Clone, Copy)]
pub struct BruteForceSIMD;

impl<const LANES: usize, T, S, V> ComputeMethod<FromMassiveSIMD<LANES, T, S>, V> for BruteForceSIMD
where
    S: SIMDScalar<LANES> + 'static,
    T: SIMDVector<LANES, SIMDScalar = S> + 'static,
    V: IntoSIMDElement<T::Element, SIMDVector = T> + 'static,
{
    type Output = Box<dyn Iterator<Item = V>>;

    #[inline]
    fn compute(self, storage: FromMassiveSIMD<LANES, T, S>) -> Self::Output {
        Box::new(
            storage
                .affected
                .into_iter()
                .map(move |p1| {
                    storage.massive.iter().fold(T::default(), |acc, p2| {
                        let dir = p2.position - p1.position;
                        let mag_2 = dir.length_squared();

                        let grav_acc = dir * p2.mass * (mag_2.recip_sqrt() * mag_2.recip());

                        acc + grav_acc.nan_to_zero()
                    })
                })
                .map(V::from_after_reduce),
        )
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`](ComputeMethod) using the CPU.
#[derive(Default, Clone, Copy)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, const DIM: usize, const N: usize, V> ComputeMethod<FromMassive<T, S>, V> for BarnesHut<S>
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S, Array = [S; DIM]> + 'static,
    BoundingBox<T::Array>: BoundingBoxDivide<PointMass<T, S>, Output = (Orthant<N>, S)>,
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
{
    type Output = Box<dyn Iterator<Item = V>>;

    #[inline]
    fn compute(self, storage: FromMassive<T, S>) -> Self::Output {
        let mut tree = Tree::new();

        let bbox = BoundingBox::containing(storage.massive.iter().map(|p| p.position.into()));
        let root = tree.build_node(storage.massive, bbox);

        Box::new(
            storage
                .affected
                .into_iter()
                .map(move |p| V::from_internal(tree.acceleration_at(root, p.position, self.theta))),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(BruteForce, f32::EPSILON);
    }

    #[test]
    fn brute_force_alt() {
        tests::acceleration_computation(BruteForceAlt, f32::EPSILON);
    }

    #[test]
    fn brute_force_simd() {
        tests::acceleration_computation(BruteForceSIMD, 1e-2);
    }

    #[test]
    fn barnes_hut() {
        tests::acceleration_computation(BarnesHut { theta: 0.0 }, f32::EPSILON);
    }
}
