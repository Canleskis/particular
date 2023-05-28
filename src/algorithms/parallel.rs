use crate::{
    algorithms::{
        tree::{BarnesHutTree, BoundingBox, BoundingBoxDivide, Orthant, Tree},
        FromMassive, InternalVector, PointMass, Scalar, Vector,
    },
    compute_method::ComputeMethod,
};

/// A brute-force [`ComputeMethod`](ComputeMethod) using the CPU with [rayon](https://github.com/rayon-rs/rayon).
#[derive(Clone, Copy)]
pub struct BruteForce;

impl<T, S, V> ComputeMethod<FromMassive<T, S>, V> for BruteForce
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: Vector<T::Array, Internal = T> + Send + 'static,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: FromMassive<T, S>) -> Self::Output {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        storage
            .affected
            .into_par_iter()
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
            .map(V::from_internal)
            .collect()
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) [`ComputeMethod`](ComputeMethod) using the CPU with [rayon](https://github.com/rayon-rs/rayon) for the force computation.
#[derive(Clone, Copy)]
pub struct BarnesHut<S> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, behaves the same as [`BruteForce`].
    pub theta: S,
}

impl<T, S, const DIM: usize, const N: usize, V> ComputeMethod<FromMassive<T, S>, V> for BarnesHut<S>
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S, Array = [S; DIM]> + 'static,
    BoundingBox<T::Array>: BoundingBoxDivide<PointMass<T, S>, Output = (Orthant<N>, S)>,
    V: Vector<T::Array, Internal = T> + Send + 'static,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: FromMassive<T, S>) -> Self::Output {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let mut tree = Tree::new();

        let bbox = BoundingBox::containing(storage.massive.iter().map(|p| p.position.into()));
        let root = tree.build_node(storage.massive, bbox);

        storage
            .affected
            .into_par_iter()
            .map(move |p| V::from_internal(tree.acceleration_at(root, p.position, self.theta)))
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
