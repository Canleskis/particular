use super::{
    bbox::{Orthant, Positionable},
    Node, NodeID, Tree, TreeData,
};

use crate::vector::{InternalVector, Scalar};

/// Trait to compute the acceleration of particles using the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm.
pub trait BarnesHutTree<T, S> {
    /// Computes the acceleration at the given position from a node of the tree.
    fn acceleration_at(&self, node: Option<NodeID>, position: T, theta: S) -> T;
}

impl<const N: usize, T, S> BarnesHutTree<T, S> for Tree<(Orthant<N>, S), (T, S)>
where
    S: Scalar,
    T: InternalVector<Scalar = S>,
{
    fn acceleration_at(&self, node: Option<NodeID>, position: T, theta: S) -> T {
        let Some(id) = node else {
            return T::default();
        };
        let id = id as usize;

        let p2 = self.data[id];

        let dir = p2.0 - position;
        let mag_2 = dir.length_squared();
        let mag = mag_2.sqrt();

        match self.nodes[id] {
            Node::Internal((Orthant(o), width)) if theta < width / mag => {
                o.iter().fold(T::default(), |acceleration, &node| {
                    acceleration + self.acceleration_at(node, position, theta)
                })
            }
            _ => {
                if mag_2 == S::default() {
                    return dir;
                }

                dir * p2.1 / (mag_2 * mag)
            }
        }
    }
}

impl<T, S> TreeData for (T, S)
where
    S: Scalar,
    T: InternalVector<Scalar = S>,
{
    fn compute_data(data: &[Self]) -> Self {
        let total_mass = data.iter().map(|p| p.1).sum();
        let com = data
            .iter()
            .map(|p| p.0 * (p.1 / total_mass))
            .sum();
        (com, total_mass)
    }
}

impl<T, S> Positionable for (T, S)
where
    S: Scalar,
    T: InternalVector<Scalar = S>,
{
    type Vector = T;

    fn position(self) -> T {
        self.0
    }
}
