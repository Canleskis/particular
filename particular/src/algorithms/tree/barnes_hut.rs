use crate::algorithms::{
    internal,
    tree::{BoundingBox, Node, NodeID, Positionable, SizedOrthant, Tree, TreeData},
    PointMass,
};

/// Trait to compute the acceleration of particles using the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm.
pub trait BarnesHutAcceleration<T, S> {
    /// Computes the acceleration at the given position from a node of the tree.
    fn acceleration_at(&self, node: Option<NodeID>, position: T, theta: S) -> T;
}

/// Trees that can be used with the [`BarnesHutAcceleration`] trait.
pub type BarnesHutTree<const N: usize, const D: usize, T, S> =
    Tree<SizedOrthant<N, BoundingBox<[S; D]>>, PointMass<T, S>>;

impl<T, S, const D: usize, const N: usize> BarnesHutAcceleration<T, S> for BarnesHutTree<N, D, T, S>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
{
    fn acceleration_at(&self, node: Option<NodeID>, position: T, theta: S) -> T {
        let Some(id) = node else {
            return T::ZERO;
        };
        let id = id as usize;

        let p2 = self.data[id];

        let dir = p2.position - position;
        let mag_2 = dir.length_squared();
        let mag = mag_2.sqrt();

        match self.nodes[id] {
            Node::Internal(SizedOrthant(orthant, bbox)) if theta < bbox.width() / mag => {
                orthant.iter().fold(T::ZERO, |acceleration, &node| {
                    acceleration + self.acceleration_at(node, position, theta)
                })
            }
            _ => {
                if mag_2 == S::ZERO {
                    return dir;
                }

                dir * p2.mass / (mag_2 * mag)
            }
        }
    }
}

impl<T, S> TreeData for PointMass<T, S>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
{
    #[inline]
    fn compute_data(data: &[Self]) -> Self {
        let total_mass = data.iter().map(|p| p.mass).sum();
        let com = data
            .iter()
            .map(|p| p.position * (p.mass / total_mass))
            .sum();

        PointMass::new(com, total_mass)
    }
}

impl<T, S> Positionable for PointMass<T, S>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
{
    type Vector = T;

    #[inline]
    fn position(&self) -> T {
        self.position
    }
}
