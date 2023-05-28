use crate::algorithms::{
    tree::{Node, NodeID, Orthant, Positionable, Tree, TreeData},
    InternalVector, PointMass, Scalar,
};

/// Trait to compute the acceleration of particles using the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm.
pub trait BarnesHutTree<T, S> {
    /// Computes the acceleration at the given position from a node of the tree.
    fn acceleration_at(&self, node: Option<NodeID>, position: T, theta: S) -> T;
}

impl<const N: usize, T, S> BarnesHutTree<T, S> for Tree<(Orthant<N>, S), PointMass<T, S>>
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

        let dir = p2.position - position;
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

                dir * p2.mass / (mag_2 * mag)
            }
        }
    }
}

impl<T, S> TreeData for PointMass<T, S>
where
    S: Scalar,
    T: InternalVector<Scalar = S>,
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
    S: Scalar,
    T: InternalVector<Scalar = S>,
{
    type Vector = T;

    #[inline]
    fn position(&self) -> T {
        self.position
    }
}
