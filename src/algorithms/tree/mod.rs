/// Bounding box related traits and types.
pub mod bbox;

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm implementation for compatible trees.
pub mod barnes_hut;
pub use barnes_hut::*;
pub use bbox::*;

use crate::algorithms::internal;

/// Index of a [`Node`] in a [`Tree`].
pub type NodeID = u32;

/// Node of a [`Tree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node<N> {
    /// Node with child nodes.
    Internal(N),
    /// Node without children.
    External,
}

/// Generic tree that can partition space into smaller regions.
#[derive(Debug, Clone)]
pub struct Tree<N, D> {
    /// Vector of [`Node`] objects that define the structure of the tree.
    pub nodes: Vec<Node<N>>,

    /// Vector of generic `D` objects that contain information about the data associated with each [`Node`].
    ///
    /// The `data` vector is parallel to the `nodes` vector, so the `i`-th element of the `data` vector corresponds to the `i`-th element of the `nodes` vector.
    pub data: Vec<D>,
}

impl<N, D> Tree<N, D> {
    /// Creates a new empty `Tree`.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            data: Vec::new(),
        }
    }
}

impl<const DIM: usize, const N: usize, S, D> Tree<SizedOrthant<N, BoundingBox<[S; DIM]>>, D>
where
    S: internal::Scalar,
    D: Positionable + TreeData + Copy,
    D::Vector: PartialEq + Into<[S; DIM]>,
    BoundingBox<[S; DIM]>: SubDivide<Divison = [BoundingBox<[S; DIM]>; N]>,
{
    /// Creates and inserts new [`Nodes`](Node) from the given data and bounding box.
    ///
    /// This will recursively insert new nodes into the tree until the given bounding box stops subdividing.
    pub fn build_node(&mut self, data: &[D], bbox: BoundingBox<[S; DIM]>) -> Option<NodeID> {
        if data.is_empty() {
            return None;
        }

        let id = self.nodes.len();
        self.nodes.push(Node::External);
        self.data.push(D::compute_data(data));

        if data.windows(2).any(|d| d[0].position() != d[1].position()) {
            let bbox_center = bbox.center();
            let mut result = bbox.subdivide().map(|bbox| (Vec::new(), bbox));

            for &d in data {
                let position = d.position().into();
                let index = (0..DIM).fold(0, |index, i| {
                    index + (((position[i] < bbox_center[i]) as usize) << i)
                });

                result[index].0.push(d);
            }

            self.nodes[id] = Node::Internal(SizedOrthant(
                result.map(|(data, bbox)| self.build_node(&data, bbox)),
                bbox,
            ));
        }

        Some(id as NodeID)
    }
}

impl<N, D> Default for Tree<N, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for data stored in a [`Tree`].
pub trait TreeData: Sized {
    /// Computes the data to store from a slice.
    fn compute_data(data: &[Self]) -> Self;
}
