/// Bounding box related traits and types.
pub mod bbox;

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm implementation for compatible trees.
pub mod barnes_hut;

pub use barnes_hut::*;
pub use bbox::*;

/// Node of a [`Tree`].
pub enum Node<N> {
    /// Node with other child nodes.
    Internal(N),
    /// Node without children.
    External,
}

/// Generic tree that can partition space into smaller regions.
pub struct Tree<N, D> {
    /// Vector of [`Node`] objects that define the structure of the tree.
    pub nodes: Vec<Node<N>>,

    /// Vector of generic `D` objects that contain information about the data associated with each [`Node`].
    ///
    /// The `data` vector is parallel to the `nodes` vector, so the `i`-th element of the `data` vector corresponds to the `i`-th element of the `nodes` vector.
    pub data: Vec<D>,
}

/// Index of a [`Node`] in a [`Tree`].
pub type NodeID = u32;

impl<N, D> Tree<N, D> {
    /// Creates a new empty `Tree`.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates and inserts new [`Nodes`](Node) from the given data and bounding box.
    ///
    /// This will recursively insert new nodes into the tree until the given bounding box stops dividing.
    pub fn build_node<B>(&mut self, data: Vec<D>, bbox: B) -> Option<NodeID>
    where
        D: TreeData,
        B: BoundingBoxDivide<D, Output = N>,
    {
        if data.is_empty() {
            return None;
        }

        let id = self.nodes.len();
        self.nodes.push(Node::External);
        self.data.push(D::compute_data(&data));

        if let Some(inner) = bbox.divide(self, data) {
            self.nodes[id] = Node::Internal(inner);
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
