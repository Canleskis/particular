/// Bounding box related traits and types.
pub mod bbox;
pub use bbox::*;

use crate::algorithms::math::Float;

/// Index of a [`Node`] in a [`Tree`].
pub type NodeID = u32;

/// Generic tree that can partition space into smaller regions.
#[derive(Debug, Clone)]
pub struct Tree<Node, Data> {
    /// Vector of `Node` objects that define the structure of the tree.
    pub nodes: Vec<Node>,

    /// Vector of generic `Data` objects that contain information about the associated `Node`.
    ///
    /// The `data` vector is parallel to the `nodes` vector, so the `i`-th element of the `data` vector corresponds to the `i`-th element of the `nodes` vector.
    pub data: Vec<Data>,
}

impl<Node, Data> Tree<Node, Data> {
    /// Creates a new empty [`Tree`].
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates a new empty [`Tree`] with at least the specified capacity in the `nodes` and `data` vectors.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }
}

impl<Node, Data> Default for Tree<Node, Data> {
    fn default() -> Self {
        Self::new()
    }
}

/// Division in N regions of the Euclidean space.
pub type Orthant<const X: usize> = [Option<NodeID>; X];

/// An [`Orthant`] and its size representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizedOrthant<const X: usize, S> {
    /// Stored orthant.
    pub orthant: Orthant<X>,
    /// Size of the stored orthant.
    pub size: S,
}

/// Node of a [`Tree`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node<N> {
    /// Node with child nodes.
    Internal(N),
    /// Node without children.
    External,
}

/// N-dimensional generalisation of quadtrees/octrees.
pub type Orthtree<const X: usize, const D: usize, S, Data> =
    Tree<Node<SizedOrthant<X, BoundingBox<[S; D]>>>, Data>;

impl<const X: usize, const D: usize, S, Data> Orthtree<X, D, S, Data> {
    /// Recursively inserts new [`Nodes`](Node) in the current [`Orthtree`] from the given input and functions until the created square bounding box stops subdividing.
    pub fn build_node<I, P, C>(&mut self, input: &[I], position: P, compute: C) -> Option<NodeID>
    where
        I: Copy,
        P: Fn(I) -> [S; D] + Copy,
        C: Fn(&[I]) -> Data + Copy,
        S: Copy + Float + PartialOrd,
        BoundingBox<[S; D]>: SubDivide<Division = [BoundingBox<[S; D]>; X]>,
    {
        self.build_node_with(
            BoundingBox::square_with(input.iter().copied().map(position)),
            input,
            position,
            compute,
        )
    }

    /// Recursively inserts new [`Nodes`](Node) in the current [`Orthtree`] from the given input and functions until the given bounding box stops subdividing.
    pub fn build_node_with<I, P, C>(
        &mut self,
        bbox: BoundingBox<[S; D]>,
        input: &[I],
        pos: P,
        compute: C,
    ) -> Option<NodeID>
    where
        I: Copy,
        P: Fn(I) -> [S; D] + Copy,
        C: Fn(&[I]) -> Data + Copy,
        S: Copy + Float + PartialOrd,
        BoundingBox<[S; D]>: SubDivide<Division = [BoundingBox<[S; D]>; X]>,
    {
        if input.is_empty() {
            return None;
        }

        let id = self.nodes.len();
        self.nodes.push(Node::External);
        self.data.push(compute(input));

        if input.windows(2).any(|d| pos(d[0]) != pos(d[1])) {
            let center = bbox.center();
            let mut result = bbox.subdivide().map(|bbox| (Vec::new(), bbox));

            for &d in input {
                let position = pos(d);
                let index = (0..D).fold(0, |index, i| {
                    index + (usize::from(position[i] < center[i]) << i)
                });

                result[index].0.push(d);
            }

            self.nodes[id] = Node::Internal(SizedOrthant {
                orthant: result.map(|(data, bbox)| self.build_node_with(bbox, &data, pos, compute)),
                size: bbox,
            });
        }

        Some(id as NodeID)
    }
}
