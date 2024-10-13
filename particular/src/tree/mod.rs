/// Bounding box related traits and types.
pub mod partition;

use std::ops::{Add, Sub};

pub use partition::*;

/// Index of a [`Node`] in a [`Tree`].
pub type NodeID = u32;

/// Generic tree data structure.
#[derive(Clone, Debug)]
pub struct Tree<Node, Data> {
    /// Vector of `Node` objects that define the structure of the tree.
    pub nodes: Vec<Node>,

    /// Vector of generic `Data` objects that contain information about the associated `Node`.
    ///
    /// The `data` vector is parallel to the `nodes` vector, so the `i`-th element of the `data`
    /// vector corresponds to the `i`-th element of the `nodes` vector.
    pub data: Vec<Data>,
}

impl<Node, Data> Tree<Node, Data> {
    /// Creates a new empty [`Tree`].
    #[inline]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates a new empty [`Tree`] with at least the specified capacity in the `nodes` and
    /// `data` vectors.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }
}

impl<Node, Data> Default for Tree<Node, Data> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Node that can either be internal and containing data or external and containing no data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node<N> {
    /// Node with child nodes.
    Internal(N),
    /// Node without children.
    External,
}

/// N-dimensional generalisation of quadtrees/octrees.
pub type Orthtree<const X: usize, const D: usize, S, Data> =
    Tree<Node<SizedOrthant<X, D, Option<NodeID>, S>>, Data>;

impl<const X: usize, const D: usize, S, Data> Orthtree<X, D, S, Data>
where
    Const<D>: SubDivide<Division = Const<X>>,
{
    /// Recursively inserts new [`Nodes`](Node) in the current [`Orthtree`] from the given input,
    /// position function and compute function until the computed square bounding box stops
    /// subdividing.
    #[inline]
    pub fn build_node<I, F, G>(&mut self, input: &[I], coordinates: F, compute: G) -> Option<NodeID>
    where
        I: Clone,
        F: Fn(&I) -> [S; D] + Copy,
        G: Fn(&[I]) -> Data + Copy,
        S: Add<Output = S> + Sub<Output = S> + MidPoint + MinMax + PartialOrd + Default + Clone,
        BoundingBox<[S; D]>: Default,
    {
        self.build_node_with(
            input,
            coordinates,
            compute,
            BoundingBox::square_with(input.iter().map(coordinates)),
        )
    }

    /// Recursively inserts new [`Nodes`](Node) in the current [`Orthtree`]  from the given input,
    /// position function and compute function until the given bounding box stops subdividing.
    pub fn build_node_with<I, F, G>(
        &mut self,
        input: &[I],
        coordinates: F,
        compute: G,
        bbox: BoundingBox<[S; D]>,
    ) -> Option<NodeID>
    where
        I: Clone,
        F: Fn(&I) -> [S; D] + Copy,
        G: Fn(&[I]) -> Data + Copy,
        S: PartialOrd + MidPoint + Default + Clone,
    {
        if input.is_empty() {
            return None;
        }

        let id = self.nodes.len();
        self.nodes.push(Node::External);
        self.data.push(compute(input));

        if input
            .windows(2)
            .any(|d| coordinates(&d[0]) != coordinates(&d[1]))
        {
            let center = bbox.center();
            let mut result = bbox.subdivide().map(|bbox| (Vec::new(), bbox));

            for d in input {
                let position = coordinates(d);
                let index = (0..D).fold(0, |index, i| {
                    index + (usize::from(position[i] < center[i]) << i)
                });

                result[index].0.push(d.clone());
            }

            self.nodes[id] = Node::Internal(SizedOrthant {
                orthant: result
                    .map(|(data, bbox)| self.build_node_with(&data, coordinates, compute, bbox)),
                bbox,
            });
        }

        Some(id as _)
    }
}
