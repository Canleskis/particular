pub(crate) mod acceleration;
pub(crate) mod bbox;
pub(crate) mod orthtree;

type NodeID = u32;

pub(crate) trait TreeData: Sized {
    type Output;

    fn compute_data(data: &[Self]) -> Self::Output;
}

enum Node<N> {
    Internal(N),
    External,
}

pub(crate) struct Tree<N, D>
where
    D: TreeData,
{
    nodes: Vec<Node<N>>,
    data: Vec<D::Output>,
}

pub(crate) trait TreeBuilder<B, D>
where
    B: bbox::BoundingBoxOps,
{
    fn build_node(&mut self, data: Vec<D>, bbox: B) -> Option<NodeID>;

    fn build(data: Vec<D>, bbox: B) -> Self
    where
        Self: Default,
    {
        let mut tree = Self::default();
        tree.build_node(data, bbox);
        tree
    }
}

impl<N, D> Default for Tree<N, D>
where
    D: TreeData,
{
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            data: Vec::new(),
        }
    }
}
