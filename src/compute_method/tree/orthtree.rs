use super::{bbox::*, Node, NodeID, Tree, TreeBuilder, TreeData};

pub(crate) struct Orthant<const N: usize, S>(pub [Option<NodeID>; N], pub S);

trait Orthtree<B, D>
where
    B: BoundingBoxOps,
{
    fn build_orthant(&mut self, data: Vec<D>, bbox: B) -> Option<B::Orthant>;
}

impl<D, B, N> TreeBuilder<B, D> for Tree<N, D>
where
    Self: Orthtree<B, D>,
    B: BoundingBoxOps<Orthant = N>,
    D: TreeData,
{
    fn build_node(&mut self, data: Vec<D>, bbox: B) -> Option<NodeID> {
        if data.is_empty() {
            return None;
        }

        let id = self.nodes.len();
        self.nodes.push(Node::External);
        self.data.push(D::compute_data(&data));

        if let Some(orthant) = self.build_orthant(data, bbox) {
            self.nodes[id] = Node::Internal(orthant);
        }

        Some(id as NodeID)
    }
}

// Implement orthtree for trees which data has a vector type (position) that can be located in a 2D bounding box.
impl<B, T, D, S> Orthtree<B, (T, D)> for Tree<Orthant<4, S>, (T, D)>
where
    B: BoundingBoxOps<Orthant = Orthant<4, S>, Array = [S; 2]>,
    T: Copy + PartialEq + Into<B::Array>,
    S: Copy + PartialOrd,
    (T, D): TreeData,
{
    fn build_orthant(&mut self, data: Vec<(T, D)>, bbox: B) -> Option<B::Orthant> {
        data.windows(2).any(|a| a[0].0 != a[1].0).then(|| {
            let [center_x, center_y] = bbox.center();
            let [bbox_min_x, bbox_min_y] = bbox.min();
            let [bbox_max_x, bbox_max_y] = bbox.max();

            let (mut ne, mut nw, mut se, mut sw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());

            for p in data {
                let [pos_x, pos_y] = p.0.into();
                let right = pos_x > center_x;
                let top = pos_y > center_y;

                if right && top {
                    ne.push(p);
                } else if !right && top {
                    nw.push(p);
                } else if right {
                    se.push(p);
                } else {
                    sw.push(p);
                }
            }

            let [nebb, nwbb, sebb, swbb] = [
                B::new([center_x, center_y], [bbox_max_x, bbox_max_y]),
                B::new([bbox_min_x, center_y], [center_x, bbox_max_y]),
                B::new([center_x, bbox_min_y], [bbox_max_x, center_y]),
                B::new([bbox_min_x, bbox_min_y], [center_x, center_y]),
            ];

            Orthant(
                [
                    self.build_node(ne, nebb),
                    self.build_node(nw, nwbb),
                    self.build_node(se, sebb),
                    self.build_node(sw, swbb),
                ],
                bbox.size()[0],
            )
        })
    }
}

// Implement orthtree for trees which data has a vector type (position) that can be located in a 3D bounding box.
impl<B, T, D, S> Orthtree<B, (T, D)> for Tree<Orthant<8, S>, (T, D)>
where
    B: BoundingBoxOps<Orthant = Orthant<8, S>, Array = [S; 3]>,
    T: Copy + PartialEq + Into<B::Array>,
    S: Copy + PartialOrd,
    (T, D): TreeData,
{
    fn build_orthant(&mut self, data: Vec<(T, D)>, bbox: B) -> Option<B::Orthant> {
        data.windows(2).any(|a| a[0].0 != a[1].0).then(|| {
            let [center_x, center_y, center_z] = bbox.center();
            let [bbox_min_x, bbox_min_y, bbox_min_z] = bbox.min();
            let [bbox_max_x, bbox_max_y, bbox_max_z] = bbox.max();

            let (mut fne, mut fnw, mut fsw, mut fse) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut bne, mut bnw, mut bsw, mut bse) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());

            for p in data {
                let [pos_x, pos_y, pos_z] = p.0.into();
                let right = pos_x > center_x;
                let top = pos_y > center_y;
                let front = pos_z > center_z;

                if right && top && front {
                    fne.push(p);
                } else if !right && top && front {
                    fnw.push(p);
                } else if right && !top && front {
                    fse.push(p);
                } else if !right && !top && front {
                    fsw.push(p);
                } else if right && top {
                    bne.push(p);
                } else if !right && top {
                    bnw.push(p);
                } else if right {
                    bse.push(p);
                } else {
                    bsw.push(p);
                }
            }

            let [fnebb, fnwbb, fsebb, fswbb, bnebb, bnwbb, bsebb, bswbb] = [
                B::new(
                    [center_x, center_y, center_z],
                    [bbox_max_x, bbox_max_y, bbox_max_z],
                ),
                B::new(
                    [bbox_min_x, center_y, center_z],
                    [center_x, bbox_max_y, bbox_max_z],
                ),
                B::new(
                    [center_x, bbox_min_y, center_z],
                    [bbox_max_x, center_y, bbox_max_z],
                ),
                B::new(
                    [bbox_min_x, bbox_min_y, center_z],
                    [center_x, center_y, bbox_max_z],
                ),
                B::new(
                    [center_x, center_y, bbox_min_z],
                    [bbox_max_x, bbox_max_y, center_z],
                ),
                B::new(
                    [bbox_min_x, center_y, bbox_min_z],
                    [center_x, bbox_max_y, center_z],
                ),
                B::new(
                    [center_x, bbox_min_y, bbox_min_z],
                    [bbox_max_x, center_y, center_z],
                ),
                B::new(
                    [bbox_min_x, bbox_min_y, bbox_min_z],
                    [center_x, center_y, center_z],
                ),
            ];

            Orthant(
                [
                    self.build_node(fne, fnebb),
                    self.build_node(fnw, fnwbb),
                    self.build_node(fse, fsebb),
                    self.build_node(fsw, fswbb),
                    self.build_node(bne, bnebb),
                    self.build_node(bnw, bnwbb),
                    self.build_node(bse, bsebb),
                    self.build_node(bsw, bswbb),
                ],
                bbox.size()[0],
            )
        })
    }
}
