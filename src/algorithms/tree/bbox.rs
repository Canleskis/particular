use crate::algorithms::{
    tree::{NodeID, Tree, TreeData},
    Scalar,
};

/// An axis-aligned bounding box using arrays.
#[derive(Clone, Copy)]
pub struct BoundingBox<A> {
    /// Minimum corner of the box.
    pub min: A,
    /// Maximum corner of the box.
    pub max: A,
}

#[allow(clippy::needless_range_loop)]
impl<const DIM: usize, S> BoundingBox<[S; DIM]>
where
    S: Scalar,
{
    /// Creates a new [`BoundingBox`] with the given min and max values.
    #[inline]
    pub fn new(min: [S; DIM], max: [S; DIM]) -> Self {
        Self { min, max }
    }

    /// Creates a new [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn containing(positions: impl Iterator<Item = [S; DIM]>) -> Self {
        let mut result = Self::default();
        for position in positions {
            result.extend(position)
        }
        result
    }

    /// Extends the [`BoundingBox`] so that it contains the given position.
    #[inline]
    pub fn extend(&mut self, position: [S; DIM]) {
        for i in 0..DIM {
            self.min[i] = self.min[i].min(position[i]);
            self.max[i] = self.max[i].max(position[i]);
        }
    }

    /// Center of the [`BoundingBox`].
    #[inline]
    pub fn center(&self) -> [S; DIM] {
        let mut r = [S::default(); DIM];
        for i in 0..DIM {
            r[i] = self.min[i].midpoint(self.max[i])
        }
        r
    }

    /// Size of the [`BoundingBox`].
    #[inline]
    pub fn size(&self) -> [S; DIM] {
        let mut r = [S::default(); DIM];
        for i in 0..DIM {
            r[i] = self.max[i] - self.min[i]
        }
        r
    }
}

impl<const DIM: usize, S> Default for BoundingBox<[S; DIM]>
where
    S: Scalar,
{
    #[inline]
    fn default() -> Self {
        Self {
            min: [S::INFINITY; DIM],
            max: [-S::INFINITY; DIM],
        }
    }
}

/// Division in N regions of the Euclidean space.
pub struct Orthant<const N: usize>(pub [Option<NodeID>; N]);

/// Trait to divide a bounding box into multiple regions.
pub trait BoundingBoxDivide<D>: Sized {
    /// The type of the divided regions.
    type Output;

    /// Divides the bounding box and inserts resulting nodes into the tree.
    fn divide(&self, tree: &mut Tree<Self::Output, D>, data: Vec<D>) -> Option<Self::Output>;
}

/// A trait for types that can be located in space.
pub trait Positionable {
    /// The type of vector used to represent the position.
    type Vector;

    /// Returns the position of the object.
    fn position(&self) -> Self::Vector;
}

impl<D, S> BoundingBoxDivide<D> for BoundingBox<[S; 2]>
where
    S: Scalar,
    D: Positionable + TreeData,
    D::Vector: PartialEq + Into<[S; 2]>,
{
    type Output = (Orthant<4>, S);

    fn divide(&self, tree: &mut Tree<Self::Output, D>, data: Vec<D>) -> Option<Self::Output> {
        data.windows(2)
            .any(|data| data[0].position() != data[1].position())
            .then(|| {
                let (mut ne_data, mut nw_data, mut se_data, mut sw_data) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());

                let [bbox_min_x, bbox_min_y] = self.min;
                let [bbox_max_x, bbox_max_y] = self.max;
                let [center_x, center_y] = self.center();

                for p in data {
                    let [pos_x, pos_y] = p.position().into();
                    let right = pos_x > center_x;
                    let top = pos_y > center_y;

                    if right && top {
                        ne_data.push(p);
                    } else if !right && top {
                        nw_data.push(p);
                    } else if right {
                        se_data.push(p);
                    } else {
                        sw_data.push(p);
                    }
                }

                let [ne_bbox, nw_bbox, se_bbox, sw_bbox] = [
                    BoundingBox::new([center_x, center_y], [bbox_max_x, bbox_max_y]),
                    BoundingBox::new([bbox_min_x, center_y], [center_x, bbox_max_y]),
                    BoundingBox::new([center_x, bbox_min_y], [bbox_max_x, center_y]),
                    BoundingBox::new([bbox_min_x, bbox_min_y], [center_x, center_y]),
                ];

                (
                    Orthant([
                        tree.build_node(ne_data, ne_bbox),
                        tree.build_node(nw_data, nw_bbox),
                        tree.build_node(se_data, se_bbox),
                        tree.build_node(sw_data, sw_bbox),
                    ]),
                    self.size()[0],
                )
            })
    }
}

impl<D, S> BoundingBoxDivide<D> for BoundingBox<[S; 3]>
where
    S: Scalar,
    D: Positionable + TreeData,
    D::Vector: PartialEq + Into<[S; 3]>,
{
    type Output = (Orthant<8>, S);

    fn divide(&self, tree: &mut Tree<Self::Output, D>, data: Vec<D>) -> Option<Self::Output> {
        data.windows(2).any(|data| data[0].position() != data[1].position()).then(|| {
            let (mut fne_data, mut fnw_data, mut fsw_data, mut fse_data) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut bne_data, mut bnw_data, mut bsw_data, mut bse_data) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());

            let [bbox_min_x, bbox_min_y, bbox_min_z] = self.min;
            let [bbox_max_x, bbox_max_y, bbox_max_z] = self.max;
            let [center_x, center_y, center_z] = self.center();

            for p in data {
                let [pos_x, pos_y, pos_z] = p.position().into();
                let right = pos_x > center_x;
                let top = pos_y > center_y;
                let front = pos_z > center_z;

                if right && top && front {
                    fne_data.push(p);
                } else if !right && top && front {
                    fnw_data.push(p);
                } else if right && !top && front {
                    fse_data.push(p);
                } else if !right && !top && front {
                    fsw_data.push(p);
                } else if right && top {
                    bne_data.push(p);
                } else if !right && top {
                    bnw_data.push(p);
                } else if right {
                    bse_data.push(p);
                } else {
                    bsw_data.push(p);
                }
            }

            let [fne_bbox, fnw_bbox, fse_bbox, fsw_bbox, bne_bbox, bnw_bbox, bse_bbox, bsw_bbox] = [
                BoundingBox::new(
                    [center_x, center_y, center_z],
                    [bbox_max_x, bbox_max_y, bbox_max_z],
                ),
                BoundingBox::new(
                    [bbox_min_x, center_y, center_z],
                    [center_x, bbox_max_y, bbox_max_z],
                ),
                BoundingBox::new(
                    [center_x, bbox_min_y, center_z],
                    [bbox_max_x, center_y, bbox_max_z],
                ),
                BoundingBox::new(
                    [bbox_min_x, bbox_min_y, center_z],
                    [center_x, center_y, bbox_max_z],
                ),
                BoundingBox::new(
                    [center_x, center_y, bbox_min_z],
                    [bbox_max_x, bbox_max_y, center_z],
                ),
                BoundingBox::new(
                    [bbox_min_x, center_y, bbox_min_z],
                    [center_x, bbox_max_y, center_z],
                ),
                BoundingBox::new(
                    [center_x, bbox_min_y, bbox_min_z],
                    [bbox_max_x, center_y, center_z],
                ),
                BoundingBox::new(
                    [bbox_min_x, bbox_min_y, bbox_min_z],
                    [center_x, center_y, center_z],
                ),
            ];

            (
                Orthant([
                    tree.build_node(fne_data, fne_bbox),
                    tree.build_node(fnw_data, fnw_bbox),
                    tree.build_node(fse_data, fse_bbox),
                    tree.build_node(fsw_data, fsw_bbox),
                    tree.build_node(bne_data, bne_bbox),
                    tree.build_node(bnw_data, bnw_bbox),
                    tree.build_node(bse_data, bse_bbox),
                    tree.build_node(bsw_data, bsw_bbox),
                ]),
                self.size()[0],
            )
        })
    }
}
