use std::ops::{Index, IndexMut};

use crate::algorithms::{internal, tree::NodeID};

/// An axis-aligned bounding box using arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundingBox<A> {
    /// Minimum corner of the box.
    pub min: A,
    /// Maximum corner of the box.
    pub max: A,
}

#[allow(clippy::needless_range_loop)]
impl<const DIM: usize, S> BoundingBox<[S; DIM]>
where
    S: internal::Scalar,
{
    /// Creates a new [`BoundingBox`] with the given min and max values.
    #[inline]
    pub fn new(min: [S; DIM], max: [S; DIM]) -> Self {
        Self { min, max }
    }

    /// Extends the [`BoundingBox`] so that it contains the given position.
    #[inline]
    pub fn extend(&mut self, position: [S; DIM]) {
        for i in 0..DIM {
            self.min[i] = self.min[i].min(position[i]);
            self.max[i] = self.max[i].max(position[i]);
        }
    }

    /// Creates a new [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn with(positions: impl Iterator<Item = [S; DIM]>) -> Self {
        let mut result = Self::default();
        for position in positions {
            result.extend(position)
        }
        result
    }

    /// Creates a new square [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn square_with(positions: impl Iterator<Item = [S; DIM]>) -> Self {
        let mut result = Self::with(positions);

        let center = result.center();
        let half_length = result
            .size()
            .into_iter()
            .fold(S::ZERO, S::max)
            .midpoint(S::ZERO);

        for i in 0..DIM {
            result.min[i] = center[i] - half_length;
            result.max[i] = center[i] + half_length;
        }

        result
    }

    /// Returns the center of the [`BoundingBox`].
    #[inline]
    pub fn center(&self) -> [S; DIM] {
        let mut r = [S::ZERO; DIM];
        for i in 0..DIM {
            r[i] = self.min[i].midpoint(self.max[i])
        }
        r
    }

    /// Returns the size of the [`BoundingBox`].
    #[inline]
    pub fn size(&self) -> [S; DIM] {
        let mut r = [S::ZERO; DIM];
        for i in 0..DIM {
            r[i] = self.max[i] - self.min[i]
        }
        r
    }

    /// Returns the width of the [`BoundingBox`] (x element of the size).
    #[inline]
    pub fn width(&self) -> S {
        self.size()[0]
    }
}

impl<const DIM: usize, S> Default for BoundingBox<[S; DIM]>
where
    S: internal::Scalar,
{
    #[inline]
    fn default() -> Self {
        Self {
            min: [S::INFINITY; DIM],
            max: [-S::INFINITY; DIM],
        }
    }
}

/// Marker type for the division of a [`BoundingBox`] into multiple bounding boxes.
pub trait SubDivide {
    /// The number of divisions.
    const N: usize;

    /// The bounding boxes [`BoundingBox`] divides into.
    type Divison: Default + Index<usize, Output = Self> + IndexMut<usize, Output = Self>;
}

impl<const DIM: usize, S> BoundingBox<[S; DIM]>
where
    Self: SubDivide,
    S: internal::Scalar,
{
    /// Subdivides this [`BoundingBox`] into mutliple bounding boxes defined by [`SubDivide`] implementation.
    #[inline]
    pub fn subdivide(&self) -> <Self as SubDivide>::Divison {
        let bbox_min = self.min;
        let bbox_max = self.max;
        let bbox_center = self.center();

        let mut result = <Self as SubDivide>::Divison::default();
        for i in 0..Self::N {
            let mut corner_min = [S::ZERO; DIM];
            let mut corner_max = [S::ZERO; DIM];

            for j in 0..DIM {
                if (i & (1 << j)) == 0 {
                    corner_min[j] = bbox_center[j];
                    corner_max[j] = bbox_max[j];
                } else {
                    corner_min[j] = bbox_min[j];
                    corner_max[j] = bbox_center[j];
                }
            }

            result[i] = Self::new(corner_min, corner_max);
        }

        result
    }
}

/// Division in N regions of the Euclidean space.
pub type Orthant<const N: usize> = [Option<NodeID>; N];

/// An [`Orthant`] and a type representing its size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizedOrthant<const N: usize, S>(pub Orthant<N>, pub S);

/// A trait for types that can be located in space.
pub trait Positionable {
    /// The type of vector used to represent the position.
    type Vector;

    /// Returns the position of the object.
    fn position(&self) -> Self::Vector;
}

macro_rules! impl_subdivide {
    ($(($dim: literal|$n: literal)),*,) => {$(
        impl<S> SubDivide for BoundingBox<[S; $dim]>
        where
            Self: Default,
        {
            const N: usize = $n;

            type Divison = [Self; $n];
        }
    )*};
}

impl_subdivide!((1 | 2), (2 | 4), (3 | 8), (4 | 16), (5 | 32),);
