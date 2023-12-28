use crate::compute_method::math::Float;

/// An axis-aligned bounding box using arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundingBox<A> {
    /// Minimum corner of the box.
    pub min: A,
    /// Maximum corner of the box.
    pub max: A,
}

impl<A> BoundingBox<A> {
    /// Creates a new [`BoundingBox`] with the given min and max values.
    #[inline]
    pub const fn new(min: A, max: A) -> Self {
        Self { min, max }
    }
}

impl<const D: usize, S> Default for BoundingBox<[S; D]>
where
    S: Copy + Float,
{
    #[inline]
    fn default() -> Self {
        Self::new([S::infinity(); D], [-S::infinity(); D])
    }
}

#[allow(clippy::needless_range_loop)]
impl<const D: usize, S> BoundingBox<[S; D]>
where
    S: Copy + Float,
{
    /// Extends the [`BoundingBox`] so that it contains the given position.
    #[inline]
    pub fn extend(&mut self, position: [S; D]) {
        for i in 0..D {
            self.min[i] = self.min[i].min(position[i]);
            self.max[i] = self.max[i].max(position[i]);
        }
    }

    /// Creates a new [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn with<I>(positions: I) -> Self
    where
        I: Iterator<Item = [S; D]>,
    {
        let mut result = Self::default();
        for position in positions {
            result.extend(position);
        }
        result
    }

    /// Creates a new square [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn square_with<I>(positions: I) -> Self
    where
        I: Iterator<Item = [S; D]>,
    {
        let mut result = Self::with(positions);

        let center = result.center();
        let half_length = result
            .size()
            .into_iter()
            .fold(S::ZERO, S::max)
            .mean(S::ZERO);

        for i in 0..D {
            result.min[i] = center[i] - half_length;
            result.max[i] = center[i] + half_length;
        }

        result
    }

    /// Returns the center of the [`BoundingBox`].
    #[inline]
    pub fn center(&self) -> [S; D] {
        let mut r = [S::ZERO; D];
        for i in 0..D {
            r[i] = self.min[i].mean(self.max[i]);
        }
        r
    }

    /// Returns the size of the [`BoundingBox`].
    #[inline]
    pub fn size(&self) -> [S; D] {
        let mut r = [S::ZERO; D];
        for i in 0..D {
            r[i] = self.max[i] - self.min[i];
        }
        r
    }

    /// Returns the width of the [`BoundingBox`] (x element of the size).
    #[inline]
    pub fn width(&self) -> S {
        self.size()[0]
    }
}

#[allow(clippy::needless_range_loop)]
impl<const X: usize, const D: usize, S> BoundingBox<[S; D]>
where
    Self: SubDivide<Division = [Self; X]>,
    S: Copy + Float,
{
    /// Subdivides this [`BoundingBox`] into mutliple bounding boxes defined by [`SubDivide`] implementation.
    #[inline]
    pub fn subdivide(&self) -> [Self; X] {
        let bbox_min = self.min;
        let bbox_max = self.max;
        let bbox_center = self.center();

        let mut result = [Self::default(); X];
        for i in 0..X {
            let mut corner_min = [S::ZERO; D];
            let mut corner_max = [S::ZERO; D];

            for j in 0..D {
                if i & (1 << j) == 0 {
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

/// Division in `X` regions of the Euclidean space.
pub type Orthant<const X: usize, N> = [Option<N>; X];

/// An [`Orthant`] and its size as a [`BoundingBox`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizedOrthant<const X: usize, const D: usize, N, S> {
    /// Stored orthant.
    pub orthant: Orthant<X, N>,
    /// Size of the stored orthant.
    pub bbox: BoundingBox<[S; D]>,
}

/// Marker trait for the division of a [`BoundingBox`] into multiple bounding boxes.
pub trait SubDivide {
    /// The type the implementer divides into.
    type Division;
}

macro_rules! impl_subdivide {
    ($($dim: literal),*) => {$(
        impl<S> SubDivide for BoundingBox<[S; $dim]> {
            type Division = [Self; 2usize.pow($dim)];
        }
    )*};
}

impl_subdivide!(1, 2, 3, 4, 5, 6, 7, 8, 9);
