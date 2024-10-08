use std::ops::{Add, Neg, Sub};

/// Trait for the element `infinity`.
pub trait Infinity {
    /// `infinity` (∞) value of the type.
    fn infinity() -> Self;
}

/// Trait to compute the minimum and maximum of a number.
pub trait MinMax {
    /// Returns the minimum between two numbers.
    fn min(self, rhs: Self) -> Self;

    /// Returns the maximum between two numbers.
    fn max(self, rhs: Self) -> Self;
}

/// Trait to compute the midpoint of two numbers.
pub trait MidPoint {
    /// Returns the middle point of `self` and `rhs`.
    fn midpoint(self, rhs: Self) -> Self;

    /// Returns half of a number, defined by the midpoint between this number
    /// and zero.
    #[inline]
    fn half(self) -> Self
    where
        Self: Default,
    {
        self.midpoint(Self::default())
    }
}

macro_rules! impl_floats {
    ($s: ty) => {
        impl Infinity for $s {
            #[inline]
            fn infinity() -> Self {
                Self::INFINITY
            }
        }

        impl MinMax for $s {
            #[inline]
            fn min(self, rhs: Self) -> Self {
                self.min(rhs)
            }

            #[inline]
            fn max(self, rhs: Self) -> Self {
                self.max(rhs)
            }
        }

        impl MidPoint for $s {
            #[inline]
            fn midpoint(self, rhs: Self) -> Self {
                (self + rhs) / 2.0
            }
        }
    };
}

impl_floats!(f32);
impl_floats!(f64);

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
    S: Neg<Output = S> + Infinity + Copy,
{
    #[inline]
    fn default() -> Self {
        Self::new([S::infinity(); D], [-S::infinity(); D])
    }
}

#[allow(clippy::needless_range_loop)]
impl<const D: usize, S> BoundingBox<[S; D]> {
    /// Creates a new [`BoundingBox`] with all values set to zero.
    pub fn zero() -> Self
    where
        S: Default + Clone,
    {
        Self::new(
            std::array::from_fn(|_| S::default()),
            std::array::from_fn(|_| S::default()),
        )
    }

    /// Extends the [`BoundingBox`] so that it contains the given position.
    #[inline]
    pub fn extend(&mut self, position: &[S; D])
    where
        S: MinMax + Clone,
    {
        for i in 0..D {
            self.min[i] = self.min[i].clone().min(position[i].clone());
            self.max[i] = self.max[i].clone().max(position[i].clone());
        }
    }

    /// Creates a new [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn with<I>(positions: I) -> Self
    where
        Self: Default,
        S: MinMax + Clone,
        I: IntoIterator<Item = [S; D]>,
    {
        let mut result = Self::default();
        for position in positions {
            result.extend(&position);
        }
        result
    }

    /// Creates a new square [`BoundingBox`] that contains the given positions.
    #[inline]
    pub fn square_with<I>(positions: I) -> Self
    where
        Self: Default,
        S: Add<Output = S> + Sub<Output = S> + MidPoint + MinMax + Default + Clone,
        I: IntoIterator<Item = [S; D]>,
    {
        let mut result = Self::with(positions);

        let center = result.center();
        let half_length = result.size().into_iter().fold(S::default(), S::max).half();

        for i in 0..D {
            result.min[i] = center[i].clone() - half_length.clone();
            result.max[i] = center[i].clone() + half_length.clone();
        }

        result
    }

    /// Returns the center of the [`BoundingBox`].
    #[inline]
    pub fn center(&self) -> [S; D]
    where
        S: MidPoint + Default + Clone,
    {
        let mut r = std::array::from_fn(|_| S::default());
        for i in 0..D {
            r[i] = self.min[i].clone().midpoint(self.max[i].clone());
        }
        r
    }

    /// Returns the size of the [`BoundingBox`].
    #[inline]
    pub fn size(&self) -> [S; D]
    where
        S: Sub<Output = S> + Default + Clone,
    {
        let mut r = std::array::from_fn(|_| S::default());
        for i in 0..D {
            r[i] = self.max[i].clone() - self.min[i].clone();
        }
        r
    }

    /// Returns the width of the [`BoundingBox`] (x element of the size).
    #[inline]
    pub fn width(&self) -> S
    where
        S: Sub<Output = S> + Default + Clone,
    {
        self.size()[0].clone()
    }

    /// Subdivides this [`BoundingBox`] into `X` bounding boxes. This only works
    /// if `X = 2^D`.
    #[inline]
    pub fn subdivide<const X: usize>(&self) -> [Self; X]
    where
        S: MidPoint + Default + Clone,
    {
        let bbox_min = self.min.clone();
        let bbox_max = self.max.clone();
        let bbox_center = self.center();

        let mut result = std::array::from_fn(|_| Self::zero());
        for i in 0..X {
            let mut corner_min = std::array::from_fn(|_| S::default());
            let mut corner_max = std::array::from_fn(|_| S::default());

            for j in 0..D {
                if i & (1 << j) == 0 {
                    corner_min[j] = bbox_center[j].clone();
                    corner_max[j] = bbox_max[j].clone();
                } else {
                    corner_min[j] = bbox_min[j].clone();
                    corner_max[j] = bbox_center[j].clone();
                }
            }

            result[i] = Self::new(corner_min, corner_max);
        }

        result
    }
}

/// Marker trait for the division of a dimension.
pub trait SubDivide {
    /// An array type with the amount of divisions as its size.
    type Division;
}

/// Marker struct for a constant.
#[derive(Clone, Copy, Debug)]
pub struct Const<const D: usize>;

macro_rules! impl_subdivide {
    ($($dim: literal),*) => {$(
        impl SubDivide for Const<$dim> {
            type Division = Const<{ 2usize.pow($dim) }>;
        }
    )*};
}

impl_subdivide!(1, 2, 3, 4, 5, 6, 7, 8, 9);

/// Division in `X` regions of the Euclidean space.
pub type Orthant<const X: usize, N> = [N; X];

/// An [`Orthant`] and its size as a [`BoundingBox`].
#[derive(Clone, Copy, Debug)]
pub struct SizedOrthant<const X: usize, const D: usize, N, S> {
    /// Stored orthant.
    pub orthant: Orthant<X, N>,
    /// Size of the stored orthant.
    pub bbox: BoundingBox<[S; D]>,
}
