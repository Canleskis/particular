use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

/// Arbitrary vectors that can be converted from and into an array.
pub trait Vector<A>: Into<A> + From<A> {
    /// Internal representation of a vector.
    type Internal;

    /// Convert the arbitrary vector into its internal representation.
    fn into_internal(self) -> Self::Internal;

    /// Convert the internal representation into the arbitrary vector.
    fn from_internal(vector: Self::Internal) -> Self;
}

/// Scalar types that compose [`InternalVector`] objects.
pub trait Scalar:
    Sum
    + Sync
    + Send
    + Copy
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Infinity (∞).
    const INFINITY: Self;

    /// Square root operation.
    fn sqrt(self) -> Self;

    /// Minimum between two scalars.
    fn min(self, rhs: Self) -> Self;

    /// Maximum between two scalars.
    fn max(self, rhs: Self) -> Self;

    /// Midpoint between two scalars.
    fn midpoint(self, rhs: Self) -> Self;
}

/// Internal vectors used for expensive computations.
pub trait InternalVector:
    Sum
    + Sync
    + Send
    + Copy
    + Default
    + PartialEq
    + AddAssign
    + SubAssign
    + From<Self::Array>
    + Into<Self::Array>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::Scalar, Output = Self>
    + Div<Self::Scalar, Output = Self>
{
    /// The scalar type of the vector.
    type Scalar: Scalar;

    /// Array type this vector can be converted from and to.
    type Array;

    /// Norm squared, defined by the dot product on itself.
    fn length_squared(self) -> Self::Scalar;
}

macro_rules! vector_impl {
    ($s: ty, $(($t: ty, $dim: literal)),*) => {
        impl Scalar for $s {
            const INFINITY: Self = <$s>::INFINITY;

            #[inline]
            fn sqrt(self) -> $s {
                self.sqrt()
            }

            #[inline]
            fn min(self, rhs: Self) -> $s {
                self.min(rhs)
            }

            #[inline]
            fn max(self, rhs: Self) -> $s {
                self.max(rhs)
            }

            #[inline]
            fn midpoint(self, rhs: Self) -> $s {
                (self + rhs) / 2.0
            }
        }
    $(
        impl InternalVector for $t {
            type Scalar = $s;

            type Array = [$s; $dim];

            #[inline]
            fn length_squared(self) -> $s {
                self.length_squared()
            }
        }

        impl<V> Vector<[$s; $dim]> for V
        where
            V: Into<[$s; $dim]> + From<[$s; $dim]>,
        {
            type Internal = $t;

            #[inline]
            fn into_internal(self) -> Self::Internal {
                Self::Internal::from(self.into())
            }

            #[inline]
            fn from_internal(vector: Self::Internal) -> V {
                Self::from(vector.into())
            }
        }
    )*
    }
}

vector_impl!(f32, (glam::Vec2, 2), (glam::Vec3A, 3), (glam::Vec4, 4));
vector_impl!(f64, (glam::DVec2, 2), (glam::DVec3, 3), (glam::DVec4, 4));
