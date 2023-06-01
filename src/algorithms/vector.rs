use std::{
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// Arbitrary vectors that can be converted from and into the [`InternalVector::Array`] of a given [`InternalVector`].
pub trait IntoInternalVector<A> {
    /// Internal representation of a vector.
    type InternalVector;

    /// Converts the arbitrary vector into its internal representation.
    fn into_internal(self) -> Self::InternalVector;

    /// Converts the internal representation into the arbitrary vector.
    fn from_internal(vector: Self::InternalVector) -> Self;
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
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
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
    + MulAssign
    + DivAssign
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

macro_rules! internal_vector {
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

        impl<V> IntoInternalVector<[$s; $dim]> for V
        where
            V: Into<[$s; $dim]> + From<[$s; $dim]>,
        {
            type InternalVector = $t;

            #[inline]
            fn into_internal(self) -> Self::InternalVector {
                Self::InternalVector::from(self.into())
            }

            #[inline]
            fn from_internal(vector: Self::InternalVector) -> V {
                Self::from(vector.into())
            }
        }
    )*
    }
}

internal_vector!(f32, (glam::Vec2, 2), (glam::Vec3A, 3), (glam::Vec4, 4));
internal_vector!(f64, (glam::DVec2, 2), (glam::DVec3, 3), (glam::DVec4, 4));

/// Arbitrary vectors that can be converted into the [`SIMD::Element`] of a given [`SIMDVector`].
pub trait IntoSIMDElement<E> {
    /// SIMD representation of a vector.
    type SIMDVector;

    /// Converts the arbitrary vector into its SIMDElement.
    fn into_simd_element(self) -> E;

    /// Reduces the SIMD vector (by summing its lanes) into the arbitrary vector.
    fn from_after_reduce(vector: Self::SIMDVector) -> Self;
}

/// SIMD objects descriptor with constructors.
pub trait SIMD<const LANES: usize> {
    /// Element from which the SIMD value can be created.
    type Element: Copy + PartialEq + Default;

    /// Creates a SIMD value with all lanes set to the specified value.
    fn splat(scalar: Self::Element) -> Self;

    /// Constructs a SIMD value from the given values.
    fn from_lanes(vecs: [Self::Element; LANES]) -> Self;
}

/// Scalar types that compose [`SIMDVector`] objects.
pub trait SIMDScalar<const LANES: usize>:
    Sync
    + Send
    + Copy
    + Default
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialEq
    + SIMD<LANES>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Square root operation.
    fn sqrt(self) -> Self;

    /// Reciprocal (inverse) square root operation.
    fn recip_sqrt(self) -> Self;

    /// Reciprocal (inverse) operation.
    fn recip(self) -> Self;
}

/// SIMD vectors used for expensive computations.
pub trait SIMDVector<const LANES: usize>:
    Send
    + Sync
    + Copy
    + Default
    + ReduceAdd
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + SIMD<LANES>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Self::SIMDScalar, Output = Self>
    + Div<Self::SIMDScalar, Output = Self>
{
    /// The scalar type of the vector.
    type SIMDScalar;

    /// Norm squared, defined by the dot product on itself.
    fn length_squared(self) -> Self::SIMDScalar;

    /// Returns itself with NaNs replaced with zeroes.
    fn nan_to_zero(self) -> Self;
}

/// Trait to sum the lanes of a SIMD vector.
pub trait ReduceAdd {
    /// The resulting type after applying the operation.
    type Output;

    /// Sums the lanes of a SIMD vector.
    fn reduce_add(self) -> Self::Output;
}

macro_rules! simd_vector {
    ($l: literal, $s: ty => $se: ty, $(($t: ty => $te: ty, $dim: literal)),*) => {
        impl SIMD<$l> for $s {
            type Element = $se;

            #[inline]
            fn splat(scalar: Self::Element) -> Self {
                Self::splat(scalar)
            }

            #[inline]
            fn from_lanes(vecs: [Self::Element; $l]) -> Self {
                Self::from(vecs)
            }
        }
    $(
        impl SIMD<$l> for $t {
            type Element = $te;

            #[inline]
            fn splat(scalar: Self::Element) -> Self {
                Self::splat(scalar)
            }

            #[inline]
            fn from_lanes(vecs: [Self::Element; $l]) -> Self {
                Self::from(vecs)
            }
        }

        impl SIMDVector<$l> for $t {
            type SIMDScalar = $s;

            #[inline]
            fn length_squared(self) -> Self::SIMDScalar {
                self.mag_sq()
            }

            #[inline]
            fn nan_to_zero(self) -> Self {
                self.map(|vector| vector & wide::CmpEq::cmp_eq(vector, vector))
            }
        }

        impl<V> IntoSIMDElement<<$t as SIMD<$l>>::Element> for V
        where
            V: Into<<$t as ReduceAdd>::Output> + From<<$t as ReduceAdd>::Output>,
        {
            type SIMDVector = $t;

            #[inline]
            fn into_simd_element(self) -> <$t as SIMD<$l>>::Element {
                <$t as SIMD<$l>>::Element::from(self.into())
            }

            #[inline]
            fn from_after_reduce(vector: Self::SIMDVector) -> Self {
                V::from(vector.reduce_add())
            }
        }
    )*
    };
}

macro_rules! impl_reduce_add {
    ($dim: literal, $([$($f: ident),+], ($v: ty, $s: ty)),*) => {
    $(
        impl ReduceAdd for $v {
            type Output = [$s; $dim];

            #[inline]
            fn reduce_add(self) -> Self::Output {
                [$(self.$f.reduce_add()),+]
            }
        }
    )*
    };
}

macro_rules! impl_reduce_add_vec2 {
    ($(($v: ty, $s: ty)),*) => ($(impl_reduce_add!(2, [x, y], ($v, $s));)*)
}

macro_rules! impl_reduce_add_vec3 {
    ($(($v: ty, $s: ty)),*) => ($(impl_reduce_add!(3, [x, y, z], ($v, $s));)*)
}

macro_rules! impl_reduce_add_vec4 {
    ($(($v: ty, $s: ty)),*) => ($(impl_reduce_add!(4, [x, y, z, w], ($v, $s));)*)
}

impl SIMDScalar<8> for ultraviolet::f32x8 {
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn recip_sqrt(self) -> Self {
        self.recip_sqrt()
    }

    #[inline]
    fn recip(self) -> Self {
        self.recip()
    }
}

impl SIMDScalar<4> for ultraviolet::f64x4 {
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline]
    fn recip_sqrt(self) -> Self {
        1.0 / self.sqrt()
    }

    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
}

simd_vector!(
    8,
    ultraviolet::f32x8 => f32,
    (ultraviolet::Vec2x8 => ultraviolet::Vec2, 2),
    (ultraviolet::Vec3x8 => ultraviolet::Vec3, 3),
    (ultraviolet::Vec4x8 => ultraviolet::Vec4, 4)
);

simd_vector!(
    4,
    ultraviolet::f64x4 => f64,
    (ultraviolet::DVec2x4 => ultraviolet::DVec2, 2),
    (ultraviolet::DVec3x4 => ultraviolet::DVec3, 3),
    (ultraviolet::DVec4x4 => ultraviolet::DVec4, 4)
);

impl_reduce_add_vec2!((ultraviolet::Vec2x8, f32), (ultraviolet::DVec2x4, f64));

impl_reduce_add_vec3!((ultraviolet::Vec3x8, f32), (ultraviolet::DVec3x4, f64));

impl_reduce_add_vec4!((ultraviolet::Vec4x8, f32), (ultraviolet::DVec4x4, f64));
