pub(crate) use {
    std::{
        iter::Sum,
        ops::{Add, AddAssign, BitAnd, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
    wide::CmpNe,
};

pub use ultraviolet::{
    f32x4, f32x8, f64x2, f64x4, DVec2, DVec2x2, DVec2x4, DVec3, DVec3x2, DVec3x4, DVec4, DVec4x2,
    DVec4x4, Vec2, Vec2x4, Vec2x8, Vec3, Vec3x4, Vec3x8, Vec4, Vec4x4, Vec4x8,
};

/// Trait for array types.
pub trait Array {
    /// The type of the elements in the array.
    type Item;
}
impl<I, const D: usize> Array for [I; D] {
    type Item = I;
}

/// Trait for the identity element `zero`.
pub trait Zero {
    /// `zero` value of the type.
    const ZERO: Self;
}

/// Trait for the identity element `one`.
pub trait One {
    /// `one` value of the type.
    const ONE: Self;
}

/// Trait for the element `infinity`.
pub trait Infinity {
    /// `infinity` (∞) value of the type.
    fn infinity() -> Self;
}

/// Trait for operations on floating-point numbers.
pub trait FloatOps:
    Sized
    + Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
{
}
impl<F> FloatOps for F where
    F: Sized
        + Neg<Output = Self>
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
        + Div<Output = Self>
        + DivAssign
{
}

/// Trait for floating-point numbers.
pub trait Float: One + Zero + Clone + Infinity + FloatOps + PartialEq {
    /// Returns the reciprocal (inverse) of a float.
    #[inline]
    fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Returns the square root of a float.
    fn sqrt(self) -> Self;

    /// Returns the reciprocal (inverse) square root of a float.
    #[inline]
    fn rsqrt(self) -> Self {
        self.recip().sqrt()
    }

    /// Returns the minimum between two floats.
    fn min(self, rhs: Self) -> Self;

    /// Returns the maximum between two floats.
    fn max(self, rhs: Self) -> Self;

    /// Returns the mean of two floats.
    #[inline]
    fn mean(self, rhs: Self) -> Self {
        (self + rhs) / (Self::ONE + Self::ONE)
    }
}

/// Trait for types that can be converted into an array.
pub trait IntoArray: Into<Self::Array> {
    /// The array the type can be converted into.
    type Array: Array;
}

/// Trait for operations on vectors of floating-point numbers.
pub trait FloatVectorOps<F>:
    Sized
    + IntoArray
    + Sum<Self>
    + Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<F, Output = Self>
    + MulAssign<F>
    + Div<F, Output = Self>
    + DivAssign<F>
{
}
impl<V, F> FloatVectorOps<F> for V where
    V: Sized
        + IntoArray
        + Sum<Self>
        + Neg<Output = Self>
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<F, Output = Self>
        + MulAssign<F>
        + Div<F, Output = Self>
        + DivAssign<F>
{
}

/// Trait for vectors of floating-point numbers.
pub trait FloatVector: Zero + Clone + FloatVectorOps<Self::Float> {
    /// Floating-point associated with the vector.
    type Float;

    /// Returns the norm squared of the vector.
    #[doc(alias = "length_squared")]
    #[doc(alias = "magnitude_squared")]
    fn norm_squared(self) -> Self::Float;
}

/// Trait for SIMD objects and their creation.
pub trait SIMD {
    /// Element from which the SIMD value can be created.
    type Element;

    /// The type a lane is equivalent to.
    type Lane;

    /// Creates a SIMD value with all lanes set to the specified value.
    fn splat(element: Self::Element) -> Self;

    /// Creates a SIMD value with lanes set to the given values.
    fn new_lane(lane: Self::Lane) -> Self;
}

/// Marker trait for [`SIMD::Element`]s.
pub trait SIMDElement<const L: usize>: Sized {
    /// The [`SIMD`] type that `Self` is an element of.
    type SIMD: SIMD<Element = Self, Lane = [Self; L]>;
}

/// Trait to reduce the lanes of a SIMD vector.
pub trait Reduce: SIMD {
    /// Sums the lanes of a SIMD vector.
    fn reduce_sum(self) -> Self::Element;
}

/// Trait for casting from one primitive to another.
pub trait FromPrimitive<U> {
    /// Converts to this primitive from the input primitive.
    fn from(p: U) -> Self;
}

/// Trait for casting one primitive to another.
pub trait AsPrimitive: Sized {
    /// Converts this primitive into the input primitive.
    #[inline]
    fn as_<F: FromPrimitive<Self>>(self) -> F {
        F::from(self)
    }
}
impl<U> AsPrimitive for U {}

macro_rules! impl_zero_value {
    ($s: ty, $zero: expr) => {
        impl Zero for $s {
            const ZERO: Self = $zero;
        }
    };
}

macro_rules! impl_float_values {
    ($s: ty, $zero: expr, $one: expr, $inf: expr) => {
        impl_zero_value!($s, $zero);

        impl One for $s {
            const ONE: Self = $one;
        }

        impl Infinity for $s {
            #[inline]
            fn infinity() -> Self {
                $inf
            }
        }
    };
}

impl_float_values!(f32, 0.0, 1.0, Self::INFINITY);
impl_float_values!(f32x4, Self::ZERO, Self::ONE, Self::splat(f32::INFINITY));
impl_float_values!(f32x8, Self::ZERO, Self::ONE, Self::splat(f32::INFINITY));
impl_float_values!(f64, 0.0, 1.0, Self::INFINITY);
impl_float_values!(f64x2, Self::ZERO, Self::ONE, Self::splat(f64::INFINITY));
impl_float_values!(f64x4, Self::ZERO, Self::ONE, Self::splat(f64::INFINITY));

impl_zero_value!(Vec2, Self::broadcast(f32::ZERO));
impl_zero_value!(Vec3, Self::broadcast(f32::ZERO));
impl_zero_value!(Vec4, Self::broadcast(f32::ZERO));
impl_zero_value!(DVec2, Self::broadcast(f64::ZERO));
impl_zero_value!(DVec3, Self::broadcast(f64::ZERO));
impl_zero_value!(DVec4, Self::broadcast(f64::ZERO));
impl_zero_value!(Vec2x4, Self::broadcast(f32x4::ZERO));
impl_zero_value!(Vec2x8, Self::broadcast(f32x8::ZERO));
impl_zero_value!(Vec3x4, Self::broadcast(f32x4::ZERO));
impl_zero_value!(Vec3x8, Self::broadcast(f32x8::ZERO));
impl_zero_value!(Vec4x4, Self::broadcast(f32x4::ZERO));
impl_zero_value!(Vec4x8, Self::broadcast(f32x8::ZERO));
impl_zero_value!(DVec2x2, Self::broadcast(f64x2::ZERO));
impl_zero_value!(DVec2x4, Self::broadcast(f64x4::ZERO));
impl_zero_value!(DVec3x2, Self::broadcast(f64x2::ZERO));
impl_zero_value!(DVec3x4, Self::broadcast(f64x4::ZERO));
impl_zero_value!(DVec4x2, Self::broadcast(f64x2::ZERO));
impl_zero_value!(DVec4x4, Self::broadcast(f64x4::ZERO));

macro_rules! impl_float {
    ($s: ty, $recip: expr, $recip_sqrt: expr) => {
        #[allow(clippy::redundant_closure_call)]
        impl Float for $s {
            #[inline]
            fn recip(self) -> Self {
                $recip(self)
            }

            #[inline]
            fn sqrt(self) -> Self {
                self.sqrt()
            }

            #[inline]
            fn rsqrt(self) -> Self {
                $recip_sqrt(self)
            }

            #[inline]
            fn min(self, rhs: Self) -> Self {
                self.min(rhs)
            }

            #[inline]
            fn max(self, rhs: Self) -> Self {
                self.max(rhs)
            }
        }
    };
}

impl_float!(f32, Self::recip, |f| Self::recip(f).sqrt());
impl_float!(f32x4, Self::recip, Self::recip_sqrt);
impl_float!(f32x8, Self::recip, Self::recip_sqrt);
impl_float!(f64, Self::recip, |f| Self::recip(f).sqrt());
impl_float!(f64x2, |f| 1.0 / f, |f| Self::recip(f).sqrt());
impl_float!(f64x4, |f| 1.0 / f, |f| Self::recip(f).sqrt());

macro_rules! impl_into_array {
    ($vector: ty, [$float: ty; $dim: literal]) => {
        impl IntoArray for $vector {
            type Array = [$float; $dim];
        }
    };
}

impl_into_array!(Vec2, [f32; 2]);
impl_into_array!(Vec3, [f32; 3]);
impl_into_array!(Vec4, [f32; 4]);
impl_into_array!(DVec2, [f64; 2]);
impl_into_array!(DVec3, [f64; 3]);
impl_into_array!(DVec4, [f64; 4]);
impl_into_array!(Vec2x4, [f32x4; 2]);
impl_into_array!(Vec2x8, [f32x8; 2]);
impl_into_array!(Vec3x4, [f32x4; 3]);
impl_into_array!(Vec3x8, [f32x8; 3]);
impl_into_array!(Vec4x4, [f32x4; 4]);
impl_into_array!(Vec4x8, [f32x8; 4]);
impl_into_array!(DVec2x2, [f64x2; 2]);
impl_into_array!(DVec2x4, [f64x4; 2]);
impl_into_array!(DVec3x2, [f64x2; 3]);
impl_into_array!(DVec3x4, [f64x4; 3]);
impl_into_array!(DVec4x2, [f64x2; 4]);
impl_into_array!(DVec4x4, [f64x4; 4]);

macro_rules! impl_float_vector {
    ($vector: ty, $float: ty) => {
        impl FloatVector for $vector {
            type Float = $float;

            #[inline]
            fn norm_squared(self) -> Self::Float {
                self.mag_sq()
            }
        }
    };
}

impl_float_vector!(Vec2, f32);
impl_float_vector!(Vec3, f32);
impl_float_vector!(Vec4, f32);
impl_float_vector!(DVec2, f64);
impl_float_vector!(DVec3, f64);
impl_float_vector!(DVec4, f64);
impl_float_vector!(Vec2x4, f32x4);
impl_float_vector!(Vec2x8, f32x8);
impl_float_vector!(Vec3x4, f32x4);
impl_float_vector!(Vec3x8, f32x8);
impl_float_vector!(Vec4x4, f32x4);
impl_float_vector!(Vec4x8, f32x8);
impl_float_vector!(DVec2x2, f64x2);
impl_float_vector!(DVec2x4, f64x4);
impl_float_vector!(DVec3x2, f64x2);
impl_float_vector!(DVec3x4, f64x4);
impl_float_vector!(DVec4x2, f64x2);
impl_float_vector!(DVec4x4, f64x4);

macro_rules! impl_simd {
    ($simd: ty, $el: ty, $lane: literal, $splat: expr, $new_lane: expr) => {
        #[allow(clippy::redundant_closure_call)]
        impl SIMD for $simd {
            type Element = $el;

            type Lane = [Self::Element; $lane];

            #[inline]
            fn splat(element: Self::Element) -> Self {
                $splat(element)
            }

            #[inline]
            fn new_lane(lane: Self::Lane) -> Self {
                $new_lane(lane)
            }
        }

        impl SIMDElement<$lane> for $el {
            type SIMD = $simd;
        }
    };
}

macro_rules! impl_simd_float {
    ($simd: ty, [$el: ty; $lane: literal]) => {
        impl_simd!($simd, $el, $lane, Self::splat, Self::from);
    };
}

macro_rules! impl_simd_vector {
    ($simd: ty, [$el: ty; $lane: literal]) => {
        impl_simd!(
            $simd,
            $el,
            $lane,
            |e| Self::splat(<$el>::from(e)),
            |l: Self::Lane| Self::from(l.map(<$el>::from))
        );
    };
}

impl_simd_float!(f32x4, [f32; 4]);
impl_simd_float!(f32x8, [f32; 8]);
impl_simd_float!(f64x2, [f64; 2]);
impl_simd_float!(f64x4, [f64; 4]);

impl_simd_vector!(Vec2x4, [Vec2; 4]);
impl_simd_vector!(Vec2x8, [Vec2; 8]);
impl_simd_vector!(Vec3x4, [Vec3; 4]);
impl_simd_vector!(Vec3x8, [Vec3; 8]);
impl_simd_vector!(Vec4x4, [Vec4; 4]);
impl_simd_vector!(Vec4x8, [Vec4; 8]);
impl_simd_vector!(DVec2x2, [DVec2; 2]);
impl_simd_vector!(DVec2x4, [DVec2; 4]);
impl_simd_vector!(DVec3x2, [DVec3; 2]);
impl_simd_vector!(DVec3x4, [DVec3; 4]);
impl_simd_vector!(DVec4x2, [DVec4; 2]);
impl_simd_vector!(DVec4x4, [DVec4; 4]);

macro_rules! impl_reduce {
    ($vector: ty, [$($f: ident),+]) => {
        impl Reduce for $vector {
            #[inline]
            fn reduce_sum(self) -> Self::Element {
                Self::Element {
                    $($f: self.$f.reduce_add()),+
                }
            }
        }
    };
}

impl_reduce!(Vec2x4, [x, y]);
impl_reduce!(Vec2x8, [x, y]);
impl_reduce!(Vec3x4, [x, y, z]);
impl_reduce!(Vec3x8, [x, y, z]);
impl_reduce!(Vec4x4, [x, y, z, w]);
impl_reduce!(Vec4x8, [x, y, z, w]);
impl_reduce!(DVec2x2, [x, y]);
impl_reduce!(DVec2x4, [x, y]);
impl_reduce!(DVec3x2, [x, y, z]);
impl_reduce!(DVec3x4, [x, y, z]);
impl_reduce!(DVec4x2, [x, y, z, w]);
impl_reduce!(DVec4x4, [x, y, z, w]);

macro_rules! impl_from_primitive {
    ($p1: ty => ($($pn: ty),*)) => {$(
        impl FromPrimitive<$pn> for $p1 {
            #[inline]
            fn from(p: $pn) -> Self {
                p as Self
            }
        }
    )*};
}

impl_from_primitive!(f32 => (usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f64));
impl_from_primitive!(f64 => (usize, u8, u16, u32, u64, u128, isize, i8, i16, i32, i64, i128, f32));
