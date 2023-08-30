/// Trait for the identity element zero.
pub trait Zero {
    /// Zero value of the type.
    const ZERO: Self;
}

impl<const D: usize, S> Zero for [S; D]
where
    S: Zero,
{
    const ZERO: Self = [S::ZERO; D];
}

/// Marker trait for arbitrary vectors that can be converted from and into an array.
pub trait ConvertArray<const D: usize, S>: From<Self::Array> + Into<Self::Array> {
    /// Internal representation of a vector.
    type Array;
}

impl<const D: usize, S, V> ConvertArray<D, S> for V
where
    V: From<[S; D]> + Into<[S; D]>,
{
    type Array = [S; D];
}

/// Internal representation of vectors used for expensive computations.
pub mod internal {
    use super::{ConvertArray, Zero};
    use std::{
        fmt::Debug,
        iter::Sum,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    };

    /// Trait for arbitrary vectors that can be converted from and into a specified [`Vector`].
    pub trait ConvertInternal<const D: usize, S>: ConvertArray<D, S, Array = [S; D]> {
        /// Internal representation of a vector.
        type Vector: From<Self::Array> + Into<Self::Array> + Vector<Scalar = S>;

        /// Converts the arbitrary vector into its internal representation.
        #[inline]
        fn into_internal(self) -> Self::Vector {
            Self::Vector::from(self.into())
        }

        /// Converts the internal representation into the arbitrary vector.
        #[inline]
        fn from_internal(vector: Self::Vector) -> Self {
            Self::from(vector.into())
        }
    }

    /// Scalar types that compose [`Vector`] objects.
    pub trait Scalar:
        Sum
        + Sync
        + Send
        + Copy
        + Zero
        + Debug
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
    pub trait Vector:
        Sum
        + Sync
        + Send
        + Copy
        + Zero
        + Debug
        + Default
        + PartialEq
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Self::Scalar, Output = Self>
        + Div<Self::Scalar, Output = Self>
    {
        /// The scalar type of the vector.
        type Scalar: Scalar;

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

            impl Zero for $s {
                const ZERO: Self = 0.0;
            }
        $(
            impl Vector for $t {
                type Scalar = $s;

                #[inline]
                fn length_squared(self) -> $s {
                    self.length_squared()
                }
            }

            impl Zero for $t {
                const ZERO: Self = <$t>::ZERO;
            }

            impl<V> ConvertInternal<$dim, $s> for V
            where
                V: Into<[$s; $dim]> + From<[$s; $dim]>,
            {
                type Vector = $t;
            }
        )*
        }
    }

    internal_vector!(f32, (glam::Vec2, 2), (glam::Vec3A, 3), (glam::Vec4, 4));
    internal_vector!(f64, (glam::DVec2, 2), (glam::DVec3, 3), (glam::DVec4, 4));
}

/// SIMD representation of vectors used for expensive computations.
pub mod simd {
    use super::{ConvertArray, Zero};
    use std::{
        fmt::Debug,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    };

    /// Marker trait for arbitrary vectors and the types used for its SIMD representation.
    pub trait ConvertSIMD<const L: usize, const D: usize, S>:
        ConvertArray<D, S, Array = [S; D]>
    {
        /// SIMD representation of the [`ConvertSIMD::Vector`]'s scalar.
        type Scalar: Scalar<L, Element = S>;

        /// SIMD representation of the vector.
        type Vector: Vector<L, Element = Self::Array, Scalar = Self::Scalar>
            + ReduceAdd<Output = Self::Array>;
    }

    /// Trait for SIMD objects and their creation.
    pub trait SIMD<const L: usize> {
        /// Element from which the SIMD value can be created.
        type Element;

        /// Creates a SIMD value with all lanes set to the specified value.
        fn splat(value: Self::Element) -> Self;

        /// Creates a SIMD value with lanes set to the given values.
        fn from_lanes(values: [Self::Element; L]) -> Self;
    }

    /// Scalar types that compose [`Vector`] objects.
    pub trait Scalar<const L: usize>:
        Send
        + Sync
        + Copy
        + Zero
        + Debug
        + Default
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + PartialEq
        + SIMD<L>
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
    pub trait Vector<const L: usize>:
        Send
        + Sync
        + Copy
        + Zero
        + Debug
        + Default
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + ReduceAdd
        + SIMD<L>
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Self::Scalar, Output = Self>
        + Div<Self::Scalar, Output = Self>
    {
        /// The scalar type of the vector.
        type Scalar: Scalar<L>;

        /// Norm squared, defined by the dot product on itself.
        fn length_squared(self) -> Self::Scalar;

        /// Returns itself with NaNs replaced with zeros.
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
        ($l: literal, ($s: ty, $se: ty), $(($t: ty, $te: ty, $dim: literal)),*) => {
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

            impl Zero for $s {
                const ZERO: Self = <$s>::ZERO;
            }
        $(
            impl SIMD<$l> for $t {
                type Element = [$se; $dim];

                #[inline]
                fn splat(value: Self::Element) -> Self {
                    Self::splat(value.into())
                }

                #[inline]
                fn from_lanes(values: [Self::Element; $l]) -> Self {
                    Self::from(values.map(<$te>::from))
                }
            }

            impl Vector<$l> for $t {
                type Scalar = $s;

                #[inline]
                fn length_squared(self) -> Self::Scalar {
                    self.mag_sq()
                }

                #[inline]
                fn nan_to_zero(self) -> Self {
                    self.map(|vector| vector & wide::CmpEq::cmp_eq(vector, vector))
                }
            }

            impl Zero for $t {
                const ZERO: Self = Self::broadcast(<$s>::ZERO);
            }

            impl<V> ConvertSIMD<$l, $dim, $se> for V
            where
                V: From<[$se; $dim]> + Into<[$se; $dim]>,
            {
                type Scalar = $s;

                type Vector = $t;
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

    impl Scalar<8> for ultraviolet::f32x8 {
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

    impl Scalar<4> for ultraviolet::f64x4 {
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
        (ultraviolet::f32x8, f32),
        (ultraviolet::Vec2x8, ultraviolet::Vec2, 2),
        (ultraviolet::Vec3x8, ultraviolet::Vec3, 3),
        (ultraviolet::Vec4x8, ultraviolet::Vec4, 4)
    );

    simd_vector!(
        4,
        (ultraviolet::f64x4, f64),
        (ultraviolet::DVec2x4, ultraviolet::DVec2, 2),
        (ultraviolet::DVec3x4, ultraviolet::DVec3, 3),
        (ultraviolet::DVec4x4, ultraviolet::DVec4, 4)
    );

    impl_reduce_add_vec2!((ultraviolet::Vec2x8, f32), (ultraviolet::DVec2x4, f64));

    impl_reduce_add_vec3!((ultraviolet::Vec3x8, f32), (ultraviolet::DVec3x4, f64));

    impl_reduce_add_vec4!((ultraviolet::Vec4x8, f32), (ultraviolet::DVec4x4, f64));
}
