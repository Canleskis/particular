/// Arbitrary vectors that can be converted from and into an array.
pub trait Vector<A>: Into<A> + From<A> {
    /// Internal representation of the vector [`Particular`](crate) can use for expensive computations.
    type Internal;

    /// Convert the arbitrary vector into its internal representation.
    fn into_internal(self) -> Self::Internal;

    /// Convert the internal representation back into the arbitrary vector.
    fn from_internal(vector: Self::Internal) -> Self;
}

/// Internal vectors with a norm squared and its root.
pub trait Normed {
    /// The type of the norm
    type Output;

    /// Norm squared, defined by the dot product on itself.
    fn length_squared(self) -> Self::Output;

    /// Square root operation on the type of the norm.
    fn sqrt(f: Self::Output) -> Self::Output;
}

macro_rules! impl_vector {
    ($dim: literal, $s: ty, $i: ty) => {
        impl Normed for $i {
            type Output = $s;

            #[inline]
            fn length_squared(self) -> $s {
                self.length_squared()
            }

            #[inline]
            fn sqrt(s: $s) -> $s {
                <$s>::sqrt(s)
            }
        }

        impl<V> Vector<[$s; $dim]> for V
        where
            V: Into<[$s; $dim]> + From<[$s; $dim]>,
        {
            type Internal = $i;

            #[inline]
            fn into_internal(self) -> Self::Internal {
                Self::Internal::from(self.into())
            }

            #[inline]
            fn from_internal(vector: Self::Internal) -> V {
                Self::from(vector.into())
            }
        }
    };
}

impl_vector!(2, f32, glam::Vec2);
impl_vector!(3, f32, glam::Vec3A);
impl_vector!(4, f32, glam::Vec4);

impl_vector!(2, f64, glam::DVec2);
impl_vector!(3, f64, glam::DVec3);
impl_vector!(4, f64, glam::DVec4);
