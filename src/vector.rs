/// Types that can be compared for equality and have a default value considered as their 'zero'.
pub trait Scalar: Default + PartialEq {}
impl<S: Default + PartialEq> Scalar for S {}

/// Arbitrary vectors that can be converted from and into an array of length N.
pub trait Vector<const DIM: usize, S: Scalar>: Into<[S; DIM]> + From<[S; DIM]> {
    /// Internal representation of the vector [`Particular`](crate) can use for expensive computations.
    type Internal;

    /// Convert the arbitrary vector into its internal representation.
    fn into_internal(self) -> Self::Internal;

    /// Convert the internal representation back into the arbitrary vector.
    fn from_internal(vector: Self::Internal) -> Self;
}

macro_rules! impl_vector {
    ($dim: literal, $s: ty, $i: ty) => {
        impl<V> Vector<$dim, $s> for V
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

impl_vector!(3, f32, glam::Vec3A);
impl_vector!(4, f32, glam::Vec4);

impl_vector!(2, f64, glam::DVec2);
impl_vector!(3, f64, glam::DVec3);
impl_vector!(4, f64, glam::DVec4);

impl<V> Vector<2, f32> for V
where
    V: Into<[f32; 2]> + From<[f32; 2]>,
{
    type Internal = glam::Vec3A;

    #[inline]
    fn into_internal(self) -> Self::Internal {
        let [x, y] = self.into();
        Self::Internal::from((x, y, 0.0))
    }

    #[inline]
    fn from_internal(vector: Self::Internal) -> Self {
        Self::from(vector.truncate().into())
    }
}
