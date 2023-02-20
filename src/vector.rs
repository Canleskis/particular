/// Types that can be compared for equality and have a default value.
pub trait Scalar: Default + PartialEq {}
impl<U: Default + PartialEq> Scalar for U {}

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
    ($scalar: ty, $dim: literal, $internal: ty) => {
        impl<V> Vector<$dim, $scalar> for V
        where
            V: Into<[$scalar; $dim]> + From<[$scalar; $dim]>,
        {
            type Internal = $internal;

            fn into_internal(self) -> Self::Internal {
                Self::Internal::from(self.into())
            }

            fn from_internal(vector: Self::Internal) -> V {
                Self::from(vector.into())
            }
        }
    };
}

impl_vector!(u32, 2, glam::UVec2);
impl_vector!(u32, 3, glam::UVec3);
impl_vector!(u32, 4, glam::UVec4);

impl_vector!(i32, 2, glam::IVec2);
impl_vector!(i32, 3, glam::IVec3);
impl_vector!(i32, 4, glam::IVec4);

impl_vector!(f32, 3, glam::Vec3A);
impl_vector!(f32, 4, glam::Vec4);

impl_vector!(f64, 2, glam::DVec2);
impl_vector!(f64, 3, glam::DVec3);
impl_vector!(f64, 4, glam::DVec4);

impl<V> Vector<2, f32> for V
where
    V: Into<[f32; 2]> + From<[f32; 2]>,
{
    type Internal = glam::Vec3A;

    fn into_internal(self) -> Self::Internal {
        let [x, y] = self.into();
        Self::Internal::from((x, y, 0.0))
    }
    
    fn from_internal(vector: Self::Internal) -> Self {
        Self::from(vector.truncate().into())
    }
}
