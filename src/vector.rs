/// Types that can be compared for equality and have a default value.
pub trait Scalar: Default + PartialEq {}
impl<U: Default + PartialEq> Scalar for U {}

/// Arbitrary vectors that can be converted from and into an array of length N.
pub trait Vector<const DIM: usize, S: Scalar>: Into<[S; DIM]> + From<[S; DIM]> {
    /// Internal representation of the vector [`Particular`](crate) can use for expensive computations.
    type Internal;

    fn from_internal(vector: Self::Internal) -> Self;
    fn into_internal(self) -> Self::Internal;
}

macro_rules! convertible {
    ($scalar: ty, $dim: literal, $internal: ty) => {
        impl<V> Vector<$dim, $scalar> for V
        where
            V: Into<[$scalar; $dim]> + From<[$scalar; $dim]>,
        {
            type Internal = $internal;

            fn from_internal(vector: Self::Internal) -> V {
                Self::from(vector.into())
            }

            fn into_internal(self) -> Self::Internal {
                Self::Internal::from(self.into())
            }
        }
    };
}

convertible!(u32, 2, glam::UVec2);
convertible!(u32, 3, glam::UVec3);
convertible!(u32, 4, glam::UVec4);

convertible!(i32, 2, glam::IVec2);
convertible!(i32, 3, glam::IVec3);
convertible!(i32, 4, glam::IVec4);

convertible!(f32, 3, glam::Vec3A);
convertible!(f32, 4, glam::Vec4);

convertible!(f64, 2, glam::DVec2);
convertible!(f64, 3, glam::DVec3);
convertible!(f64, 4, glam::DVec4);

impl<V> Vector<2, f32> for V
where
    V: Into<[f32; 2]> + From<[f32; 2]>,
{
    type Internal = glam::Vec3A;

    fn from_internal(vector: Self::Internal) -> Self {
        Self::from(vector.truncate().into())
    }

    fn into_internal(self) -> Self::Internal {
        let [x, y] = self.into();
        Self::Internal::from((x, y, 0.0))
    }
}
