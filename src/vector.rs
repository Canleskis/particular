pub(crate) use glam::Vec3A as SIMD;

/// Types that are able to be converted from and into an array of length N.
pub trait Vector<const N: usize>: Into<[f32; N]> + From<[f32; N]> {}
impl<const N: usize, T: Into<[f32; N]> + From<[f32; N]>> Vector<N> for T {}

/// Trait for the type of an arbitrary vector.
pub trait Descriptor {
    /// Return type of the computed accelerations.
    type Type;
}

/// Trait to convert a [`SIMD`](glam::Vec3A) vector to an arbitrary vector.
pub trait FromVector: Descriptor {
    fn from_simd(vector: SIMD) -> Self::Type;
}

/// Trait to convert an arbitrary vector to a [`SIMD`](glam::Vec3A) vector.
pub trait IntoVector: Descriptor {
    fn into_simd(vector: Self::Type) -> SIMD;
}

/// Describes an arbitrary vector of a given dimension `DIM`.
///
/// It allows an association between a dimension and an arbitrary vector so that `Particular` can use the appropriate SIMD conversion.
pub struct VectorDescriptor<const DIM: usize, V: Vector<DIM>>(std::marker::PhantomData<V>);

impl<const N: usize, T: Vector<N>> Descriptor for VectorDescriptor<N, T> {
    type Type = T;
}

impl<T: Vector<2>> FromVector for VectorDescriptor<2, T> {
    #[inline]
    fn from_simd(vector: SIMD) -> T {
        T::from(vector.truncate().into())
    }
}

impl<T: Vector<2>> IntoVector for VectorDescriptor<2, T> {
    #[inline]
    fn into_simd(vector: T) -> SIMD {
        let array = vector.into();
        SIMD::from((array[0], array[1], 0.0))
    }
}

impl<T: Vector<3>> FromVector for VectorDescriptor<3, T> {
    #[inline]
    fn from_simd(vector: SIMD) -> T {
        T::from(vector.into())
    }
}

impl<T: Vector<3>> IntoVector for VectorDescriptor<3, T> {
    #[inline]
    fn into_simd(vector: T) -> SIMD {
        SIMD::from(vector.into())
    }
}
