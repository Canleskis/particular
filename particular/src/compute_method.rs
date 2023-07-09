/// Trait to perform a computation of values of type `V` between objects contained in a storage of type `S`.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use glam::{Vec3A, Vec3};
/// use particular::algorithms::MassiveAffected;
///
/// struct AccelerationCalculator;
///
/// impl ComputeMethod<MassiveAffected<Vec3A, f32>, Vec3> for AccelerationCalculator {
///     type Output = Vec<Vec3>;
///     
///     fn compute(self, storage: MassiveAffected<Vec3A, f32>) -> Self::Output {
///         // ...
///         # Vec::new()
///     }
/// }
/// ```
pub trait ComputeMethod<S, V> {
    /// IntoIterator that yields the computed values `V`.
    type Output: IntoIterator<Item = V>;

    /// Performs the computation between objects contained in the storage.
    fn compute(self, storage: S) -> Self::Output;
}

/// Trait for storage created from objects of type `P`.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3;
/// use particular::algorithms::PointMass;
///
/// // An emtpy storage...
/// struct MyStorage;
///
/// impl Storage<PointMass<Vec3, f32>> for MyStorage {
///     fn store<I: Iterator<Item = PointMass<Vec3, f32>>>(input: I) -> Self {
///         // ...
///         # MyStorage
///     }
/// }
/// ```
pub trait Storage<P> {
    /// Creates a new storage.
    fn store<I: Iterator<Item = P>>(input: I) -> Self;
}

/// Trait to perform a computation from an iterator using a provided [`ComputeMethod`].
pub trait Compute: Iterator + Sized {
    /// Returns the computed value of each item using the provided [`ComputeMethod`].
    #[inline]
    fn compute<S, C, V>(self, cm: C) -> <C::Output as IntoIterator>::IntoIter
    where
        S: Storage<Self::Item>,
        C: ComputeMethod<S, V>,
    {
        cm.compute(S::store(self)).into_iter()
    }
}

impl<I: Iterator> Compute for I {}

impl<C, S, V> ComputeMethod<S, V> for &C
where
    C: ComputeMethod<S, V> + Copy,
{
    type Output = C::Output;

    #[inline]
    fn compute(self, storage: S) -> Self::Output {
        (*self).compute(storage)
    }
}

impl<C, S, V> ComputeMethod<S, V> for &mut C
where
    C: ComputeMethod<S, V> + Copy,
{
    type Output = C::Output;

    #[inline]
    fn compute(self, storage: S) -> Self::Output {
        (*self).compute(storage)
    }
}
