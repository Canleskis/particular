/// Trait to perform a computation of values of type `V` between objects contained in a storage of type `S`.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use glam::{Vec3A, Vec3};
/// struct AccelerationCalculator;
///
/// impl compute_method::ComputeMethod<FromMassive<Vec3A, f32>, Vec3> for AccelerationCalculator {
///     type Output = Box<dyn Iterator<Item = Vec3>>;
///     
///     fn compute(self, storage: FromMassive<Vec3A, f32>) -> Self::Output {
///         // ...
///         # Box::new(Vec::new().into_iter())
///     }
/// }
/// ```
pub trait ComputeMethod<S, V> {
    /// Iterator that yields the computed values `V`.
    type Output: IntoIterator<Item = V>;

    /// Performs the computation between objects contained in the storage.
    ///
    /// The computed values of type `V` are returned as an iterator defined by [`ComputeMethod::Output`].
    fn compute(self, storage: S) -> Self::Output;
}

/// Trait for storage created from objects of type `P`.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use compute_method::*;
/// # use glam::Vec3;
/// // An emtpy storage...
/// struct MyStorage;
///
/// impl Storage<PointMass<Vec3, f32>> for MyStorage {
///     fn new(input: impl Iterator<Item = PointMass<Vec3, f32>>) -> Self {
///         // ...
///         # MyStorage
///     }
/// }
/// ```
pub trait Storage<P> {
    /// Creates a new storage.
    fn new(input: impl Iterator<Item = P>) -> Self;
}

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

use std::{iter, vec};
pub(crate) type ComputeResult<U, O> = iter::Zip<vec::IntoIter<U>, <O as IntoIterator>::IntoIter>;

/// Trait to perform a computation from an iterator using a provided [`ComputeMethod`].
pub trait Compute: Iterator {
    /// Performs the computation on the iterated items using the provided [`ComputeMethod`] after calling the closure on them.
    ///
    /// This method effectively returns the original iterator zipped with the computed value of each item.
    #[inline]
    fn compute<B, F, S, C, V>(self, f: F, cm: C) -> ComputeResult<Self::Item, C::Output>
    where
        Self: Sized,
        F: FnMut(&Self::Item) -> B,
        S: Storage<B>,
        C: ComputeMethod<S, V>,
    {
        let items: Vec<_> = self.collect();
        let storage = S::new(items.iter().map(f));

        items.into_iter().zip(cm.compute(storage).into_iter())
    }
}

impl<I: Iterator> Compute for I {}
