/// Trait to perform a computation of values of type `V` between objects contained in a storage of type `S`.
///
/// # Example
///
/// ```
/// # use particular::prelude::*;
/// # use ultraviolet::Vec3;
/// use particular::algorithms::PointMass;
///
/// struct AccelerationCalculator;
///
/// impl ComputeMethod<&[PointMass<Vec3, f32>]> for AccelerationCalculator {
///     type Output = Vec<Vec3>;
///     
///     fn compute(&mut self, storage: &[PointMass<Vec3, f32>]) -> Self::Output {
///         // ...
///         # Vec::new()
///     }
/// }
/// ```
pub trait ComputeMethod<Storage> {
    /// IntoIterator that yields the computed values.
    type Output: IntoIterator;

    /// Performs the computation between objects contained in the storage.
    fn compute(&mut self, storage: Storage) -> Self::Output;
}
