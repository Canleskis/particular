use crate::{
    compute_method::{Compute, ComputeMethod, Storage},
    particle::{IntoPointMass, Particle, ParticlePointMass},
};

/// Trait to compute the acceleration from an iterator of [`Particle`] objects using a provided [`ComputeMethod`].
pub trait Accelerations: Compute
where
    Self::Item: Particle,
{
    /// Returns the computed acceleration of each [`Particle`] using the provided [`ComputeMethod`].
    ///
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// # use glam::Vec2;
    /// let mut particles = vec![(Vec2::Y, 1.0), (Vec2::ZERO, 1.0)];
    /// let mut accelerations = particles.iter().accelerations(sequential::BruteForce);
    ///
    /// assert_eq!(accelerations.next().unwrap(), Vec2::NEG_Y);
    /// assert_eq!(accelerations.next().unwrap(), Vec2::Y);
    /// ```
    #[inline]
    fn accelerations<S, C>(self, cm: C) -> <C::Output as IntoIterator>::IntoIter
    where
        S: Storage<ParticlePointMass<Self::Item>>,
        C: ComputeMethod<S, <Self::Item as Particle>::Vector>,
    {
        self.map(|item| item.point_mass()).compute(cm)
    }
}

impl<I: Compute> Accelerations for I where I::Item: Particle {}
