use crate::{
    compute_method::{Compute, ComputeMethod, ComputeResult, Storage},
    particle::{IntoPointMass, Particle, PointMass},
};

/// Trait to compute the acceleration of types implementing [`Particle`] from an iterator using a provided [`ComputeMethod`].
pub trait Accelerations: Compute
where
    Self::Item: Particle,
{
    /// Computes the acceleration of the iterated [`Particles`](Particle) using the provided [`ComputeMethod`].
    ///
    /// This method effectively returns the original iterator zipped with the acceleration of its particles. <br>
    ///
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// # use glam::Vec2;
    /// let mut particles = vec![(Vec2::Y, 1.0), (Vec2::ZERO, 1.0)];
    ///
    /// let mut accelerations = particles
    ///     .iter_mut()
    ///     .accelerations(&mut sequential::BruteForce);
    ///
    /// assert_eq!(accelerations.next().unwrap(), (&mut (Vec2::Y, 1.0), Vec2::NEG_Y));
    /// assert_eq!(accelerations.next().unwrap(), (&mut (Vec2::ZERO, 1.0), Vec2::Y));
    /// ```
    #[inline]
    fn accelerations<S, C>(self, cm: C) -> ComputeResult<Self::Item, C::Output>
    where
        Self: Sized,
        S: Storage<PointMass<Self::Item>>,
        C: ComputeMethod<S, <Self::Item as Particle>::Vector>,
    {
        self.compute(|item| item.point_mass(), cm)
    }
}

/// Trait to compute the acceleration of any type from an iterator using a provided [`ComputeMethod`].
pub trait MapAccelerations: Compute {
    /// Computes the acceleration of the iterated items using the provided [`ComputeMethod`] after calling the closure on them.
    /// The closure should return a type implementing [`Particle`]. <br>
    /// Note that [`Particle`] is implemented for tuples of a vector and its scalar type.
    ///
    /// This method effectively returns the original iterator zipped with the acceleration of its mapped particles. <br>
    ///
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// // Items are arrays of three floats.
    /// let mut particles = vec![[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]];
    ///
    /// // Map the item to a tuple, consisting of an array of the first two floats as the position and the third float as μ.
    /// let mut accelerations = particles.iter_mut().map_accelerations(
    ///     |array| ([array[0], array[1]], array[2]),
    ///     &mut sequential::BruteForce,
    /// );
    ///
    /// assert_eq!(accelerations.next().unwrap(), (&mut [0.0, 1.0, 1.0], [0.0, -1.0]));
    /// assert_eq!(accelerations.next().unwrap(), (&mut [0.0, 0.0, 1.0], [0.0, 1.0]));
    /// ```
    #[inline]
    fn map_accelerations<P, F, S, C>(self, mut f: F, cm: C) -> ComputeResult<Self::Item, C::Output>
    where
        Self: Sized,
        P: Particle,
        F: FnMut(&Self::Item) -> P,
        S: Storage<PointMass<P>>,
        C: ComputeMethod<S, P::Vector>,
    {
        self.compute(|item| f(item).point_mass(), cm)
    }
}

impl<I: Compute> Accelerations for I where I::Item: Particle {}

impl<I: Compute> MapAccelerations for I {}
