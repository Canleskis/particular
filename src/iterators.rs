use std::vec::IntoIter;

use crate::{
    compute_method::ComputeMethod,
    particle::{IntoPointMass, Particle},
    vector::Vector,
};

/// An iterator of items of type U and their computed acceleration.
///
/// It is returned by both [`accelerations`](Compute::accelerations) and [`map_accelerations`](MapCompute::map_accelerations).
#[derive(Debug)]
pub struct Accelerations<U, const DIM: usize, S, V>
where
    V: Vector<[S; DIM]>,
{
    iter: IntoIter<U>,
    accelerations: IntoIter<V::Internal>,
}

impl<U, const DIM: usize, S, V> Iterator for Accelerations<U, DIM, S, V>
where
    V: Vector<[S; DIM]>,
{
    type Item = (U, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            .zip(self.accelerations.next().map(Vector::from_internal))
    }
}

/// Blanket implementation for iterators of types implementing [`Particle`] to compute their acceleration.
pub trait Compute<const DIM: usize, P, T>: Iterator<Item = P> + Sized
where
    P: Particle,
    P::Vector: Vector<[P::Scalar; DIM], Internal = T>,
{
    /// Computes the acceleration of the iterated [`Particles`](Particle) using the provided [`ComputeMethod`].
    ///
    /// This method effectively returns the original iterator zipped with the acceleration of its particles. <br>
    /// The computation is achieved eagerly; when this method is called, the acceleration of all the particles is computed before iteration.
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
    fn accelerations<C>(self, cm: &mut C) -> Accelerations<P, DIM, P::Scalar, P::Vector>
    where
        C: ComputeMethod<T, P::Scalar>,
    {
        let items: Vec<_> = self.collect();

        Accelerations {
            accelerations: cm
                .compute(&items.iter().map(|i| i.point_mass()).collect::<Vec<_>>())
                .into_iter(),
            iter: items.into_iter(),
        }
    }
}

/// Blanket implementation for iterators of any type to compute their acceleration.
pub trait MapCompute<const DIM: usize, P, T>: Iterator + Sized
where
    P: Particle,
    P::Vector: Vector<[P::Scalar; DIM], Internal = T>,
{
    /// Computes the acceleration of the iterated items using the provided [`ComputeMethod`] after calling the closure on them.
    /// The closure should return a type implementing [`Particle`]. <br>
    /// Note that it is implemented for tuples of a vector and its scalar type.
    ///
    /// This method effectively returns the original iterator zipped with the acceleration of its mapped particles. <br>
    /// The computation is achieved eagerly; when this method is called, the acceleration of all the items is computed before iteration.
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
    fn map_accelerations<F, C>(
        self,
        mut f: F,
        cm: &mut C,
    ) -> Accelerations<Self::Item, DIM, P::Scalar, P::Vector>
    where
        F: FnMut(&Self::Item) -> P,
        C: ComputeMethod<T, P::Scalar>,
    {
        let items: Vec<_> = self.collect();

        Accelerations {
            accelerations: cm
                .compute(&items.iter().map(|i| f(i).point_mass()).collect::<Vec<_>>())
                .into_iter(),
            iter: items.into_iter(),
        }
    }
}

impl<const DIM: usize, P, T, I> Compute<DIM, P, T> for I
where
    P: Particle,
    I: Iterator<Item = P>,
    P::Vector: Vector<[P::Scalar; DIM], Internal = T>,
{
}

impl<const DIM: usize, P, T, I> MapCompute<DIM, P, T> for I
where
    P: Particle,
    I: Iterator,
    P::Vector: Vector<[P::Scalar; DIM], Internal = T>,
{
}
