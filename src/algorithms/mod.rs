#[cfg(feature = "gpu")]
mod wgpu_data;

/// Tree, bounding box and BarnesHut implementation details.
pub mod tree;

/// Internal representation of vectors used for expensive computations.
///
/// These traits and their methods are used by the built-in [`ComputeMethods`](crate::compute_method::ComputeMethod)
/// to abstract common internal vectors operations and their conversion from and into arbitrary vectors.
pub mod vector;

/// Compute methods that use the GPU.
#[cfg(feature = "gpu")]
pub mod gpu;

/// Compute methods that use multiple CPU threads.
#[cfg(feature = "parallel")]
pub mod parallel;

/// Compute methods that use one CPU thread.
pub mod sequential;

/// Built-in [`ComputeMethods`](crate::compute_method::ComputeMethod) modules.
pub mod compute_methods {
    #[cfg(feature = "gpu")]
    pub use super::gpu;

    #[cfg(feature = "parallel")]
    pub use super::parallel;

    pub use super::sequential;
}

pub use vector::*;

use crate::compute_method::Storage;

/// Point-mass representation of an object in space.
#[derive(Debug, Default, Clone, Copy)]
pub struct PointMass<V, S> {
    /// Position of the object.
    pub position: V,
    /// Mass of the object.
    pub mass: S,
}

impl<V, S> PointMass<V, S> {
    /// Creates a new [`PointMass`] with the given position and mass.
    pub fn new(position: V, mass: S) -> Self {
        Self { position, mass }
    }

    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<T, S>`] provided `V` implements [`Into<T>`].
    #[inline]
    pub fn into<T>(self) -> PointMass<T, S>
    where
        V: Into<T>,
    {
        PointMass {
            position: self.position.into(),
            mass: self.mass,
        }
    }

    /// Returns the position of the object.
    #[inline]
    pub fn position(self) -> V {
        self.position
    }

    /// Returns the mass of the object.
    #[inline]
    pub fn mass(self) -> S {
        self.mass
    }

    /// Returns true if the mass is zero.
    #[inline]
    pub fn is_massless(&self) -> bool
    where
        S: Default + PartialEq,
    {
        self.mass == S::default()
    }

    /// Returns false if the mass is zero.
    #[inline]
    pub fn is_massive(&self) -> bool
    where
        S: Default + PartialEq,
    {
        self.mass != S::default()
    }
}

impl<V, S> PointMass<V, S> {
    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<V::Internal, S>`] provided `V` implements [`Vector`].
    #[inline]
    pub fn into_internal<A>(self) -> PointMass<V::Internal, S>
    where
        V: Vector<A>,
    {
        PointMass {
            position: self.position.into_internal(),
            mass: self.mass,
        }
    }

    /// Converts from a [`PointMass<T, S>`] to a [`PointMass<V, S>`] provided `T` is the internal vector representation of `V`.
    #[inline]
    pub fn from_internal<T>(self) -> PointMass<T, S>
    where
        V: InternalVector<Scalar = S>,
        T: Vector<V::Array, Internal = V>,
    {
        PointMass {
            position: T::from_internal(self.position),
            mass: self.mass,
        }
    }
}

/// Storage for particles with massive and affected particles in two separate vectors.
pub struct FromMassive<T, S> {
    /// Particles used to compute the acceleration of the `affected` particles.
    pub massive: Vec<PointMass<T, S>>,
    /// Particles for which the acceleration is computed.
    ///
    /// This vector and the `massive` vector can share particles.
    pub affected: Vec<PointMass<T, S>>,
}

impl<T, S> FromMassive<T, S> {
    /// Creates a new [`FromMassive`] instance with the given vectors of particles.
    #[inline]
    pub fn new(massive: Vec<PointMass<T, S>>, affected: Vec<PointMass<T, S>>) -> Self {
        Self { massive, affected }
    }

    /// Creates a new [`FromMassive`] instance from an iterator of particles.
    ///
    /// This method collects the particles from the iterator into the `affected` vector and copies the ones with mass to populate the `massive` vector.
    #[inline]
    pub fn from(particles: impl Iterator<Item = PointMass<T, S>>) -> Self
    where
        T: Copy,
        S: Default + PartialEq + Copy,
    {
        let affected: Vec<_> = particles.collect();
        let massive: Vec<_> = affected
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect();

        Self { massive, affected }
    }
}

impl<T, S, V> Storage<PointMass<V, S>> for FromMassive<T, S>
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: Vector<T::Array, Internal = T> + 'static,
{
    #[inline]
    fn new(input: impl Iterator<Item = PointMass<V, S>>) -> Self {
        Self::from(input.map(PointMass::into_internal))
    }
}

/// Simple storage for particles using a vector.
pub struct ParticleSet<T, S> {
    /// Particles for which the acceleration is computed.
    pub particles: Vec<PointMass<T, S>>,
}

impl<T, S, V> Storage<PointMass<V, S>> for ParticleSet<T, S>
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: Vector<T::Array, Internal = T> + 'static,
{
    #[inline]
    fn new(input: impl Iterator<Item = PointMass<V, S>>) -> Self {
        ParticleSet {
            particles: input.map(PointMass::into_internal).collect(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::compute_method::ComputeMethod;
    use glam::Vec3A;

    pub fn acceleration_computation<S, C>(cm: C)
    where
        C: ComputeMethod<S, Vec3A>,
        S: Storage<PointMass<Vec3A, f32>>,
    {
        let massive = vec![
            PointMass::new(Vec3A::splat(0.0), 2.0),
            PointMass::new(Vec3A::splat(1.0), 3.0),
        ];

        let particles = vec![
            PointMass::new(Vec3A::splat(3.0), 0.0),
            massive[0],
            massive[1],
            PointMass::new(Vec3A::splat(5.0), 0.0),
        ];

        let storage = S::new(particles.clone().into_iter());
        let computed = cm.compute(storage);

        for (&point_mass1, computed) in particles.iter().zip(computed) {
            let mut acceleration = Vec3A::ZERO;

            for &point_mass2 in massive.iter() {
                let dir = point_mass2.position - point_mass1.position;
                let mag_2 = dir.length_squared();

                if mag_2 != 0.0 {
                    acceleration += dir * point_mass2.mass / (mag_2 * mag_2.sqrt());
                }
            }

            assert!((acceleration).abs_diff_eq(computed, f32::EPSILON))
        }
    }
}
