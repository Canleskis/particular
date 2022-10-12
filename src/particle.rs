use crate::vector::{Descriptor, FromVector, IntoVector, SIMD};

/// Trait to describe a particle which consists of a [position](Particle::position) and a [gravitational parameter mu](Particle::mu).
///
/// ## Implementing the [`Particle`] trait.
///
/// #### Deriving:
///
/// Used in most cases, when your type has a field named `mu`, annotating the position field with `#[dim]`:
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3;
/// #
/// #[derive(Particle)]
/// pub struct Body {
///     #[dim(3)]
///     position: Vec3,
///     mu: f32,
/// //  ...
/// }
/// ```
///
/// #### Manual implementation:
///
/// Used when your type has more complex fields and cannot directly provide a position and a gravitational parameter.
/// ```
/// # const G: f32 = 1.0;
/// #
/// # use particular::prelude::*;
/// # use glam::Vec3;
/// #
/// struct Body {
///     position: Vec3,
///     mass: f32,
/// //  ...
/// }
///
/// impl Particle for Body {
///     type Vector = VectorDescriptor<3, Vec3>;
///
///     fn position(&self) -> Vec3 {
///         self.position
///     }
///     
///     fn mu(&self) -> f32 {
///         self.mass * G
///     }
/// }
/// ```
pub trait Particle {
    /// Descriptor for the the type used for the particle's position.
    ///
    /// Use a [`VectorDescriptor`](crate::vector::VectorDescriptor).
    type Vector: FromVector + IntoVector;

    /// The position of the particle.
    ///
    /// Expects to return the type described by [`Particle::Vector`]
    fn position(&self) -> <Self::Vector as Descriptor>::Type;

    /// The [standard gravitational parameter](https://en.wikipedia.org/wiki/Standard_gravitational_parameter) of the particle, annoted `µ`.
    ///
    /// `µ = gravitational constant * mass`.
    fn mu(&self) -> f32;
}

/// Convertion to a point-mass.
///
/// A point-mass is a tuple of the position and the gravitational parameter of the particle.
pub(crate) trait ToPointMass {
    fn point(&self) -> SIMD;

    fn point_mass(&self) -> (SIMD, f32);
}

impl<P: Particle> ToPointMass for P {
    #[inline]
    fn point(&self) -> SIMD {
        P::Vector::into_simd(self.position())
    }

    #[inline]
    fn point_mass(&self) -> (SIMD, f32) {
        (self.point(), self.mu())
    }
}
