use crate::NormedVector;

/// Trait to describe a particle which consists of a [position](Particle::position) and a [gravitational parameter mu](Particle::mu).
///
/// ## Implementing the [`Particle`] trait.
///
/// #### Deriving:
///
/// Used in most cases, when your type has fields named `position` and `mu`
/// ```
/// # use particular::prelude::Particle;
/// # use glam::Vec3;
/// #
/// #[derive(Particle)]
/// pub struct Body {
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
/// # use particular::Particle;
/// # use glam::Vec3;
/// #
/// struct Body {
///     position: Vec3,
///     mass: f32,
/// //  ...
/// }
///
/// impl Particle for Body {
///     type Vector = Vec3;
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
    /// The type used to describe the particle's position.
    type Vector: NormedVector;

    /// The position of the particle described by a [`NormedVector`](crate::NormedVector).
    fn position(&self) -> Self::Vector;

    /// The [standard gravitational parameter](https://en.wikipedia.org/wiki/Standard_gravitational_parameter) of the particle, annoted `µ`.
    ///
    /// `µ = gravitational constant * mass of the particle`.
    fn mu(&self) -> f32;
}

pub(crate) trait ToPointMass<T> {
    fn point_mass(&self) -> (T, f32);
}

impl<P: Particle> ToPointMass<P::Vector> for P {
    #[inline]
    fn point_mass(&self) -> (P::Vector, f32) {
        (self.position(), self.mu())
    }
}
