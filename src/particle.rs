use glam::Vec3;

/// Trait to describe a particle which consists of a `position` and a gravitational parameter `mu`.
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
    /// The position of the particle described by a [`Vec3`].
    fn position(&self) -> Vec3;

    /// The [standard gravitational parameter](https://en.wikipedia.org/wiki/Standard_gravitational_parameter) of the particle, annoted `µ`.
    ///
    /// `µ = gravitational constant * mass of the particle`.
    fn mu(&self) -> f32;
}

pub(crate) type PointMass = (Vec3, f32);

pub(crate) trait ToPointMass {
    fn point_mass(&self) -> PointMass;
}

impl<P: Particle> ToPointMass for P {
    fn point_mass(&self) -> PointMass {
        (self.position(), self.mu())
    }
}
