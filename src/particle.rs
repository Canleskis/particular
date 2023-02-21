use crate::vector::Vector;

/// Trait to describe a particle which consists of a [position](Particle::position) and a [gravitational parameter mu](Particle::mu).
///
/// #### Deriving:
///
/// Used in most cases, when the type has fields named `position` and `mu`:
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3;
/// #
/// #[derive(Particle)]
/// struct Body {
///     position: Vec3,
///     mu: f32,
/// //  ...
/// }
/// ```
/// #### Manual implementation:
///
/// Used when the type has more complex fields and cannot directly provide a [position](Particle::position) and a [gravitational parameter](Particle::mu).
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
///     type Scalar = f32;
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
    /// Type of the elements composing the [position](Particle::position) vector and [mu](Particle::mu).
    type Scalar;

    /// Type of the [position](Particle::position).
    type Vector;

    /// The position of the particle in space.
    fn position(&self) -> Self::Vector;

    /// The [standard gravitational parameter](https://en.wikipedia.org/wiki/Standard_gravitational_parameter) of the particle, annotated `µ`.
    ///
    /// `µ = gravitational constant * mass`.
    fn mu(&self) -> Self::Scalar;
}

type VectorOf<P> = <P as Particle>::Vector;
type ScalarOf<P> = <P as Particle>::Scalar;

/// The internal type used for the position of a given [`Particle`].
pub type Internal<const DIM: usize, P> = <VectorOf<P> as Vector<DIM, ScalarOf<P>>>::Internal;

/// Conversion to a point-mass.
///
/// A point-mass is a tuple of the [position](Particle::position) and the [gravitational parameter](Particle::mu) of a particle.
pub(crate) trait ToPointMass<const DIM: usize>: Particle
where
    Self::Vector: Vector<DIM, Self::Scalar>,
{
    fn point(&self) -> Internal<DIM, Self>;

    fn point_mass(&self) -> (Internal<DIM, Self>, Self::Scalar);
}

impl<const DIM: usize, P> ToPointMass<DIM> for P
where
    P: Particle,
    P::Vector: Vector<DIM, P::Scalar>,
{
    #[inline]
    fn point(&self) -> Internal<DIM, P> {
        self.position().into_internal()
    }

    #[inline]
    fn point_mass(&self) -> (Internal<DIM, P>, P::Scalar) {
        (self.point(), self.mu())
    }
}
