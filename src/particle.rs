use crate::vector::Vector;

/// Trait to describe a particle which consists of a [position](Particle::position) and a [gravitational parameter mu](Particle::mu).
///
/// #### Deriving:
///
/// Used when the type has fields named `position` and `mu`:
///
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
/// Used when the type cannot directly provide a position and a gravitational parameter.
///
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
///
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
///
/// If you can't implement [`Particle`] on a type, you can use the fact that it is implemented for tuples of a vector and its scalar type instead of creating an intermediate type.
///
/// ```
/// # use particular::prelude::*;
/// # use glam::Vec3;
/// let particle = (Vec3::ONE, 5.0);
///
/// assert_eq!(particle.position(), Vec3::ONE);
/// assert_eq!(particle.mu(), 5.0);
/// ```
pub trait Particle {
    /// Type of the elements composing the elements of the [position](Particle::position) vector and [mu](Particle::mu).
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

/// Conversion to a point-mass.
///
/// A point-mass is a tuple of the [position](Particle::position) and the [gravitational parameter](Particle::mu) of a particle.
pub(crate) trait IntoPointMass<A>: Particle
where
    Self::Vector: Vector<A>,
{
    #[inline]
    fn point(&self) -> <Self::Vector as Vector<A>>::Internal {
        self.position().into_internal()
    }

    #[inline]
    fn point_mass(&self) -> (<Self::Vector as Vector<A>>::Internal, Self::Scalar) {
        (self.point(), self.mu())
    }
}

impl<P, A> IntoPointMass<A> for P
where
    P: Particle,
    P::Vector: Vector<A>,
{
}

impl<V, S> Particle for (V, S)
where
    S: Clone,
    V: Clone,
{
    type Scalar = S;

    type Vector = V;

    #[inline]
    fn position(&self) -> Self::Vector {
        self.0.clone()
    }

    #[inline]
    fn mu(&self) -> Self::Scalar {
        self.1.clone()
    }
}

impl<P> Particle for &P
where
    P: Particle,
{
    type Scalar = P::Scalar;

    type Vector = P::Vector;

    #[inline]
    fn position(&self) -> Self::Vector {
        (**self).position()
    }

    #[inline]
    fn mu(&self) -> Self::Scalar {
        (**self).mu()
    }
}

impl<P> Particle for &mut P
where
    P: Particle,
{
    type Scalar = P::Scalar;

    type Vector = P::Vector;

    #[inline]
    fn position(&self) -> Self::Vector {
        (**self).position()
    }

    #[inline]
    fn mu(&self) -> Self::Scalar {
        (**self).mu()
    }
}
