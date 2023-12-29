use crate::compute_method::{
    math::{Array, Float, FloatVector},
    storage::{ParticleReordered, PointMass},
    ComputeMethod,
};
use std::vec::IntoIter;

/// Trait to describe a particle which consists of a [position](Particle::position) and a [gravitational parameter mu](Particle::mu).
///
/// #### Deriving:
///
/// Used when the type has fields named `position` and `mu`:
///
/// ```
/// # use particular::prelude::*;
/// # use ultraviolet::Vec3;
/// #
/// #[derive(Particle)]
/// #[dim(3)]
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
/// # use ultraviolet::Vec3;
/// #
/// struct Body {
///     position: Vec3,
///     mass: f32,
/// //  ...
/// }
///
/// impl Particle for Body {
///     type Array = [f32; 3];
///
///     fn position(&self) -> [f32; 3] {
///         self.position.into()
///     }
///     
///     fn mu(&self) -> f32 {
///         self.mass * G
///     }
/// }
/// ```
///
/// If you can't implement [`Particle`] on a type, you can use the fact that it is implemented for tuples of an array and its scalar type instead of creating an intermediate type.
///
/// ```
/// # use particular::prelude::*;
/// let particle = ([1.0, 1.0, 0.0], 5.0);
///
/// assert_eq!(particle.position(), [1.0, 1.0, 0.0]);
/// assert_eq!(particle.mu(), 5.0);
/// ```
pub trait Particle {
    /// Type of the [position](Particle::position).
    type Array: Array;

    /// The position of the particle in space.
    fn position(&self) -> Self::Array;

    /// The [standard gravitational parameter](https://en.wikipedia.org/wiki/Standard_gravitational_parameter) of the particle, annotated `µ`.
    ///
    /// `µ = gravitational constant * mass`.
    fn mu(&self) -> <Self::Array as Array>::Item;
}

/// Marker trait to bind a [`FloatVector`] to an array.
pub trait ScalarArray: Array + Sized {
    /// Associated [`FloatVector`] of the array, which should ideally have the same layout.
    type Vector: FloatVector<Float = Self::Item> + From<Self> + Into<Self>;
}

macro_rules! impl_scalar_array {
    ([$scalar: ty; $dim: literal], $vector: ty) => {
        impl ScalarArray for [$scalar; $dim] {
            type Vector = $vector;
        }
    };
}

impl_scalar_array!([f32; 2], ultraviolet::Vec2);
impl_scalar_array!([f32; 3], ultraviolet::Vec3);
impl_scalar_array!([f32; 4], ultraviolet::Vec4);
impl_scalar_array!([f64; 2], ultraviolet::DVec2);
impl_scalar_array!([f64; 3], ultraviolet::DVec3);
impl_scalar_array!([f64; 4], ultraviolet::DVec4);

/// Associated [`Array`](Particle::Array) of a [`Particle`].
pub type ParticleArray<P> = <P as Particle>::Array;

/// Associated vector of the [`Array`](Particle::Array) of a [`Particle`].
pub type ParticleVector<P> = <<P as Particle>::Array as ScalarArray>::Vector;

/// Associated item of the [`Array`](Particle::Array) of a [`Particle`].
pub type ParticleScalar<P> = <<P as Particle>::Array as Array>::Item;

/// Trait to convert a type implementing [`Particle`] to a [`PointMass`].
pub trait IntoPointMass: Particle + Sized {
    /// Converts the particle to a [`PointMass`].
    #[inline]
    fn point_mass(&self) -> PointMass<ParticleVector<Self>, ParticleScalar<Self>>
    where
        Self::Array: ScalarArray,
    {
        PointMass::new(self.position().into(), self.mu())
    }
}
impl<P: Particle> IntoPointMass for P {}

/// Marker trait for [`ComputeMethod`]s implemented with a [`ParticleReordered`] storage for a [`Particle`] of type `P`.
pub trait ReorderedCompute<P>:
    for<'a> ComputeMethod<
    ParticleReordered<'a, ParticleVector<P>, ParticleScalar<P>>,
    Output = Vec<ParticleVector<P>>,
>
where
    P: Particle,
    P::Array: ScalarArray,
{
}
impl<C, P> ReorderedCompute<P> for C
where
    P: Particle,
    P::Array: ScalarArray,
    for<'a> C: ComputeMethod<
        ParticleReordered<'a, ParticleVector<P>, ParticleScalar<P>>,
        Output = Vec<ParticleVector<P>>,
    >,
{
}

/// Trait to perform the computation of accelerations using a provided [`ComputeMethod`] from an iterator of [`Particle`] objects.
pub trait Accelerations: Iterator + Sized
where
    Self::Item: Particle,
{
    /// Returns the computed acceleration of each [`Particle`] using the provided [`ComputeMethod`].
    ///
    /// # Example
    /// ```
    /// # use particular::prelude::*;
    /// let mut particles = vec![([0.0, 1.0], 1.0), ([0.0, 0.0], 1.0)];
    /// let mut accelerations = particles.iter().accelerations(&mut sequential::BruteForceScalar);
    ///
    /// assert_eq!(accelerations.next().unwrap(), [0.0, -1.0]);
    /// assert_eq!(accelerations.next().unwrap(), [0.0, 1.0]);
    /// ```
    #[inline]
    fn accelerations<C>(self, cm: &mut C) -> IntoIter<ParticleArray<Self::Item>>
    where
        ParticleArray<Self::Item>: ScalarArray,
        ParticleScalar<Self::Item>: Float,
        C: ReorderedCompute<Self::Item>,
    {
        #[inline]
        fn scalar_to_array<A>(vec: Vec<A::Vector>) -> Vec<A>
        where
            A: ScalarArray,
        {
            // If the array and its associated vector have the same layout (which is the case for all current implementations of `ScalarArray`),
            // this doesn't actually allocate and is a no-op.
            vec.into_iter().map(Into::into).collect()
        }

        scalar_to_array(cm.compute(ParticleReordered::from(
            &*self.map(|p| p.point_mass()).collect::<Vec<_>>(),
        )))
        .into_iter()
    }
}
impl<I: Iterator> Accelerations for I where I::Item: Particle {}

impl<P: Particle> Particle for &P {
    type Array = P::Array;

    #[inline]
    fn position(&self) -> Self::Array {
        (**self).position()
    }

    #[inline]
    fn mu(&self) -> <Self::Array as Array>::Item {
        (**self).mu()
    }
}

impl<P: Particle> Particle for &mut P {
    type Array = P::Array;

    #[inline]
    fn position(&self) -> Self::Array {
        (**self).position()
    }

    #[inline]
    fn mu(&self) -> <Self::Array as Array>::Item {
        (**self).mu()
    }
}

impl<const D: usize, S: Clone> Particle for ([S; D], S) {
    type Array = [S; D];

    #[inline]
    fn position(&self) -> Self::Array {
        self.0.clone()
    }

    #[inline]
    fn mu(&self) -> <Self::Array as Array>::Item {
        self.1.clone()
    }
}
