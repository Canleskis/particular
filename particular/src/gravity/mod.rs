mod impls;

/// Newtonian gravity implementations.
pub mod newtonian;

use std::ops::Sub;

#[cfg(feature = "glam")]
pub use impls::glam::wide_glam;

/// Representation of a gravitational field.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
#[allow(missing_docs)]
pub struct GravitationalField<V, S> {
    pub position: V,
    pub m: S,
}

impl<V, S> GravitationalField<V, S> {
    /// Creates a new [`GravitationalField`] with the given position and strength.
    #[inline]
    pub const fn new(position: V, m: S) -> Self {
        Self { position, m }
    }

    /// Returns `true` if this gravitational field has a non-zero strength.
    #[inline]
    pub fn is_affecting(&self) -> bool
    where
        S: Default + PartialEq,
    {
        self.m != S::default()
    }
}

/// Trait for types that can be located in space.
///
/// You can derive this trait if your type has a field named `position`.
pub trait Position {
    /// The type used to represent the position.
    type Vector;

    /// Returns the position of a particle.
    fn position(&self) -> Self::Vector;
}

impl<V, T> Position for GravitationalField<V, T>
where
    V: Clone,
{
    type Vector = V;

    #[inline]
    fn position(&self) -> Self::Vector {
        self.position.clone()
    }
}

/// Trait for an object that defines a mass. Because the standard gravitational parameter is defined
/// as `µ = gravitational constant * mass`, this trait also provides a method to retrieve it.
///
/// Implementing this trait for a type will allow it to be used to create a [`GravitationalField`]
/// to compute gravitational accelerations. The mass can be used to compute gravitational forces
/// under the influence of one or multiple [`GravitationalField`]s by multiplying it by the
/// acceleration.
///
/// # Example
///
/// ## Manual implementation
///
/// ```
/// use particular::prelude::*;
/// use glam::DVec3;
///
/// const G: f64 = 6.67430e-11;
///
/// #[derive(Position)]
/// struct Planet {
///     position: DVec3,
///     mass: f64,
/// }
///
/// impl Mass for Planet {
///     type Scalar = f64;
///
///     fn mass(&self) -> Self::Scalar {
///         self.mass
///     }
///
///     fn mu(&self) -> Self::Scalar {
///         G * self.mass
///     }
/// }
///
/// let earth = Planet {
///     position: DVec3::new(1.496e+8, 0.0, 0.0),
///     mass: 5.97e+24,
/// };
///
/// let field = GravitationalField::from(&earth);
/// assert_eq!(earth.mu(), field.m);
/// assert_eq!(earth.mass(), field.m / G);
/// ```
///
/// ## Deriving
///
/// You can also derive this trait for your struct. The derive macro requires the struct to have a
/// field named either `mu` or `mass`. An optional attribute `#[G = ...]` defining the gravitational
/// constant can also be added on the field. If the attribute is missing, the value of the
/// gravitational constant is `1.0`.
///
/// ```
/// use particular::prelude::*;
/// use glam::DVec3;
///
/// #[derive(Position, Mass)]
/// struct Planet {
///     position: DVec3,
///     #[G = 6.67430e-11]
///     mass: f64,
/// }
///
/// let mars = Planet {
///     position: DVec3::new(2.28e+8, 0.0, 0.0),
///     mass: 6.42e23,
/// };
///
/// let field = GravitationalField::from(&mars);
/// assert_eq!(mars.mu(), field.m);
/// assert_eq!(mars.mass(), field.m / 6.67430e-11);
/// ```
pub trait Mass {
    /// The scalar type used to represent the mass or gravitational parameter of the particle.
    type Scalar;

    /// Returns the mass of the particle.
    ///
    /// `mass = µ / gravitational constant`.
    fn mass(&self) -> Self::Scalar;

    /// Returns the [standard gravitational parameter] of the particle, annotated `µ` (mu).
    ///
    /// `µ = gravitational constant * mass`.
    ///
    /// [standard gravitational parameter]: https://en.wikipedia.org/wiki/Standard_gravitational_parameter
    fn mu(&self) -> Self::Scalar;
}

impl<P> From<&P> for GravitationalField<P::Vector, P::Scalar>
where
    P: Position + Mass,
{
    #[inline]
    fn from(particle: &P) -> Self {
        Self {
            position: particle.position(),
            m: particle.mu(),
        }
    }
}

impl<V, S> Position for (V, S)
where
    V: Clone,
{
    type Vector = V;

    #[inline]
    fn position(&self) -> Self::Vector {
        self.0.clone()
    }
}

impl<V, S> Mass for (V, S)
where
    V: Clone,
    S: Clone,
{
    type Scalar = S;

    #[inline]
    fn mass(&self) -> Self::Scalar {
        self.1.clone()
    }

    #[inline]
    fn mu(&self) -> Self::Scalar {
        self.1.clone()
    }
}

/// Trait for computing the norm of a vector.
pub trait Norm {
    /// The type of the norm.
    type Output;

    /// Computes the squared norm of the vector.
    fn norm_squared(self) -> <Self as Norm>::Output;
}

/// Trait for computing the Euclidean distance between two vectors.
pub trait Distance: Norm + Sub<Output = Self> + Sized {
    /// Returns the Euclidean distance between this vector and the given vector.
    #[inline]
    fn distance_squared(self, other: Self) -> <Self as Norm>::Output {
        (self - other).norm_squared()
    }
}

impl<V: Norm + Sub<Output = V> + Sized> Distance for V {}

/// Trait for vectors that can be converted into an array.
pub trait IntoArray: Into<Self::Array> {
    /// The array the vector can be converted into.
    type Array;
}

impl<V: IntoArray + Clone> Position for V {
    type Vector = V;

    #[inline]
    fn position(&self) -> Self::Vector {
        self.clone()
    }
}

/// Reduce the lanes of a SIMD vector.
pub trait Reduce {
    /// The scalar vector after reducing the lanes.
    type Output;

    /// Sums the lanes of the SIMD vector.
    fn reduce_sum(self) -> Self::Output;
}

/// Types with added padding for alignment.
#[cfg(feature = "gpu")]
pub mod padded {
    use bytemuck::{Pod, Zeroable};

    /// A [`GravitationalField`](crate::gravity::GravitationalField) with padding. Used to align its
    /// before sending it to the GPU.
    #[repr(C)]
    #[allow(missing_docs)]
    #[derive(Clone, Copy)]
    pub struct GravitationalField<const BYTES: usize, V, S> {
        pub position: V,
        pub mu: S,
        _padding: [u8; BYTES],
    }

    unsafe impl<const BYTES: usize, V: Zeroable, S: Zeroable> Zeroable
        for GravitationalField<BYTES, V, S>
    {
    }
    unsafe impl<const BYTES: usize, V: Pod, S: Pod> Pod for GravitationalField<BYTES, V, S> {}

    impl<const BYTES: usize, V, T> From<crate::gravity::GravitationalField<V, T>>
        for GravitationalField<BYTES, V, T>
    where
        [u8; BYTES]: Zeroable,
    {
        #[inline]
        fn from(p: crate::gravity::GravitationalField<V, T>) -> Self {
            Self {
                position: p.position,
                mu: p.m,
                _padding: Zeroable::zeroed(),
            }
        }
    }

    /// A vector with padding. Used to align the size of a vector before sending it to the GPU.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Vector<const BYTES: usize, V> {
        /// The vector.
        pub vector: V,
        _padding: [u8; BYTES],
    }

    unsafe impl<const BYTES: usize, V: Zeroable> Zeroable for Vector<BYTES, V> {}
    unsafe impl<const BYTES: usize, V: Pod> Pod for Vector<BYTES, V> {}

    impl<const BYTES: usize, V> From<V> for Vector<BYTES, V>
    where
        [u8; BYTES]: Zeroable,
    {
        #[inline]
        fn from(v: V) -> Self {
            Self {
                vector: v,
                _padding: Zeroable::zeroed(),
            }
        }
    }
}
