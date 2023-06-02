use crate::{
    algorithms::{
        InternalVector, IntoInternalVector, IntoSIMDElement, SIMDScalar, SIMDVector, Scalar, SIMD,
    },
    compute_method::Storage,
};

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
    pub const fn new(position: V, mass: S) -> Self {
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
    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<V::Internal, S>`] provided `V` implements [`IntoInternalVector`].
    #[inline]
    pub fn into_internal<A>(self) -> PointMass<V::InternalVector, S>
    where
        V: IntoInternalVector<A>,
    {
        PointMass::new(self.position.into_internal(), self.mass)
    }

    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<E, S>`] provided `V` implements [`IntoSIMDElement<E>`].
    #[inline]
    pub fn into_simd_element<E>(self) -> PointMass<E, S>
    where
        V: IntoSIMDElement<E>,
    {
        PointMass::new(self.position.into_simd_element(), self.mass)
    }
}

/// Simple storage for particles using a vector.
#[derive(Debug, Default, Clone)]
pub struct ParticleSet<T, S> {
    /// Particles for which the acceleration is computed.
    pub particles: Vec<PointMass<T, S>>,
}

impl<T, S> ParticleSet<T, S> {
    /// Creates a new [`ParticleSet`] instance with the given vector of particles.
    #[inline]
    pub fn new(particles: Vec<PointMass<T, S>>) -> Self {
        Self { particles }
    }
}

/// Storage for particles with massive and affected particles in two separate vectors.
#[derive(Debug, Default, Clone)]
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
}

/// Storage for particles with massive and affected particles in two separate vectors.
///
/// To be used with [`SIMDVector`] and [`SIMDScalar`] types.
#[derive(Debug, Default, Clone)]
pub struct FromMassiveSIMD<const LANES: usize, T, S>
where
    T: SIMD<LANES>,
    S: SIMD<LANES>,
{
    /// Particles used to compute the acceleration of the `affected` particles.
    pub massive: Vec<PointMass<T, S>>,
    /// Particles for which the acceleration is computed.
    ///
    /// This vector and the `massive` vector can share particles.
    pub affected: Vec<PointMass<T::Element, S::Element>>,
}

impl<const LANES: usize, T, S> FromMassiveSIMD<LANES, T, S>
where
    T: SIMD<LANES>,
    S: SIMD<LANES>,
{
    /// Creates a new [`FromMassiveSIMD`] instance with the given vectors of particles.
    #[inline]
    pub fn new(
        massive: Vec<PointMass<T::Element, S::Element>>,
        affected: Vec<PointMass<T::Element, S::Element>>,
    ) -> Self {
        let massive = massive
            .chunks(LANES)
            .map(|slice| {
                let mut body_8 = [PointMass::default(); LANES];
                body_8[..slice.len()].copy_from_slice(slice);

                PointMass::new(
                    T::from_lanes(body_8.map(|p| p.position)),
                    S::from_lanes(body_8.map(|p| p.mass)),
                )
            })
            .collect();

        Self { massive, affected }
    }

    /// Creates a new [`FromMassiveSIMD`] instance from an iterator of particles.
    ///
    /// This method collects the particles from the iterator into the `affected` vector and copies the ones with mass to populate the `massive` vector.
    #[inline]
    pub fn from(particles: impl Iterator<Item = PointMass<T::Element, S::Element>>) -> Self {
        let affected: Vec<_> = particles.collect();
        let massive = affected
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect();

        Self::new(massive, affected)
    }
}

impl<I, T, S> From<I> for ParticleSet<T, S>
where
    T: Copy,
    S: Default + PartialEq + Copy,
    I: Iterator<Item = PointMass<T, S>>,
{
    /// Creates a new [`ParticleSet`] instance from an iterator of particles.
    #[inline]
    fn from(particles: I) -> Self {
        Self::new(particles.collect())
    }
}

impl<I, T, S> From<I> for FromMassive<T, S>
where
    T: Copy,
    S: Default + PartialEq + Copy,
    I: Iterator<Item = PointMass<T, S>>,
{
    /// Creates a new [`FromMassive`] instance from an iterator of particles.
    ///
    /// This method collects the particles from the iterator into the `affected` vector and copies the ones with mass to populate the `massive` vector.
    #[inline]
    fn from(particles: I) -> Self {
        let affected: Vec<_> = particles.collect();
        let massive = affected
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect();

        Self::new(massive, affected)
    }
}

impl<I, const LANES: usize, T, S> From<I> for FromMassiveSIMD<LANES, T, S>
where
    T: SIMD<LANES>,
    S: SIMD<LANES>,
    I: Iterator<Item = PointMass<T::Element, S::Element>>,
{
    /// Creates a new [`FromMassiveSIMD`] instance from an iterator of particles.
    ///
    /// This method collects the particles from the iterator into the `affected` vector and copies the ones with mass to populate the `massive` vector.
    #[inline]
    fn from(particles: I) -> Self {
        let affected: Vec<_> = particles.collect();
        let massive = affected
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect();

        Self::new(massive, affected)
    }
}

impl<T, S, V> Storage<PointMass<V, S>> for ParticleSet<T, S>
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
{
    #[inline]
    fn store(input: impl Iterator<Item = PointMass<V, S>>) -> Self {
        Self::from(input.map(PointMass::into_internal))
    }
}

impl<T, S, V> Storage<PointMass<V, S>> for FromMassive<T, S>
where
    S: Scalar + 'static,
    T: InternalVector<Scalar = S> + 'static,
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
{
    #[inline]
    fn store(input: impl Iterator<Item = PointMass<V, S>>) -> Self {
        Self::from(input.map(PointMass::into_internal))
    }
}

impl<const LANES: usize, T, S, V> Storage<PointMass<V, S::Element>> for FromMassiveSIMD<LANES, T, S>
where
    S: SIMDScalar<LANES>,
    T: SIMDVector<LANES, SIMDScalar = S>,
    V: IntoSIMDElement<T::Element, SIMDVector = T>,
{
    #[inline]
    fn store(input: impl Iterator<Item = PointMass<V, S::Element>>) -> Self {
        Self::from(input.map(PointMass::into_simd_element))
    }
}
