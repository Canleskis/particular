use crate::{
    algorithms::{internal, simd},
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
    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<V::Internal, S>`] provided `V` implements [`internal::IntoVectorArray`].
    #[inline]
    pub fn into_internal<A>(self) -> PointMass<V::Vector, S>
    where
        V: internal::IntoVectorArray<A>,
    {
        PointMass::new(self.position.into_internal(), self.mass)
    }

    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<E, S>`] provided `V` implements [`IntoSIMDElement<E>`].
    #[inline]
    pub fn into_simd_element<E>(self) -> PointMass<E, S>
    where
        V: simd::IntoVectorElement<E>,
    {
        PointMass::new(self.position.into_simd_element(), self.mass)
    }
}

/// Simple storage for particles using a vector.
#[derive(Debug, Default, Clone)]
pub struct ParticleSet<T, S>(pub Vec<PointMass<T, S>>);

impl<T, S> ParticleSet<T, S> {
    /// Creates a new [`ParticleSet`] instance with the given vector of particles.
    #[inline]
    pub fn new(particles: Vec<PointMass<T, S>>) -> Self {
        Self(particles)
    }
}

/// Storage for particles with massive and affected particles in two separate vectors.
#[derive(Debug, Default, Clone)]
pub struct MassiveAffected<T, S> {
    /// Particles used to compute the acceleration of the `affected` particles.
    pub massive: Vec<PointMass<T, S>>,
    /// Particles for which the acceleration is computed.
    ///
    /// This vector and the `massive` vector can share particles.
    pub affected: Vec<PointMass<T, S>>,
}

impl<T, S> MassiveAffected<T, S> {
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
pub struct MassiveAffectedSIMD<const LANES: usize, T, S>
where
    T: simd::SIMD<LANES>,
    S: simd::SIMD<LANES>,
{
    /// Particles used to compute the acceleration of the `affected` particles.
    pub massive: Vec<PointMass<T, S>>,
    /// Particles for which the acceleration is computed.
    ///
    /// This vector and the `massive` vector can share particles.
    pub affected: Vec<PointMass<T::Element, S::Element>>,
}

impl<const LANES: usize, T, S> MassiveAffectedSIMD<LANES, T, S>
where
    T: simd::SIMD<LANES>,
    S: simd::SIMD<LANES>,
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

impl<I, T, S> From<I> for MassiveAffected<T, S>
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

impl<I, const LANES: usize, T, S> From<I> for MassiveAffectedSIMD<LANES, T, S>
where
    T: simd::SIMD<LANES>,
    S: simd::SIMD<LANES>,
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
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::IntoVectorArray<T::Array, Vector = T>,
{
    #[inline]
    fn store<I: Iterator<Item = PointMass<V, S>>>(input: I) -> Self {
        Self::from(input.map(PointMass::into_internal))
    }
}

impl<T, S, V> Storage<PointMass<V, S>> for MassiveAffected<T, S>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::IntoVectorArray<T::Array, Vector = T>,
{
    #[inline]
    fn store<I: Iterator<Item = PointMass<V, S>>>(input: I) -> Self {
        Self::from(input.map(PointMass::into_internal))
    }
}

impl<const LANES: usize, T, S, V> Storage<PointMass<V, S::Element>>
    for MassiveAffectedSIMD<LANES, T, S>
where
    S: simd::Scalar<LANES>,
    T: simd::Vector<LANES, Scalar = S>,
    V: simd::IntoVectorElement<T::Element, Vector = T>,
{
    #[inline]
    fn store<I: Iterator<Item = PointMass<V, S::Element>>>(input: I) -> Self {
        Self::from(input.map(PointMass::into_simd_element))
    }
}
