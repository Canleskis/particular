use crate::{
    algorithms::{
        InternalVector, IntoInternalVector, IntoSIMDElement, PointMass, SIMDScalar, SIMDVector,
        Scalar,
    },
    compute_method::Storage,
};

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
    pub fn new_with(massive: Vec<PointMass<T, S>>, affected: Vec<PointMass<T, S>>) -> Self {
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
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
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
    V: IntoInternalVector<T::Array, InternalVector = T> + 'static,
{
    #[inline]
    fn new(input: impl Iterator<Item = PointMass<V, S>>) -> Self {
        let particles = input.map(PointMass::into_internal).collect();

        Self { particles }
    }
}

/// Storage for particles with massive and affected particles in two separate vectors.
///
/// To be used with [`SIMDVector`] and [`SIMDScalar`] types.
pub struct FromMassiveSIMD<const LANES: usize, T, S> {
    /// Particles used to compute the acceleration of the `affected` particles.
    pub massive: Vec<PointMass<T, S>>,
    /// Particles for which the acceleration is computed.
    ///
    /// This vector and the `massive` vector can share particles.
    pub affected: Vec<PointMass<T, S>>,
}

impl<const LANES: usize, T, S> FromMassiveSIMD<LANES, T, S> {
    /// Creates a new [`FromMassiveSIMD`] instance with the given vectors of particles.
    #[inline]
    pub fn new_with(massive: Vec<PointMass<T, S>>, affected: Vec<PointMass<T, S>>) -> Self {
        Self { massive, affected }
    }

    /// Creates a new [`FromMassiveSIMD`] instance from an iterator of particles.
    ///
    /// This method collects the particles from the iterator into the `affected` vector and copies the ones with mass to populate the `massive` vector.
    #[inline]
    pub fn from(particles: impl Iterator<Item = PointMass<T::Element, S::Element>>) -> Self
    where
        S: SIMDScalar<LANES>,
        T: SIMDVector<LANES, SIMDScalar = S>,
    {
        let buffer: Vec<_> = particles.collect();

        let affected = buffer
            .iter()
            .map(|p| PointMass::new(T::splat(p.position), S::splat(p.mass)))
            .collect();

        let massive = buffer
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect::<Vec<_>>()
            .chunks(LANES)
            .map(|slice| {
                let mut body_8 = [PointMass::default(); LANES];
                body_8[..slice.len()].copy_from_slice(slice);

                PointMass::new(
                    T::from_lanes(body_8.map(|p| p.position)),
                    S::from_lanes(body_8.map(|p| p.mass)),
                )
            })
            .collect::<Vec<_>>();

        Self { affected, massive }
    }

    /// Creates a new [`FromMassiveSIMD`] instance from an iterator of particles.
    ///
    /// This method collects the particles from the iterator into the `affected` vector and copies the ones with mass to populate the `massive` vector.
    #[inline]
    pub fn from1(particles: impl Iterator<Item = PointMass<T::Element, S::Element>>) -> Self
    where
        S: SIMDScalar<LANES>,
        T: SIMDVector<LANES, SIMDScalar = S>,
    {
        let buffer: Vec<_> = particles.collect();

        let massive = buffer
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect::<Vec<_>>()
            .chunks(LANES)
            .map(|slice| {
                let mut body_8 = [PointMass::default(); LANES];
                body_8[..slice.len()].copy_from_slice(slice);

                PointMass::new(
                    T::from_lanes(body_8.map(|p| p.position)),
                    S::from_lanes(body_8.map(|p| p.mass)),
                )
            })
            .collect::<Vec<_>>();

        let affected = buffer
            .iter()
            .map(|p| PointMass::new(T::splat(p.position), S::splat(p.mass)))
            .collect();

        Self { affected, massive }
    }
}

impl<const LANES: usize, T, S, V> Storage<PointMass<V, S::Element>> for FromMassiveSIMD<LANES, T, S>
where
    S: SIMDScalar<LANES>,
    T: SIMDVector<LANES, SIMDScalar = S>,
    V: IntoSIMDElement<T::Element, SIMDVector = T>,
{
    #[inline]
    fn new(input: impl Iterator<Item = PointMass<V, S::Element>>) -> Self {
        Self::from(input.map(|p| PointMass::new(p.position.into_simd_element(), p.mass)))
    }
}
