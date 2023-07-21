use crate::{
    algorithms::{
        internal, simd,
        tree::{BoundingBox, NodeID, SizedOrthant, SubDivide, Tree},
        vector, Zero,
    },
    compute_method::Storage,
};

/// Point-mass representation of an object in space.
#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct PointMass<V, S> {
    /// Position of the object.
    pub position: V,
    /// Mass of the object.
    pub mass: S,
}

impl<V: Zero, S: Zero> PointMass<V, S> {
    const ZERO: Self = PointMass::new(V::ZERO, S::ZERO);
}

impl<V, S> PointMass<V, S> {
    /// Creates a new [`PointMass`] with the given position and mass.
    pub const fn new(position: V, mass: S) -> Self {
        Self { position, mass }
    }

    /// Returns true if the mass is zero.
    #[inline]
    pub fn is_massless(&self) -> bool
    where
        S: Zero + PartialEq,
    {
        self.mass == S::ZERO
    }

    /// Returns false if the mass is zero.
    #[inline]
    pub fn is_massive(&self) -> bool
    where
        S: Zero + PartialEq,
    {
        self.mass != S::ZERO
    }
}

impl<V, S> PointMass<V, S> {
    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<T, S>`]
    /// provided `V` implements [`Into<T>`].
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

    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<V::Internal, S>`]
    /// provided `V` implements [`internal::IntoVectorArray`].
    #[inline]
    pub fn into_internal<const DIM: usize>(self) -> PointMass<V::Vector, S>
    where
        V: internal::ConvertInternalVector<DIM, S>,
    {
        PointMass::new(self.position.into_internal(), self.mass)
    }

    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<E, S>`]
    /// provided `V` implements [`simd::IntoVectorElement<E>`].
    #[inline]
    pub fn into_element<E>(self) -> PointMass<E, S>
    where
        V: simd::IntoVectorElement<E>,
    {
        PointMass::new(self.position.into_element(), self.mass)
    }
}

impl<V, S> PointMass<V, S> {
    /// Computes the gravitational acceleration exerted on this particle by the provided particle
    /// provided `S` and `V` implement [`internal::Scalar`] and [`internal::Vector`] respectively.
    #[inline]
    pub fn acceleration_internal(&self, particle: &Self) -> V
    where
        S: internal::Scalar,
        V: internal::Vector<Scalar = S>,
    {
        let dir = particle.position - self.position;
        let mag_2 = dir.length_squared();

        if mag_2 != S::ZERO {
            dir * particle.mass / (mag_2 * mag_2.sqrt())
        } else {
            dir
        }
    }

    /// Computes the total gravitational acceleration exerted on this particle by the provided slice of particles
    /// provided `S` and `V` implement [`internal::Scalar`] and [`internal::Vector`] respectively.
    #[inline]
    pub fn total_acceleration_internal(&self, particles: &[Self]) -> V
    where
        S: internal::Scalar,
        V: internal::Vector<Scalar = S>,
    {
        particles.iter().fold(V::ZERO, |acceleration, p2| {
            acceleration + self.acceleration_internal(p2)
        })
    }

    /// Computes the gravitational acceleration exerted on this particle by the provided particle
    /// provided `S` and `V` implement [`simd::Scalar`] and [`simd::Vector`] respectively.
    #[inline]
    pub fn acceleration_simd<const LANES: usize>(&self, particle: &Self) -> V
    where
        S: simd::Scalar<LANES>,
        V: simd::Vector<LANES, Scalar = S>,
    {
        let dir = particle.position - self.position;
        let mag_2 = dir.length_squared();
        let grav_acc = dir * particle.mass * (mag_2.recip_sqrt() * mag_2.recip());

        grav_acc.nan_to_zero()
    }

    /// Computes the total gravitational acceleration exerted on this particle by the provided slice of particles
    /// provided `S` and `V` implement [`simd::Scalar`] and [`simd::Vector`] respectively.
    #[inline]
    pub fn total_acceleration_simd<const LANES: usize>(&self, particles: &[Self]) -> V
    where
        S: simd::Scalar<LANES>,
        V: simd::Vector<LANES, Scalar = S>,
    {
        particles.iter().fold(V::ZERO, |acceleration, p2| {
            acceleration + self.acceleration_simd(p2)
        })
    }
}

/// Simple storage for particles using a vector.
#[derive(Debug, Default, Clone)]
pub struct ParticleSetInternal<const DIM: usize, S, V>(pub Vec<PointMass<V::Vector, S>>)
where
    V: internal::ConvertInternalVector<DIM, S>;

/// Storage for particles with massive and affected particles in two separate vectors.
#[derive(Debug, Default, Clone)]
pub struct MassiveAffected<T, S, TM = T, SM = S> {
    /// Particles used to compute the acceleration of the `affected` particles.
    pub massive: Vec<PointMass<TM, SM>>,
    /// Particles for which the acceleration is computed.
    ///
    /// This vector and the `massive` vector can share particles.
    pub affected: Vec<PointMass<T, S>>,
}

impl<T, S> MassiveAffected<T, S> {
    /// Creates a new [`MassiveAffected`] from the given vector of particles.
    ///
    /// This method populates the `affected` vector with the given particles and copies the ones with mass to the `massive` vector.
    pub fn from_affected(affected: Vec<PointMass<T, S>>) -> Self
    where
        S: Copy + Zero + PartialEq,
        T: Copy,
    {
        let massive = affected
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect();

        Self { massive, affected }
    }
}

/// Storage for particles with massive and affected particles in two separate vectors using arrays for the position.
#[derive(Debug, Default, Clone)]
pub struct MassiveAffectedArray<const DIM: usize, S, V>(pub MassiveAffected<V::Array, S>)
where
    V: vector::ConvertArray<DIM, S>;

/// Storage for particles with massive and affected particles in two separate vectors using [`internal::Vector`]s for the position.
#[derive(Debug, Default, Clone)]
pub struct MassiveAffectedInternal<const DIM: usize, S, V>(pub MassiveAffected<V::Vector, S>)
where
    V: internal::ConvertInternalVector<DIM, S>;

/// Storage with a [`Tree`] built from the massive particles and affected particles in another vector.
#[derive(Debug, Default, Clone)]
#[allow(clippy::type_complexity)]
pub struct TreeAffectedInternal<const N: usize, const DIM: usize, S, V>
where
    V: internal::ConvertInternalVector<DIM, S>,
{
    /// [`Tree`] built from the massive particles.
    pub tree: Tree<SizedOrthant<N, BoundingBox<[S; DIM]>>, PointMass<V::Vector, S>>,
    /// Root of the `tree`.
    pub root: Option<NodeID>,
    /// Particles for which the acceleration is computed.
    pub affected: Vec<PointMass<V::Vector, S>>,
}

impl<const N: usize, const DIM: usize, S, V, T> TreeAffectedInternal<N, DIM, S, V>
where
    V: internal::ConvertInternalVector<DIM, S, Vector = T>,
{
    /// Creates a new [`MassiveAffected`] from the given vector of particles.
    ///
    /// This method populates the `affected` vector with the given particles and copies the ones with mass to the `massive` vector.
    pub fn from_affected(affected: Vec<PointMass<T, S>>) -> Self
    where
        S: internal::Scalar,
        T: internal::Vector<Scalar = S> + Into<[S; DIM]>,
        BoundingBox<[S; DIM]>: SubDivide<Divison = [BoundingBox<[S; DIM]>; N]>,
    {
        let MassiveAffected { massive, affected } = MassiveAffected::from_affected(affected);

        let mut tree = Tree::new();
        let bbox = BoundingBox::square_with(massive.iter().map(|p| p.position.into()));
        let root = tree.build_node(&massive, bbox);

        Self {
            tree,
            root,
            affected,
        }
    }
}

/// Storage for particles with massive and affected particles in two separate vectors using [`simd::Vector`]s for the position and [`simd::Scalar`]s for the mass.
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
    /// Creates a new [`MassiveAffectedSIMD`] from the given vector of particles of [`simd::SIMD::Element`].
    ///
    /// This method populates the `affected` vector with the given particles and copies the ones with mass to the `massive` vector.
    pub fn from_affected(affected: Vec<PointMass<T::Element, S::Element>>) -> Self {
        let massive = affected
            .iter()
            .filter(|p| p.is_massive())
            .copied()
            .collect();

        Self::particles_to_simd(massive, affected)
    }

    /// Creates a new [`MassiveAffectedSIMD`] instance with the given vectors of particles of [`simd::SIMD::Element`].
    ///
    /// The given massive vector will be iterated over `LANES` elements at a time and mapped to particles of [`simd::SIMD`] values.
    #[inline]
    pub fn particles_to_simd(
        massive: Vec<PointMass<T::Element, S::Element>>,
        affected: Vec<PointMass<T::Element, S::Element>>,
    ) -> Self {
        let massive = massive
            .chunks(LANES)
            .map(|slice| {
                let mut body_8 = [PointMass::ZERO; LANES];
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

impl<const DIM: usize, S, V, T> Storage<PointMass<V, S>> for ParticleSetInternal<DIM, S, V>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::ConvertInternalVector<DIM, S, Vector = T>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        Self(input.map(PointMass::into_internal).collect())
    }
}

impl<const DIM: usize, S, V, A> Storage<PointMass<V, S>> for MassiveAffectedArray<DIM, S, V>
where
    S: internal::Scalar,
    A: Copy,
    V: vector::ConvertArray<DIM, S, Array = A>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        Self(MassiveAffected::from_affected(
            input.map(PointMass::into).collect(),
        ))
    }
}

impl<const DIM: usize, S, V, T> Storage<PointMass<V, S>> for MassiveAffectedInternal<DIM, S, V>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S>,
    V: internal::ConvertInternalVector<DIM, S, Vector = T>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        Self(MassiveAffected::from_affected(
            input.map(PointMass::into_internal).collect(),
        ))
    }
}

impl<const L: usize, T, S, V> Storage<PointMass<V, S::Element>> for MassiveAffectedSIMD<L, T, S>
where
    S: simd::Scalar<L>,
    T: simd::Vector<L, Scalar = S>,
    V: simd::IntoVectorElement<T::Element, Vector = T>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S::Element>>,
    {
        Self::from_affected(input.map(PointMass::into_element).collect())
    }
}

impl<const N: usize, const DIM: usize, S, V, T> Storage<PointMass<V, S>>
    for TreeAffectedInternal<N, DIM, S, V>
where
    S: internal::Scalar,
    T: internal::Vector<Scalar = S> + Into<[S; DIM]>,
    V: internal::ConvertInternalVector<DIM, S, Vector = T>,
    BoundingBox<[S; DIM]>: SubDivide<Divison = [BoundingBox<[S; DIM]>; N]>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        Self::from_affected(input.map(PointMass::into_internal).collect())
    }
}
