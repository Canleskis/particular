use crate::{
    algorithms::{
        internal, simd,
        tree::{BarnesHutTree, BoundingBox, NodeID, SubDivide},
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
        PointMass::new(self.position.into(), self.mass)
    }

    /// Converts from a [`PointMass<V, S>`] to a [`PointMass<V::Internal, S>`]
    /// provided `V` implements [`internal::IntoVectorArray`].
    #[inline]
    pub fn into_internal<const D: usize>(self) -> PointMass<V::Vector, S>
    where
        V: internal::ConvertInternal<D, S>,
    {
        PointMass::new(self.position.into_internal(), self.mass)
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
    pub fn acceleration_simd<const L: usize>(&self, particle: &Self) -> V
    where
        S: simd::Scalar<L>,
        V: simd::Vector<L, Scalar = S>,
    {
        let dir = particle.position - self.position;
        let mag_2 = dir.length_squared();
        let grav_acc = dir * particle.mass * (mag_2.recip_sqrt() * mag_2.recip());

        grav_acc.nan_to_zero()
    }

    /// Computes the total gravitational acceleration exerted on this particle by the provided slice of particles
    /// provided `S` and `V` implement [`simd::Scalar`] and [`simd::Vector`] respectively.
    #[inline]
    pub fn total_acceleration_simd<const L: usize>(&self, particles: &[Self]) -> V
    where
        S: simd::Scalar<L>,
        V: simd::Vector<L, Scalar = S>,
    {
        particles.iter().fold(V::ZERO, |acceleration, p2| {
            acceleration + self.acceleration_simd(p2)
        })
    }
}

/// Simple storage for particles using a vector.
#[derive(Debug, Default, Clone)]
pub struct ParticleSetInternal<const D: usize, S, V>(pub Vec<PointMass<V::Vector, S>>)
where
    V: internal::ConvertInternal<D, S>;

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
pub struct MassiveAffectedArray<const D: usize, S, V>(pub MassiveAffected<V::Array, S>)
where
    V: vector::ConvertArray<D, S>;

/// Storage for particles with massive and affected particles in two separate vectors using [`internal::Vector`]s for the position.
#[derive(Debug, Default, Clone)]
pub struct MassiveAffectedInternal<const D: usize, S, V>(pub MassiveAffected<V::Vector, S>)
where
    V: internal::ConvertInternal<D, S>;

/// Storage for particles with massive and affected particles in two separate vectors.
///
/// Particles are stored using [`simd::Vector`] and [`simd::Scalar`] for the massive ones and arrays for the massless ones.
#[derive(Clone)]
pub struct MassiveAffectedSIMD<const L: usize, const D: usize, S, V>(
    pub MassiveAffected<V::Array, S, V::Vector, V::Scalar>,
)
where
    V: simd::ConvertSIMD<L, D, S>;

/// Storage with a [`Tree`] built from the massive particles and affected particles in another vector.
#[derive(Debug, Default, Clone)]
pub struct TreeAffectedInternal<const N: usize, const D: usize, S, V>
where
    V: internal::ConvertInternal<D, S>,
{
    /// [`Tree`] built from the massive particles.
    pub tree: BarnesHutTree<N, D, V::Vector, S>,
    /// Root of the `tree`.
    pub root: Option<NodeID>,
    /// Particles for which the acceleration is computed.
    pub affected: Vec<PointMass<V::Vector, S>>,
}

impl<const D: usize, S, V> Storage<PointMass<V, S>> for ParticleSetInternal<D, S, V>
where
    V: internal::ConvertInternal<D, S>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        Self(input.map(PointMass::into_internal).collect())
    }
}

impl<const D: usize, S, V> Storage<PointMass<V, S>> for MassiveAffectedArray<D, S, V>
where
    S: Copy + Zero + PartialEq,
    V: vector::ConvertArray<D, S, Array = [S; D]>,
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

impl<const D: usize, S, V> Storage<PointMass<V, S>> for MassiveAffectedInternal<D, S, V>
where
    S: Copy + Zero + PartialEq,
    V: internal::ConvertInternal<D, S>,
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

impl<const L: usize, const D: usize, S, V> Storage<PointMass<V, S>>
    for MassiveAffectedSIMD<L, D, S, V>
where
    S: Copy + Zero + PartialEq,
    V: simd::ConvertSIMD<L, D, S>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        let affected = input.map(PointMass::into).collect();
        let MassiveAffected { massive, affected } = MassiveAffected::from_affected(affected);

        let massive = massive
            .chunks(L)
            .map(|slice| {
                let mut body_8 = [PointMass::ZERO; L];
                body_8[..slice.len()].copy_from_slice(slice);
                PointMass::new(
                    simd::SIMD::from_lanes(body_8.map(|p| p.position)),
                    simd::SIMD::from_lanes(body_8.map(|p| p.mass)),
                )
            })
            .collect();

        Self(MassiveAffected { massive, affected })
    }
}

impl<const N: usize, const D: usize, S, V> Storage<PointMass<V, S>>
    for TreeAffectedInternal<N, D, S, V>
where
    S: internal::Scalar,
    V: internal::ConvertInternal<D, S>,
    BoundingBox<V::Array>: SubDivide<Divison = [BoundingBox<V::Array>; N]>,
{
    #[inline]
    fn store<I>(input: I) -> Self
    where
        I: Iterator<Item = PointMass<V, S>>,
    {
        let affected = input.map(PointMass::into_internal).collect();
        let MassiveAffected { massive, affected } = MassiveAffected::from_affected(affected);

        let mut tree = BarnesHutTree::new();
        let bbox = BoundingBox::square_with(massive.iter().map(|p| p.position.into()));
        let root = tree.build_node(&massive, bbox);

        Self {
            tree,
            root,
            affected,
        }
    }
}
