use crate::compute_method::{
    math::{Float, FloatVector, FromPrimitive, Sum, Zero},
    point_mass::{BarnesHutTree, PointMass},
    tree::{
        partition::{BoundingBox, SubDivide},
        NodeID,
    },
    ComputeMethod,
};

/// Flexible, copyable storage with references to affected particles and a generic massive storage.
#[derive(Debug)]
pub struct ParticleSystem<'p, V, S, T: ?Sized> {
    /// Particles for which the acceleration is computed.
    pub affected: &'p [PointMass<V, S>],
    /// Particle storage responsible for the acceleration exerted on the `affected` particles.
    pub massive: &'p T,
}

impl<V, S, T: ?Sized> Clone for ParticleSystem<'_, V, S, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V, S, T: ?Sized> Copy for ParticleSystem<'_, V, S, T> {}

impl<'p, V, S, T: ?Sized> ParticleSystem<'p, V, S, T> {
    /// Creates a new [`ParticleSystem`] with the given slice of particles and massive storage.
    #[inline]
    pub const fn with(affected: &'p [PointMass<V, S>], massive: &'p T) -> Self {
        Self { affected, massive }
    }
}

/// [`ParticleSystem`] with a slice of particles for the massive storage.
pub type ParticleSliceSystem<'p, V, S> = ParticleSystem<'p, V, S, [PointMass<V, S>]>;

/// Storage with particles in a [`BarnesHutTree`].
#[derive(Clone, Debug)]
pub struct ParticleTree<const X: usize, const D: usize, V, S> {
    root: Option<NodeID>,
    tree: BarnesHutTree<X, D, V, S>,
}

impl<const X: usize, const D: usize, V, S> ParticleTree<X, D, V, S> {
    /// Returns the root of the [`BarnesHutTree`].
    #[inline]
    pub const fn root(&self) -> Option<NodeID> {
        self.root
    }

    /// Returns a reference to the [`BarnesHutTree`].
    #[inline]
    pub const fn tree(&self) -> &BarnesHutTree<X, D, V, S> {
        &self.tree
    }
}

impl<const X: usize, const D: usize, V, S> From<&[PointMass<V, S>]> for ParticleTree<X, D, V, S>
where
    V: Copy + FloatVector<Float = S, Array = [S; D]>,
    S: Copy + Float + Sum + PartialOrd + FromPrimitive<usize>,
    BoundingBox<[S; D]>: SubDivide<Division = [BoundingBox<[S; D]>; X]>,
{
    #[inline]
    fn from(slice: &[PointMass<V, S>]) -> Self {
        let mut tree = BarnesHutTree::with_capacity(slice.len());
        let root = tree.build_node(slice, |p| p.position.into(), PointMass::new_com);

        Self { root, tree }
    }
}

/// [`ParticleSystem`] with a [`ParticleTree`] for the massive storage.
pub type ParticleTreeSystem<'p, const X: usize, const D: usize, V, S> =
    ParticleSystem<'p, V, S, ParticleTree<X, D, V, S>>;

/// Storage inside of which the massive particles are placed before the massless ones.
///
/// Allows for easy optimisation of the computation of forces between massive and massless particles.
#[derive(Clone, Debug)]
pub struct ParticleOrdered<V, S> {
    massive_len: usize,
    particles: Vec<PointMass<V, S>>,
}

impl<V, S> ParticleOrdered<V, S> {
    /// Returns the number of stored massive particles.
    #[inline]
    pub const fn massive_len(&self) -> usize {
        self.massive_len
    }

    /// Returns a reference to the massive particles.
    #[inline]
    pub fn massive(&self) -> &[PointMass<V, S>] {
        &self.particles[..self.massive_len]
    }

    /// Returns a reference to the massless particles.
    #[inline]
    pub fn massless(&self) -> &[PointMass<V, S>] {
        &self.particles[self.massive_len..]
    }

    /// Returns a reference to the particles.
    #[inline]
    pub fn particles(&self) -> &[PointMass<V, S>] {
        &self.particles
    }

    /// Returns a mutable reference to the massive particles.
    #[inline]
    pub fn massive_mut(&mut self) -> &mut [PointMass<V, S>] {
        &mut self.particles[..self.massive_len]
    }

    /// Returns a mutable reference to the massless particles.
    #[inline]
    pub fn massless_mut(&mut self) -> &mut [PointMass<V, S>] {
        &mut self.particles[self.massive_len..]
    }

    /// Returns a mutable reference to the stored ordered particles.
    #[inline]
    pub fn particles_mut(&mut self) -> &mut [PointMass<V, S>] {
        &mut self.particles
    }
}

impl<V, S> From<&[PointMass<V, S>]> for ParticleOrdered<V, S>
where
    V: Clone,
    S: Clone + PartialEq + Zero,
{
    #[inline]
    fn from(particles: &[PointMass<V, S>]) -> Self {
        let particles: Vec<_> = particles
            .iter()
            .filter(|p| p.is_massive())
            .chain(particles.iter().filter(|p| p.is_massless()))
            .cloned()
            .collect();

        let massive_len = particles
            .iter()
            .position(PointMass::is_massless)
            .unwrap_or(particles.len());

        Self {
            massive_len,
            particles,
        }
    }
}

/// Storage for particles which has a copy of the stored particles inside a [`ParticleOrdered`].
#[derive(Clone, Debug)]
pub struct ParticleReordered<'p, V, S> {
    /// Original, unordered particles.
    pub unordered: &'p [PointMass<V, S>],
    ordered: ParticleOrdered<V, S>,
}

impl<V, S> ParticleReordered<'_, V, S> {
    /// Returns a reference to the [`ParticleOrdered`].
    #[inline]
    pub const fn ordered(&self) -> &ParticleOrdered<V, S> {
        &self.ordered
    }

    /// Returns the number of stored massive particles.
    #[inline]
    pub const fn massive_len(&self) -> usize {
        self.ordered.massive_len()
    }

    /// Returns a reference to the massive particles.
    #[inline]
    pub fn massive(&self) -> &[PointMass<V, S>] {
        self.ordered.massive()
    }

    /// Returns a reference to the massless particles.
    #[inline]
    pub fn massless(&self) -> &[PointMass<V, S>] {
        self.ordered.massless()
    }

    /// Returns a reference to the stored ordered particles.
    #[inline]
    pub fn reordered(&self) -> &[PointMass<V, S>] {
        self.ordered.particles()
    }
}

impl<'p, V, S> From<&'p [PointMass<V, S>]> for ParticleReordered<'p, V, S>
where
    V: Clone,
    S: Clone + Zero + PartialEq,
{
    #[inline]
    fn from(affected: &'p [PointMass<V, S>]) -> Self {
        Self {
            unordered: affected,
            ordered: ParticleOrdered::from(affected),
        }
    }
}

impl<const X: usize, const D: usize, V, S, C, O> ComputeMethod<ParticleSliceSystem<'_, V, S>> for C
where
    O: IntoIterator,
    for<'a> C: ComputeMethod<ParticleTreeSystem<'a, X, D, V, S>, Output = O>,
    V: Copy + FloatVector<Float = S, Array = [S; D]>,
    S: Copy + Float + Sum + PartialOrd + FromPrimitive<usize>,
    BoundingBox<[S; D]>: SubDivide<Division = [BoundingBox<[S; D]>; X]>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<V, S>) -> Self::Output {
        self.compute(ParticleTreeSystem {
            affected: system.affected,
            massive: &ParticleTree::from(system.massive),
        })
    }
}

impl<V, S, C, O> ComputeMethod<&[PointMass<V, S>]> for C
where
    O: IntoIterator,
    for<'a> C: ComputeMethod<ParticleSliceSystem<'a, V, S>, Output = O>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, slice: &[PointMass<V, S>]) -> Self::Output {
        self.compute(ParticleSliceSystem {
            affected: slice,
            massive: slice,
        })
    }
}

impl<V, S, C, O> ComputeMethod<&ParticleOrdered<V, S>> for C
where
    O: IntoIterator,
    for<'a> C: ComputeMethod<ParticleSliceSystem<'a, V, S>, Output = O>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, ordered: &ParticleOrdered<V, S>) -> Self::Output {
        self.compute(ParticleSliceSystem {
            affected: ordered.particles(),
            massive: ordered.massive(),
        })
    }
}

impl<V, S, C, O> ComputeMethod<ParticleReordered<'_, V, S>> for C
where
    O: IntoIterator,
    for<'a> C: ComputeMethod<ParticleSliceSystem<'a, V, S>, Output = O>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, reordered: ParticleReordered<V, S>) -> Self::Output {
        self.compute(ParticleSliceSystem {
            affected: reordered.unordered,
            massive: reordered.massive(),
        })
    }
}
