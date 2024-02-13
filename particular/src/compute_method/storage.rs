use crate::compute_method::{
    math::{AsPrimitive, BitAnd, CmpNe, Float, FloatVector, FromPrimitive, Sum, Zero, SIMD},
    tree::{
        partition::{BoundingBox, SubDivide},
        Node, NodeID, Orthtree,
    },
    ComputeMethod,
};

/// Point-mass representation of an object in space.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct PointMass<V, S> {
    /// Position of the object.
    pub position: V,
    /// Mass of the object.
    pub mass: S,
}

impl<V: Zero, S: Zero> PointMass<V, S> {
    /// [`PointMass`] with position and mass set to [`Zero::ZERO`].
    pub const ZERO: Self = PointMass::new(V::ZERO, S::ZERO);
}

impl<V, S> PointMass<V, S> {
    /// Creates a new [`PointMass`] with the given position and mass.
    #[inline]
    pub const fn new(position: V, mass: S) -> Self {
        Self { position, mass }
    }

    /// Creates a new [`PointMass`] with the given lanes of positions and masses.
    #[inline]
    pub fn new_lane(position: V::Lane, mass: S::Lane) -> Self
    where
        V: SIMD,
        S: SIMD,
    {
        Self::new(V::new_lane(position), S::new_lane(mass))
    }

    /// Returns the [`PointMass`] corresponding to the center of mass and total mass of the given
    /// slice of point-masses.
    #[inline]
    pub fn new_com(data: &[Self]) -> Self
    where
        V: FloatVector<Float = S> + Sum + Copy,
        S: Float + FromPrimitive<usize> + Sum + Copy,
    {
        let tot = data.iter().map(|p| p.mass).sum();
        let com = if tot == S::ZERO {
            data.iter().map(|p| p.position).sum::<V>() / data.len().as_()
        } else {
            data.iter().map(|p| p.position * (p.mass / tot)).sum()
        };

        Self::new(com, tot)
    }

    /// Creates a new [`PointMass`] with all lanes set to the given position and mass.
    #[inline]
    pub fn splat_lane(position: V::Element, mass: S::Element) -> Self
    where
        V: SIMD,
        S: SIMD,
    {
        Self::new(V::splat(position), S::splat(mass))
    }

    /// Returns a [`SIMD`] point-masses from a slice of [`SIMD::Element`] point-masses.
    #[inline]
    pub fn slice_to_lane<const L: usize, T, E>(slice: &[PointMass<T, E>]) -> Self
    where
        T: Clone + Zero,
        E: Clone + Zero,
        V: SIMD<Lane = [T; L], Element = T>,
        S: SIMD<Lane = [E; L], Element = E>,
    {
        let mut lane = [PointMass::ZERO; L];
        lane[..slice.len()].clone_from_slice(slice);
        Self::new_lane(lane.clone().map(|p| p.position), lane.map(|p| p.mass))
    }

    /// Returns an iterator of [`SIMD`] point-masses from a slice of [`SIMD::Element`] point-masses.
    #[inline]
    pub fn slice_to_lanes<'a, const L: usize, T, E>(
        slice: &'a [PointMass<T, E>],
    ) -> impl Iterator<Item = Self> + 'a
    where
        T: Clone + Zero,
        E: Clone + Zero,
        V: SIMD<Lane = [T; L], Element = T> + 'a,
        S: SIMD<Lane = [E; L], Element = E> + 'a,
    {
        slice.chunks(L).map(Self::slice_to_lane)
    }

    /// Returns true if the mass is zero.
    #[inline]
    pub fn is_massless(&self) -> bool
    where
        S: PartialEq + Zero,
    {
        self.mass == S::ZERO
    }

    /// Returns false if the mass is zero.
    #[inline]
    pub fn is_massive(&self) -> bool
    where
        S: PartialEq + Zero,
    {
        self.mass != S::ZERO
    }

    /// Computes the gravitational force exerted on the current point-mass using the given position
    /// and mass. This method is optimised in the case where `V` and `S` are scalar types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the given
    /// position, this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn force_scalar<const CHECK_ZERO: bool>(&self, position: V, mass: S, softening: S) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + Copy,
    {
        let dir = position - self.position;
        let norm = dir.norm_squared();

        let force = |norm_s: S| {
            // Branch removed by the compiler when `CHECK_ZERO` is false.
            if CHECK_ZERO && norm == S::ZERO {
                dir
            } else {
                dir * mass / (norm_s * norm_s.sqrt())
            }
        };

        // Using this branch results in better performance when softening is zero.
        if softening == S::ZERO {
            force(norm)
        } else {
            force(norm + (softening * softening))
        }
    }

    /// Computes the gravitational force exerted on the current point-mass using the given position
    /// and mass. This method is optimised in the case where `V` and `S` are simd types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the given
    /// position, this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn force_simd<const CHECK_ZERO: bool>(&self, position: V, mass: S, softening: S) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + BitAnd<Output = S> + CmpNe<Output = S> + Copy,
    {
        let dir = position - self.position;
        let norm = dir.norm_squared();

        let force = |norm_s: S| {
            let f = mass * (norm_s * norm_s * norm_s).rsqrt();
            // Branch removed by the compiler when `CHECK_ZERO` is false.
            if CHECK_ZERO {
                dir * f.bitand(norm.cmp_ne(S::ZERO))
            } else {
                dir * f
            }
        };

        // Using this branch results in better performance when softening is zero.
        if softening == S::ZERO {
            force(norm)
        } else {
            force(norm + (softening * softening))
        }
    }

    /// Computes the gravitational acceleration exerted on the current point-mass by the specified
    /// node of the given [`Orthtree`] following the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation)
    /// approximation with the given `theta` parameter, provided `V` and `S` are scalar types.
    #[inline]
    pub fn acceleration_tree<const X: usize, const D: usize>(
        &self,
        tree: &Orthtree<X, D, S, PointMass<V, S>>,
        node: Option<NodeID>,
        theta: S,
        softening: S,
    ) -> V
    where
        V: FloatVector<Float = S> + Copy + Sum,
        S: Float + PartialOrd + Copy,
    {
        let mut acceleration = V::ZERO;

        let estimate = X * (tree.nodes.len() as f32).ln() as usize; // TODO: find a proper estimate
        let mut stack = Vec::with_capacity(estimate);
        stack.push(node);

        while let Some(node) = stack.pop() {
            let id = match node {
                Some(id) => id as usize,
                None => continue,
            };

            let p2 = tree.data[id];
            let dir = p2.position - self.position;
            let norm = dir.norm_squared();

            if norm == S::ZERO {
                acceleration += dir;
            } else {
                match tree.nodes[id] {
                    Node::Internal(node) if theta < node.bbox.width() / norm.sqrt() => {
                        stack.extend(node.orthant);
                    }
                    _ => {
                        let norm_s = norm + (softening * softening);
                        acceleration += dir * p2.mass / (norm_s * norm_s.sqrt());
                    }
                }
            }
        }

        acceleration
    }
}

/// Flexible, copyable storage with references to affected particles and a generic massive storage.
#[derive(Debug)]
pub struct ParticleSystem<'p, V, S, T: ?Sized> {
    /// Particles for which the acceleration is computed.
    pub affected: &'p [PointMass<V, S>],
    /// Particles responsible for the acceleration exerted on the `affected` particles, in a
    /// storage `S`.
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

/// Storage with particles in an [`Orthtree`] and its root.
#[derive(Clone, Debug)]
pub struct ParticleTree<const X: usize, const D: usize, V, S> {
    root: Option<NodeID>,
    tree: Orthtree<X, D, S, PointMass<V, S>>,
}

impl<const X: usize, const D: usize, V, S> ParticleTree<X, D, V, S> {
    /// Returns the root of the [`Orthtree`].
    #[inline]
    pub const fn root(&self) -> Option<NodeID> {
        self.root
    }

    /// Returns a reference to the [`Orthtree`].
    #[inline]
    pub const fn get(&self) -> &Orthtree<X, D, S, PointMass<V, S>> {
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
        let mut tree = Orthtree::with_capacity(slice.len());
        let root = tree.build_node(slice, |p| p.position.into(), PointMass::new_com);

        Self { root, tree }
    }
}

/// [`ParticleSystem`] with a [`ParticleTree`] for the massive storage.
pub type ParticleTreeSystem<'p, const X: usize, const D: usize, V, S> =
    ParticleSystem<'p, V, S, ParticleTree<X, D, V, S>>;

/// Storage inside of which the massive particles are placed before the massless ones.
///
/// Allows for easy optimisation of the computation of forces between massive and massless
/// particles.
#[derive(Clone, Debug)]
pub struct ParticleOrdered<V, S> {
    massive_len: usize,
    particles: Vec<PointMass<V, S>>,
}

impl<V, S> ParticleOrdered<V, S> {
    /// Creates a new [`ParticleOrdered`] with the given massive and massless particles.
    #[inline]
    pub fn with<I, U>(massive: I, massless: U) -> Self
    where
        S: PartialEq + Zero,
        I: IntoIterator<Item = PointMass<V, S>>,
        U: IntoIterator<Item = PointMass<V, S>>,
    {
        let particles = massive.into_iter().chain(massless).collect::<Vec<_>>();
        let massive_len = particles
            .iter()
            .position(PointMass::is_massless)
            .unwrap_or(particles.len());

        Self {
            massive_len,
            particles,
        }
    }

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
        Self::with(
            particles.iter().filter(|p| p.is_massive()).cloned(),
            particles.iter().filter(|p| p.is_massless()).cloned(),
        )
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
