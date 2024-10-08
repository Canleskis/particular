use crate::{
    tree::{
        partition::{BoundingBox, Const, MidPoint, MinMax, SubDivide},
        NodeID, Orthtree,
    },
    Between, Interaction,
};
use std::ops::{Add, Sub};

/// Storage as an [`Orthtree`] and its root.
#[derive(Clone, Debug)]
pub struct RootedOrthtree<const X: usize, const D: usize, S, Data> {
    root: Option<NodeID>,
    tree: Orthtree<X, D, S, Data>,
}

impl<const X: usize, const D: usize, S, Data> RootedOrthtree<X, D, S, Data> {
    /// Creates a new [`RootedOrthtree`] from the given slice of particles and functions.
    #[inline]
    pub fn new<P, F, G>(slice: &[P], coordinates: F, compute: G) -> Self
    where
        Const<D>: SubDivide<Division = Const<X>>,
        P: Clone,
        F: Fn(&P) -> [S; D] + Copy,
        G: Fn(&[P]) -> Data + Copy,
        S: Add<Output = S> + Sub<Output = S> + MidPoint + MinMax + PartialOrd + Default + Clone,
        BoundingBox<[S; D]>: Default,
    {
        let mut tree = Orthtree::with_capacity(slice.len());
        let root = tree.build_node(slice, coordinates, compute);

        Self { root, tree }
    }

    /// Returns the root of the [`Orthtree`], most likely 0.
    #[inline]
    pub const fn root(&self) -> Option<NodeID> {
        self.root
    }

    /// Returns a reference to the [`Orthtree`].
    #[inline]
    pub const fn get(&self) -> &Orthtree<X, D, S, Data> {
        &self.tree
    }
}

/// Storage inside of which the affecting particles are placed before the non-affecting ones.
///
/// Allows for easy optimisation of the computation between affecting and non-affecting particles.
#[derive(Clone, Debug)]
pub struct Ordered<P> {
    affecting_len: usize,
    particles: Vec<P>,
}

impl<P> Ordered<P> {
    /// Creates a new [`Ordered`] storage with the given affecting and non-affecting particles and
    /// function determining if a particle affects others.
    #[inline]
    pub fn with<I, U, F>(affecting: I, non_affecting: U, is_affecting_fn: F) -> Self
    where
        I: IntoIterator<Item = P>,
        U: IntoIterator<Item = P>,
        F: Fn(&P) -> bool,
    {
        let particles = affecting
            .into_iter()
            .chain(non_affecting)
            .collect::<Vec<_>>();
        let affecting_len = particles
            .iter()
            .position(|p| !is_affecting_fn(p))
            .unwrap_or(particles.len());

        Self {
            affecting_len,
            particles,
        }
    }

    /// Creates a new [`Ordered`] storage from the given unordered particles and function
    /// determining if a particle affects others.
    #[inline]
    pub fn new<F>(unordered: &[P], is_affecting_fn: F) -> Self
    where
        P: Clone,
        F: Fn(&P) -> bool + Clone,
    {
        Self::with(
            unordered.iter().filter(|p| is_affecting_fn(p)).cloned(),
            unordered.iter().filter(|p| !is_affecting_fn(p)).cloned(),
            is_affecting_fn.clone(),
        )
    }

    /// Returns the number of stored affecting particles.
    #[inline]
    pub const fn affecting_len(&self) -> usize {
        self.affecting_len
    }

    /// Returns a reference to the affecting particles.
    #[inline]
    pub fn affecting(&self) -> &[P] {
        &self.particles[..self.affecting_len]
    }

    /// Returns a reference to the non-affecting particles.
    #[inline]
    pub fn non_affecting(&self) -> &[P] {
        &self.particles[self.affecting_len..]
    }

    /// Returns a reference to the particles.
    #[inline]
    pub fn particles(&self) -> &[P] {
        &self.particles
    }

    /// Returns a mutable reference to the affecting particles.
    #[inline]
    pub fn affecting_mut(&mut self) -> &mut [P] {
        &mut self.particles[..self.affecting_len]
    }

    /// Returns a mutable reference to the non-affecting particles.
    #[inline]
    pub fn non_affecting_mut(&mut self) -> &mut [P] {
        &mut self.particles[self.affecting_len..]
    }

    /// Returns a mutable reference to the stored ordered particles.
    #[inline]
    pub fn particles_mut(&mut self) -> &mut [P] {
        &mut self.particles
    }
}

/// Storage with a copy of the stored particles inside an [`Ordered`] storage.
#[derive(Clone, Debug)]
pub struct Reordered<'p, P, F> {
    /// Original, unordered particles.
    pub unordered: &'p [P],
    ordered: Ordered<P>,
    is_affecting_fn: F,
}

impl<'a, P, F> Reordered<'a, P, F> {
    /// Creates a new [`Reordered`] storage with the given unordered particles and function
    /// determining if a particle affects others.
    #[inline]
    pub fn new(unordered: &'a [P], is_affecting_fn: F) -> Self
    where
        P: Clone,
        F: Fn(&P) -> bool + Copy,
    {
        Self {
            unordered,
            ordered: Ordered::new(unordered, is_affecting_fn),
            is_affecting_fn,
        }
    }
}

impl<P, F> Reordered<'_, P, F> {
    /// Returns a reference to the [`Ordered`] storage.
    #[inline]
    pub const fn ordered(&self) -> &Ordered<P> {
        &self.ordered
    }

    /// Returns the number of stored affecting particles.
    #[inline]
    pub const fn affecting_len(&self) -> usize {
        self.ordered.affecting_len()
    }

    /// Returns a reference to the affecting particles.
    #[inline]
    pub fn affecting(&self) -> &[P] {
        self.ordered.affecting()
    }

    /// Returns a reference to the non-affecting particles.
    #[inline]
    pub fn non_affecting(&self) -> &[P] {
        self.ordered.non_affecting()
    }

    /// Returns a reference to the stored ordered particles.
    #[inline]
    pub fn reordered(&self) -> &[P] {
        self.ordered.particles()
    }

    /// Returns a copy of the function used to determine if a particle affects others.
    #[inline]
    pub fn is_affecting_fn(&self) -> F
    where
        F: Clone,
    {
        self.is_affecting_fn.clone()
    }
}

impl<'p, P, C, O> Interaction<&'p Ordered<P>> for C
where
    C: Interaction<Between<&'p [P], &'p [P]>, Output = O>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, ordered: &'p Ordered<P>) -> Self::Output {
        self.compute(Between(ordered.particles(), ordered.affecting()))
    }
}

impl<'p, P, F, C, O> Interaction<&'p Reordered<'p, P, F>> for C
where
    C: Interaction<Between<&'p [P], &'p [P]>, Output = O>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, reordered: &'p Reordered<'p, P, F>) -> Self::Output {
        self.compute(Between(reordered.unordered, reordered.affecting()))
    }
}

impl<'p, P, C, O> Interaction<&'p [P]> for C
where
    C: Interaction<Between<&'p [P], &'p [P]>, Output = O>,
{
    type Output = O;

    #[inline]
    fn compute(&mut self, slice: &'p [P]) -> Self::Output {
        self.compute(Between(slice, slice))
    }
}
