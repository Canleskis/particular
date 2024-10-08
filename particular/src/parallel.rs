use crate::{
    storage::{Ordered, Reordered, RootedOrthtree},
    tree::{BoundingBox, Const, MidPoint, MinMax, SubDivide},
    Between, Interaction, ReduceSimdInteraction, SimdInteraction, TreeInteraction,
};
use rayon::iter::{
    plumbing::{bridge, Consumer, ProducerCallback, UnindexedConsumer},
    IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::{
    iter::Sum,
    ops::{Add, Sub},
};

/// Trait to compute the interaction between particles using different parallel algorithms.
pub trait ParallelCompute<T>: Sized {
    /// Returns the interaction(s) between these particles using a parallel brute-force algorithm.
    ///
    /// Refer to [`BruteForce`] to for more information.
    #[inline]
    fn par_brute_force(self, interaction: T) -> <BruteForce<T> as Interaction<Self>>::Output
    where
        BruteForce<T>: Interaction<Self>,
    {
        BruteForce(interaction).compute(self)
    }

    /// Returns the interaction(s) between these particles using a parallel brute-force algorithm
    /// after converting scalar values to SIMD values.
    ///
    /// Refer to [`BruteForceSimd`] to for more information.
    #[inline]
    fn par_brute_force_simd<const L: usize>(
        self,
        interaction: T,
    ) -> <BruteForceSimd<L, T> as Interaction<Self>>::Output
    where
        BruteForceSimd<L, T>: Interaction<Self>,
    {
        BruteForceSimd(interaction).compute(self)
    }

    /// Returns the interaction(s) between these particles using a parallel Barnes-Hut algorithm.
    ///
    /// Refer to [`BarnesHut`] to for more information.
    #[inline]
    fn par_barnes_hut<S>(
        self,
        theta: S,
        interaction: T,
    ) -> <BarnesHut<S, T> as Interaction<Self>>::Output
    where
        Self: Sized,
        BarnesHut<S, T>: Interaction<Self>,
    {
        BarnesHut::new(theta, interaction).compute(self)
    }
}

// Manual implementations for better linting.
impl<T, P> ParallelCompute<T> for &[P] {}
impl<T, P> ParallelCompute<T> for &Ordered<P> {}
impl<T, P, F> ParallelCompute<T> for &Reordered<'_, P, F> {}
impl<T, S1, S2> ParallelCompute<T> for Between<S1, S2> {}

/// A parallel iterator that computes the interactions between particles in an iterator and a
/// storage using a given interaction algorithm.
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[derive(Clone, Debug)]
pub struct Interactions<A, I, S> {
    algorithm: A,
    affected: I,
    affecting: S,
}

impl<A, I, S> Interactions<A, I, S> {
    #[inline]
    const fn new(algorithm: A, affected: I, affecting: S) -> Self {
        Self {
            algorithm,
            affected,
            affecting,
        }
    }
}

impl<A, I, S, U> ParallelIterator for Interactions<A, I, S>
where
    I: IndexedParallelIterator,
    S: std::ops::Deref + Send + Sync,
    A: for<'a> Interaction<Between<I::Item, &'a S::Target>, Output = U> + Clone + Send + Sync,
    U: Send,
{
    type Item = U;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<A, I, S, U> IndexedParallelIterator for Interactions<A, I, S>
where
    I: IndexedParallelIterator,
    S: std::ops::Deref + Send + Sync,
    A: for<'a> Interaction<Between<I::Item, &'a S::Target>, Output = U> + Clone + Send + Sync,
    U: Send,
{
    #[inline]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        self.affected
            .map(|affected| {
                self.algorithm
                    .clone()
                    .compute(Between(affected, &self.affecting))
            })
            .with_producer(callback)
    }

    #[inline]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    #[inline]
    fn len(&self) -> usize {
        self.affected.len()
    }
}

impl<const X: usize, const D: usize, S, Data, A, I, U> ParallelIterator
    for Interactions<A, I, RootedOrthtree<X, D, S, Data>>
where
    I: IndexedParallelIterator,
    RootedOrthtree<X, D, S, Data>: Send,
    A: Send,
    for<'a> Interactions<A, I, &'a RootedOrthtree<X, D, S, Data>>:
        IndexedParallelIterator<Item = U>,
    U: Send,
{
    type Item = U;

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<const X: usize, const D: usize, S, Data, A, I, U> IndexedParallelIterator
    for Interactions<A, I, RootedOrthtree<X, D, S, Data>>
where
    I: IndexedParallelIterator,
    RootedOrthtree<X, D, S, Data>: Send,
    A: Send,
    for<'a> Interactions<A, I, &'a RootedOrthtree<X, D, S, Data>>:
        IndexedParallelIterator<Item = U>,
    U: Send,
{
    #[inline]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        Interactions::new(self.algorithm, self.affected, &self.affecting).with_producer(callback)
    }

    #[inline]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    #[inline]
    fn len(&self) -> usize {
        self.affected.len()
    }
}

/// Brute-force algorithm using multiple CPU threads with [rayon](https://github.com/rayon-rs/rayon).
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `T` should implement
/// [`Interaction<Between<&P1, &P2>>`].
#[derive(Clone, Copy, Default, Debug)]
pub struct BruteForce<T>(pub T);

impl<P1, P2, U, T> Interaction<Between<&P1, &[P2]>> for BruteForce<T>
where
    P1: Sync,
    P2: Sync,
    U: Sum + Send,
    T: for<'a> Interaction<Between<&'a P1, &'a P2>, Output = U> + Clone + Sync,
{
    type Output = U;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&P1, &[P2]>) -> Self::Output {
        affecting
            .par_iter()
            .map(|p2| self.0.clone().compute(Between(affected, p2)))
            .sum() // Contrary to the sequential version, a map + sum is faster than a fold here.
    }
}

impl<'a, P1, P2, T> Interaction<Between<&'a [P1], &'a [P2]>> for BruteForce<T>
where
    P1: Sync,
    T: Clone,
{
    type Output =
        Interactions<crate::sequential::BruteForce<T>, rayon::slice::Iter<'a, P1>, &'a [P2]>;

    #[inline]
    fn compute(
        &mut self,
        Between(affected, affecting): Between<&'a [P1], &'a [P2]>,
    ) -> Self::Output {
        let inner = crate::sequential::BruteForce(self.0.clone()); // Sequential inner loop is faster.
        Interactions::new(inner, affected.par_iter(), affecting)
    }
}

/// A Brute-force algorithm using multiple CPU threads that converts scalar values to SIMD values
/// before doing the computation.
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `T` should implement
/// [`SimdInteraction<LANES, P1>`], [`SimdInteraction<LANES, P2>`] and
/// [`ReduceSimdInteraction<Between<&P1Simd, &P2Simd>>`] where `P1Simd` and `P2Simd` are the
/// respective SIMD values of `P1` and `P2` defined by the [`SimdInteraction`] implementations.
#[derive(Clone, Copy, Default, Debug)]
pub struct BruteForceSimd<const L: usize, T>(pub T);

// We don't implement `Interaction<Between<&P1, &[P2]>>` because it is generally faster to use
// the scalar version and we don't want to reallocate the SIMD values each inner iteration.
impl<const L: usize, P1, P2Simd, U, R, T> Interaction<Between<&P1, &[P2Simd]>>
    for BruteForceSimd<L, T>
where
    P2Simd: Sync,
    U: Sum + Send,
    T: SimdInteraction<L, P1>
        + for<'a> ReduceSimdInteraction<Between<&'a T::Simd, &'a P2Simd>, Output = U, Reduced = R>
        + Clone
        + Sync,
    T::Simd: Sync,
{
    type Output = R;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&P1, &[P2Simd]>) -> Self::Output {
        let affected = &T::lanes_splat(affected);
        T::reduce_sum(BruteForce(self.0.clone()).compute(Between(affected, affecting)))
    }
}

impl<'a, const L: usize, P1, P2, T> Interaction<Between<&'a [P1], &'a [P2]>>
    for BruteForceSimd<L, T>
where
    P1: Sync,
    T: SimdInteraction<L, P2> + Clone,
{
    type Output = Interactions<
        crate::sequential::BruteForceSimd<L, T>,
        rayon::slice::Iter<'a, P1>,
        Vec<T::Simd>,
    >;

    #[inline]
    fn compute(
        &mut self,
        Between(affected, affecting): Between<&'a [P1], &'a [P2]>,
    ) -> Self::Output {
        let inner = crate::sequential::BruteForceSimd(self.0.clone()); // Sequential inner loop is faster.
        let simd_affecting = T::values_to_simd_vec(affecting);
        Interactions::new(inner, affected.par_iter(), simd_affecting)
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm using
/// multiple CPU threads.
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `T` should implement
/// [`TreeInteraction<P2>`] and
/// [`BarnesHutInteraction<&P1, &TreeData>`](crate::BarnesHutInteraction) where `TreeData` is the
/// computed data stored in the tree created from the `P2` particles defined by the
/// [`TreeInteraction`] implementation.
#[derive(Clone, Copy, Default, Debug)]
pub struct BarnesHut<S, T> {
    /// Parameter ruling the accuracy and speed of the algorithm. If 0, degenerates to
    /// [`BruteForce`].
    pub theta: S,
    /// Interaction to compute between two particles.
    pub interaction: T,
}

impl<S, T> BarnesHut<S, T> {
    /// Creates a new [`BarnesHut`] instance.
    #[inline]
    pub const fn new(theta: S, interaction: T) -> Self {
        Self { theta, interaction }
    }
}

impl<'a, const X: usize, const D: usize, S, P1, Data, T>
    Interaction<Between<&'a [P1], &'a RootedOrthtree<X, D, S, Data>>> for BarnesHut<S, T>
where
    S: Clone,
    P1: Sync,
    T: Clone,
{
    type Output = Interactions<
        crate::sequential::BarnesHut<S, T>,
        rayon::slice::Iter<'a, P1>,
        &'a RootedOrthtree<X, D, S, Data>,
    >;

    #[inline]
    fn compute(
        &mut self,
        Between(affected, affecting): Between<&'a [P1], &'a RootedOrthtree<X, D, S, Data>>,
    ) -> Self::Output {
        let inner = crate::sequential::BarnesHut {
            theta: self.theta.clone(),
            interaction: self.interaction.clone(),
        }; // Sequential inner loop is faster.
        Interactions::new(inner, affected.par_iter(), affecting)
    }
}

// This implementation returns an iterator owning the [`RootedOrthtree`] instead of a reference
// because the it is local to that function.
impl<'a, const X: usize, const D: usize, S, P1, P2, T> Interaction<Between<&'a [P1], &[P2]>>
    for BarnesHut<S, T>
where
    P1: Sync,
    P2: Clone,
    Const<D>: SubDivide<Division = Const<X>>,
    S: Add<Output = S> + Sub<Output = S> + MidPoint + MinMax + PartialOrd + Default + Clone,
    BoundingBox<[S; D]>: Default,
    T: TreeInteraction<P2, Coordinates = [S; D]> + Clone,
{
    type Output = Interactions<
        crate::sequential::BarnesHut<S, T>,
        rayon::slice::Iter<'a, P1>,
        RootedOrthtree<X, D, S, T::TreeData>,
    >;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&'a [P1], &[P2]>) -> Self::Output {
        let inner = crate::sequential::BarnesHut {
            theta: self.theta.clone(),
            interaction: self.interaction.clone(),
        }; // Sequential inner loop is faster.
        let tree = RootedOrthtree::new(affecting, T::coordinates, T::compute_data);
        Interactions::new(inner, affected.par_iter(), tree)
    }
}
