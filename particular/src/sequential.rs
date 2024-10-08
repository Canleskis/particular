use crate::{
    storage::{Ordered, Reordered, RootedOrthtree},
    tree::{BoundingBox, Const, MidPoint, MinMax, Node, SubDivide},
    BarnesHutInteraction, Between, Interaction, ReduceSimdInteraction, SimdInteraction,
    TreeInteraction,
};
use std::ops::{Add, AddAssign, Mul, Sub};

/// Trait to compute the interaction between particles using different sequential algorithms.
pub trait SequentialCompute<T>: Sized {
    /// Returns the interaction between these particles using a brute-force algorithm.
    ///
    /// Refer to [`BruteForce`] for more information.
    #[inline]
    fn brute_force(self, interaction: T) -> <BruteForce<T> as Interaction<Self>>::Output
    where
        BruteForce<T>: Interaction<Self>,
    {
        BruteForce(interaction).compute(self)
    }

    /// Returns the interaction between these particles using a brute-force algorithm, only
    /// performing the computation over the combination of pairs.
    ///
    /// Refer to [`BruteForcePairs`] for more information.
    #[inline]
    fn brute_force_pairs(self, interaction: T) -> <BruteForcePairs<T> as Interaction<Self>>::Output
    where
        BruteForcePairs<T>: Interaction<Self>,
    {
        BruteForcePairs(interaction).compute(self)
    }

    /// Returns the interaction between these particles using a brute-force algorithm, after
    /// converting scalar values to SIMD values.
    ///
    /// Refer to [`BruteForceSimd`] for more information.
    #[inline]
    fn brute_force_simd<const L: usize>(
        self,
        interaction: T,
    ) -> <BruteForceSimd<L, T> as Interaction<Self>>::Output
    where
        BruteForceSimd<L, T>: Interaction<Self>,
    {
        BruteForceSimd(interaction).compute(self)
    }

    /// Returns the interaction between these particles using a Barnes-Hut algorithm.
    ///
    /// Refer to [`BarnesHut`] for more information.
    #[inline]
    fn barnes_hut<S>(
        self,
        theta: S,
        interaction: T,
    ) -> <BarnesHut<S, T> as Interaction<Self>>::Output
    where
        BarnesHut<S, T>: Interaction<Self>,
    {
        BarnesHut::new(theta, interaction).compute(self)
    }
}

// Manual implementations for better linting.
impl<T, P> SequentialCompute<T> for &[P] {}
impl<T, P> SequentialCompute<T> for &Ordered<P> {}
impl<T, P, F> SequentialCompute<T> for &Reordered<'_, P, F> {}
impl<T, S1, S2> SequentialCompute<T> for Between<S1, S2> {}

/// An iterator that computes the interactions between particles in an iterator and a storage using
/// a given interaction algorithm.
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

// This implementation covers slices, types that dereference to slices and references to storages.
impl<A, I, S, U> Iterator for Interactions<A, I, S>
where
    I: Iterator,
    S: std::ops::Deref,
    A: for<'a> Interaction<Between<I::Item, &'a S::Target>, Output = U>,
{
    type Item = U;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.affected
            .next()
            .map(|p1| self.algorithm.compute(Between(p1, &self.affecting)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.affected.size_hint()
    }
}

impl<const X: usize, const D: usize, S, Data, A, I, U> Iterator
    for Interactions<A, I, RootedOrthtree<X, D, S, Data>>
where
    I: Iterator,
    A: for<'a> Interaction<Between<I::Item, &'a RootedOrthtree<X, D, S, Data>>, Output = U>,
{
    type Item = U;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.affected
            .next()
            .map(|p1| self.algorithm.compute(Between(p1, &self.affecting)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.affected.size_hint()
    }
}

impl<A, I, S, U> DoubleEndedIterator for Interactions<A, I, S>
where
    I: DoubleEndedIterator,
    S: std::ops::Deref,
    A: for<'a> Interaction<Between<I::Item, &'a S::Target>, Output = U>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.affected
            .next_back()
            .map(|p1| self.algorithm.compute(Between(p1, &self.affecting)))
    }
}

impl<const X: usize, const D: usize, S, Data, A, I, U> DoubleEndedIterator
    for Interactions<A, I, RootedOrthtree<X, D, S, Data>>
where
    I: DoubleEndedIterator,
    A: for<'a> Interaction<Between<I::Item, &'a RootedOrthtree<X, D, S, Data>>, Output = U>,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.affected
            .next_back()
            .map(|p1| self.algorithm.compute(Between(p1, &self.affecting)))
    }
}

impl<A, I, S, U> ExactSizeIterator for Interactions<A, I, S>
where
    I: ExactSizeIterator,
    Self: Iterator<Item = U>,
{
    #[inline]
    fn len(&self) -> usize {
        self.affected.len()
    }
}

/// Brute-force algorithm using one CPU thread.
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `T` should implement
/// [`Interaction<Between<&P1, &P2>>`].
#[derive(Clone, Copy, Default, Debug)]
pub struct BruteForce<T>(pub T);

impl<P1, P2, U, T> Interaction<Between<&P1, &[P2]>> for BruteForce<T>
where
    U: Add<Output = U> + Default,
    T: for<'a> Interaction<Between<&'a P1, &'a P2>, Output = U>,
{
    type Output = U;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&P1, &[P2]>) -> Self::Output {
        affecting.iter().fold(U::default(), |interaction, p2| {
            interaction + self.0.compute(Between(affected, p2))
        })
    }
}

impl<'a, P1, P2, T> Interaction<Between<&'a [P1], &'a [P2]>> for BruteForce<T>
where
    T: Clone,
{
    type Output = Interactions<Self, std::slice::Iter<'a, P1>, &'a [P2]>;

    #[inline]
    fn compute(
        &mut self,
        Between(affected, affecting): Between<&'a [P1], &'a [P2]>,
    ) -> Self::Output {
        Interactions::new(self.clone(), affected.iter(), affecting)
    }
}

/// A Brute-force algorithm using one CPU thread that converts scalar values to SIMD values before
/// doing the computation.
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `T` should implement
/// [`SimdInteraction<LANES, P1>`], [`SimdInteraction<LANES, P2>`] and
/// [`ReduceSimdInteraction<Between<&P1Simd, &P2Simd>>`] where `P1Simd` and `P2Simd` are the
/// respective SIMD values of `P1` and `P2` defined by the [`SimdInteraction`] implementations.
#[derive(Clone, Copy, Default, Debug)]
pub struct BruteForceSimd<const L: usize, T>(pub T);

// We don't implement `Interaction<Between<&P1, &[P2]>>` because we don't want to reallocate the
// SIMD values each inner iteration.
impl<const L: usize, P1, P2Simd, U, R, T> Interaction<Between<&P1, &[P2Simd]>>
    for BruteForceSimd<L, T>
where
    U: Add<Output = U> + Default,
    T: SimdInteraction<L, P1>
        + for<'a> ReduceSimdInteraction<Between<&'a T::Simd, &'a P2Simd>, Output = U, Reduced = R>
        + Clone,
{
    type Output = R;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&P1, &[P2Simd]>) -> Self::Output {
        let affected = &T::lanes_splat(affected);
        T::reduce_sum(BruteForce(self.0.clone()).compute(Between(affected, affecting)))
    }
}

impl<'a, const L: usize, P1, P2, T> Interaction<Between<&'a [P1], &[P2]>> for BruteForceSimd<L, T>
where
    T: SimdInteraction<L, P2> + Clone,
{
    type Output = Interactions<Self, std::slice::Iter<'a, P1>, Vec<T::Simd>>;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&'a [P1], &[P2]>) -> Self::Output {
        let simd_affecting = T::values_to_simd_vec(affecting);
        Interactions::new(self.clone(), affected.iter(), simd_affecting)
    }
}

/// Trait to implement optimised pair computation of an interaction. Such implementations are used
///  by the [`BruteForcePairs`] algorithm.
pub trait InteractionPair<P> {
    /// The computed interaction.
    type Output;

    /// Returns the computed interactions between two distinct particles.
    fn compute_pair(&mut self, pair: Between<P, P>) -> (Self::Output, Self::Output);
}

/// Brute-force algorithm for computing a subset of pair-wise interactions.
///
/// See [`BruteForcePairs`] for an ergonomic way to use this algorithm.
#[derive(Clone, Copy, Default, Debug)]
pub struct RestrictedBruteForce<T> {
    /// The subset of pairs to compute.
    pub restrict_len: usize,
    /// Interaction to compute between two particles.
    pub interaction: T,
}

impl<T> RestrictedBruteForce<T> {
    /// Creates a new [`RestrictedBruteForce`] instance.
    #[inline]
    pub const fn new(restrict_len: usize, interaction: T) -> Self {
        Self {
            restrict_len,
            interaction,
        }
    }
}

impl<P, U, T> Interaction<&[P]> for RestrictedBruteForce<T>
where
    U: AddAssign + Default,
    T: for<'a> InteractionPair<&'a P, Output = U>,
{
    type Output = Vec<U>;

    #[inline]
    fn compute(&mut self, slice: &[P]) -> Self::Output {
        let len = slice.len();
        assert!(
            len >= self.restrict_len,
            "restrict_len is greater than the length of the slice"
        );

        let mut output: Vec<_> = std::iter::repeat_with(Default::default).take(len).collect();

        for i in 0..self.restrict_len {
            let mut output_i = U::default();

            for j in (i + 1)..len {
                let computed = self.interaction.compute_pair(Between(&slice[i], &slice[j]));

                output_i += computed.0;
                output[j] += computed.1;
            }

            output[i] += output_i;
        }

        output
    }
}

/// Brute-force algorithm using one CPU thread.
///
/// Typically faster than [`BruteForce`] because it performs the computation over the combination
/// of pairs instead of over all the pairs.
///
/// To use particles `P` with this algorithm, the interaction `T` should implement
/// [`InteractionPair<&P>`].
#[derive(Clone, Copy, Default, Debug)]
pub struct BruteForcePairs<T>(pub T);

impl<P, U, T> Interaction<&[P]> for BruteForcePairs<T>
where
    U: AddAssign + Default,
    T: for<'a> InteractionPair<&'a P, Output = U> + Clone,
{
    type Output = std::vec::IntoIter<U>;

    #[inline]
    fn compute(&mut self, slice: &[P]) -> Self::Output {
        RestrictedBruteForce::new(slice.len(), self.0.clone())
            .compute(slice)
            .into_iter()
    }
}

impl<P, U, T> Interaction<&Ordered<P>> for BruteForcePairs<T>
where
    U: AddAssign + Default,
    T: for<'a> InteractionPair<&'a P, Output = U> + Clone,
{
    type Output = std::vec::IntoIter<U>;

    #[inline]
    fn compute(&mut self, ordered: &Ordered<P>) -> Self::Output {
        RestrictedBruteForce::new(ordered.affecting_len(), self.0.clone())
            .compute(ordered.particles())
            .into_iter()
    }
}

/// An iterator that reorders the computed interactions of an unordered iterator of particles.
#[derive(Clone)]
pub struct ReorderedInteractions<I, F, U> {
    unordered: I,
    is_affecting_fn: F,
    interactions: Vec<U>,
    affected_index: usize,
    affecting_index: usize,
}

impl<I, F, U> std::fmt::Debug for ReorderedInteractions<I, F, U>
where
    I: std::fmt::Debug,
    U: std::fmt::Debug,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReorderedInteractions")
            .field("unordered", &self.unordered)
            .field("is_affecting_fn", &std::any::type_name::<F>())
            .field("interactions", &self.interactions)
            .field("affected_index", &self.affected_index)
            .field("affecting_index", &self.affecting_index)
            .finish()
    }
}

impl<'a, P, F, U> ReorderedInteractions<std::slice::Iter<'a, P>, F, U> {
    #[inline]
    fn new(reordered: &Reordered<'a, P, F>, interactions: Vec<U>) -> Self
    where
        F: Clone,
    {
        Self {
            unordered: reordered.unordered.iter(),
            is_affecting_fn: reordered.is_affecting_fn(),
            interactions,
            affected_index: reordered.affecting_len(),
            affecting_index: 0,
        }
    }
}

// Note: `Ordered`'s reordering preserves the order inside the two groups of particles.
impl<P, F, U> Iterator for ReorderedInteractions<std::slice::Iter<'_, P>, F, U>
where
    F: Fn(&P) -> bool,
    U: Clone + Default,
{
    type Item = U;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = match (self.is_affecting_fn)(self.unordered.next()?) {
            true => &mut self.affecting_index,
            false => &mut self.affected_index,
        };

        let interaction = std::mem::take(&mut self.interactions[*index]);
        *index += 1;

        Some(interaction)
    }
}

impl<'a, P, F, U, T> Interaction<&'a Reordered<'_, P, F>> for BruteForcePairs<T>
where
    F: Clone,
    U: AddAssign + Default,
    T: for<'b> InteractionPair<&'b P, Output = U> + Clone,
{
    type Output = ReorderedInteractions<std::slice::Iter<'a, P>, F, U>;

    #[inline]
    fn compute(&mut self, reordered: &'a Reordered<P, F>) -> Self::Output {
        ReorderedInteractions::new(
            reordered,
            RestrictedBruteForce::new(reordered.affecting_len(), self.0.clone())
                .compute(reordered.reordered()),
        )
    }
}

/// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) algorithm using one
/// CPU thread.
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `T` should implement
/// [`TreeInteraction<P2>`] and [`BarnesHutInteraction<&P1, &TreeData>`] where `TreeData` is the
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

impl<const X: usize, const D: usize, S, P1, Data, U, T>
    Interaction<Between<&P1, &RootedOrthtree<X, D, S, Data>>> for BarnesHut<S, T>
where
    S: Sub<Output = S> + Mul<Output = S> + PartialOrd + Default + Clone,
    U: AddAssign + Default,
    T: for<'a> BarnesHutInteraction<&'a P1, &'a Data, Distance = S, Output = U>,
{
    type Output = U;

    #[inline]
    fn compute(
        &mut self,
        Between(affected, affecting): Between<&P1, &RootedOrthtree<X, D, S, Data>>,
    ) -> Self::Output {
        let mut interaction = U::default();

        let estimate = X * (affecting.get().nodes.len() as f32).ln() as usize; // TODO: find a proper estimate
        let mut stack = Vec::with_capacity(estimate);
        stack.push(affecting.root());

        while let Some(node) = stack.pop() {
            let id = match node {
                Some(id) => id as usize,
                None => continue,
            };

            let p2 = &affecting.get().data[id];
            let distance_squared = T::distance_squared(affected, p2);

            if distance_squared == S::default() {
                continue;
            }

            match &affecting.get().nodes[id] {
                Node::Internal(node)
                    if (self.theta.clone() * self.theta.clone()) * distance_squared
                        < (node.bbox.width() * node.bbox.width()) =>
                {
                    stack.extend(node.orthant);
                }
                _ => {
                    // Eventual additional computation of the distance can be optimized out by the
                    // compiler.
                    interaction += self.interaction.compute(Between(affected, p2));
                }
            }
        }

        interaction
    }
}

impl<'a, const X: usize, const D: usize, S, P1, Data, T>
    Interaction<Between<&'a [P1], &'a RootedOrthtree<X, D, S, Data>>> for BarnesHut<S, T>
where
    Self: Clone,
{
    type Output = Interactions<Self, std::slice::Iter<'a, P1>, &'a RootedOrthtree<X, D, S, Data>>;

    #[inline]
    fn compute(
        &mut self,
        Between(affected, affecting): Between<&'a [P1], &'a RootedOrthtree<X, D, S, Data>>,
    ) -> Self::Output {
        Interactions::new(self.clone(), affected.iter(), affecting)
    }
}

// This implementation returns an iterator owning the [`RootedOrthtree`] instead of a reference
// because the it is local to that function.
impl<'a, const X: usize, const D: usize, S, P1, P2, T> Interaction<Between<&'a [P1], &[P2]>>
    for BarnesHut<S, T>
where
    P2: Clone,
    Const<D>: SubDivide<Division = Const<X>>,
    S: Add<Output = S> + Sub<Output = S> + MidPoint + MinMax + PartialOrd + Default + Clone,
    BoundingBox<[S; D]>: Default,
    T: TreeInteraction<P2, Coordinates = [S; D]> + Clone,
{
    type Output =
        Interactions<Self, std::slice::Iter<'a, P1>, RootedOrthtree<X, D, S, T::TreeData>>;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&'a [P1], &[P2]>) -> Self::Output {
        let tree = RootedOrthtree::new(affecting, T::coordinates, T::compute_data);
        Interactions::new(self.clone(), affected.iter(), tree)
    }
}
