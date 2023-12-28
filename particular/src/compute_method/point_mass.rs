use crate::compute_method::{
    math::{AsPrimitive, Float, FloatVector, FromPrimitive, InfToZero, Sum, Zero, SIMD},
    tree::{partition::SizedOrthant, Node, NodeID, Orthtree},
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

    /// Returns the [`PointMass`] corresponding to the center of mass and total mass of the given slice of point-masses.
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

    /// Computes the gravitational force exerted on the current point-mass using the given position and mass,
    /// provided `V` and `S` are scalar types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the given position,
    /// this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn force_mul_mass_scalar<const CHECK_ZERO: bool>(&self, position: V, mass: S) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + Copy,
    {
        let dir = position - self.position;
        let mag = dir.norm_squared();

        // Branch removed by the compiler when `CHECK_ZERO` is false.
        if CHECK_ZERO && mag == S::ZERO {
            dir
        } else {
            dir * mass / (mag * mag.sqrt())
        }
    }

    /// Computes the gravitational force exerted on the current point-mass by the given point-mass,
    /// provided `V` and `S` are scalar types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the position of the given point-mass,
    /// this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn force_scalar<const CHECK_ZERO: bool>(&self, point_mass: &Self) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + Copy,
    {
        self.force_mul_mass_scalar::<CHECK_ZERO>(point_mass.position, self.mass * point_mass.mass)
    }

    /// Computes the gravitational acceleration exerted on the current point-mass by the given point-mass,
    /// provided `V` and `S` are scalar types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the position of the given point-mass,
    /// this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn acceleration_scalar<const CHECK_ZERO: bool>(&self, point_mass: &Self) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + Copy,
    {
        self.force_mul_mass_scalar::<CHECK_ZERO>(point_mass.position, point_mass.mass)
    }

    /// Computes the gravitational force exerted on the current point-mass using the given position and mass,
    /// provided `V` and `S` are SIMD types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the given position,
    /// this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn force_mul_mass_simd<const CHECK_ZERO: bool>(&self, position: V, mass: S) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + InfToZero + Copy,
    {
        let dir = position - self.position;
        let mag = dir.norm_squared();

        // Branch removed by the compiler when `CHECK_ZERO` is false.
        if CHECK_ZERO {
            dir * mass * (mag.recip() * mag.recip_sqrt()).inf_to_zero()
        } else {
            dir * mass * (mag.recip() * mag.recip_sqrt())
        }
    }

    /// Computes the gravitational acceleration exerted on the current point-mass by the given point-mass,
    /// provided `V` and `S` are SIMD types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the position of the given point-mass,
    /// this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn force_simd<const CHECK_ZERO: bool>(&self, point_mass: &Self) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + InfToZero + Copy,
    {
        self.force_mul_mass_simd::<CHECK_ZERO>(point_mass.position, self.mass * point_mass.mass)
    }

    /// Computes the gravitational acceleration exerted on the current point-mass by the given point-mass,
    /// provided `V` and `S` are SIMD types.
    ///
    /// If the position of the current point-mass is guaranteed to be different from the position of the given point-mass,
    /// this computation can be more efficient with `CHECK_ZERO` set to false.
    #[inline]
    pub fn acceleration_simd<const CHECK_ZERO: bool>(&self, point_mass: &Self) -> V
    where
        V: FloatVector<Float = S> + Copy,
        S: Float + InfToZero + Copy,
    {
        self.force_mul_mass_simd::<CHECK_ZERO>(point_mass.position, point_mass.mass)
    }

    /// Computes the gravitational acceleration exerted on the current point-mass by the specified node of the given [`BarnesHutTree`] following the
    /// [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) approximation with the given `theta` parameter,
    /// provided `V` and `S` are scalar types.
    #[inline]
    pub fn acceleration_tree<const X: usize, const D: usize>(
        &self,
        tree: &BarnesHutTree<X, D, V, S>,
        node: Option<NodeID>,
        theta: S,
    ) -> V
    where
        V: FloatVector<Float = S> + Copy + Sum,
        S: Float + PartialOrd + Copy,
    {
        let Some(id) = node else {
            return V::ZERO;
        };
        let id = id as usize;

        let p2 = tree.data[id];

        let dir = p2.position - self.position;
        let mag = dir.norm_squared();

        if mag == S::ZERO {
            return dir;
        }

        match tree.nodes[id] {
            Node::Internal(SizedOrthant { orthant, bbox }) if theta < bbox.width() / mag.sqrt() => {
                orthant
                    .map(|node| self.acceleration_tree(tree, node, theta))
                    .into_iter()
                    .sum()
            }
            _ => dir * p2.mass / (mag * mag.sqrt()),
        }
    }
}

/// [`Orthtree`] suitable for the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation) approximation algorithm.
pub type BarnesHutTree<const X: usize, const D: usize, V, S> = Orthtree<X, D, S, PointMass<V, S>>;
