#[cfg(feature = "gpu")]
use crate::{gpu::InteractionShader, gravity::newtonian::AccelerationGPU};
use crate::{
    gravity::{
        newtonian::{AccelerationAt, AccelerationPaired, ToSimd, TreeData},
        Distance, IntoArray, Norm, Position, Reduce,
    },
    sequential::InteractionPair,
    BarnesHutInteraction, Between, Interaction, ReduceSimdInteraction, SimdInteraction,
    TreeInteraction,
};

/// [`Interaction`] representing the gravitational acceleration between two point-masses using
/// Newton's law of universal gravitation.
///
/// If the positions of the affected and affecting particles are guaranteed to be different, this
/// computation can be more efficient with `CHECKED` set to false.
#[derive(Clone, Copy, Debug)]
pub struct Acceleration<const CHECKED: bool>;

impl Default for Acceleration<true> {
    #[inline]
    fn default() -> Self {
        Self::checked()
    }
}

impl Acceleration<true> {
    /// Creates a new [`Acceleration`] that checks if particles share their positions.
    #[inline]
    pub const fn checked() -> Self {
        Self
    }
}

impl Acceleration<false> {
    /// Creates a new [`Acceleration`] that does not check if particles share their positions.
    ///
    /// Unless the affected and affecting particles (which can overlap or be the same) are
    /// guaranteed to have different positions, use [`Acceleration::checked`] instead.
    #[inline]
    pub const fn unchecked() -> Self {
        Self
    }
}

impl<const CHECKED: bool, P1, P2> Interaction<Between<&P1, &P2>> for Acceleration<CHECKED>
where
    P1: Position + ?Sized,
    P2: AccelerationAt<Vector = P1::Vector> + ?Sized,
    P2::Softening: Default,
{
    type Output = P2::Output;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&P1, &P2>) -> Self::Output {
        affecting.acceleration_at::<CHECKED>(&affected.position(), &Default::default())
    }
}

impl<const CHECKED: bool, const L: usize, P> SimdInteraction<L, P> for Acceleration<CHECKED>
where
    P: ToSimd<L>,
{
    type Simd = P::Simd;

    #[inline]
    fn lanes_splat(particle: &P) -> Self::Simd {
        particle.lanes_splat()
    }

    #[inline]
    fn lanes_array(particles: [P; L]) -> Self::Simd {
        P::lanes_array(particles)
    }

    #[inline]
    fn lanes_slice(particles: &[P]) -> Self::Simd {
        P::lanes_slice(particles)
    }
}

impl<const CHECKED: bool, Storage, U> ReduceSimdInteraction<Storage> for Acceleration<CHECKED>
where
    U: Reduce,
    Acceleration<CHECKED>: Interaction<Storage, Output = U>,
{
    type Reduced = U::Output;

    #[inline]
    fn reduce_sum(output: <Self as Interaction<Storage>>::Output) -> Self::Reduced {
        output.reduce_sum()
    }
}

impl<const CHECKED: bool, P> InteractionPair<&P> for Acceleration<CHECKED>
where
    P: AccelerationPaired + ?Sized,
    P::Softening: Default,
{
    type Output = P::Output;

    #[inline]
    fn compute_pair(
        &mut self,
        Between(affected, affecting): Between<&P, &P>,
    ) -> (P::Output, P::Output) {
        affected.acceleration_paired(affecting, &Default::default())
    }
}

impl<'a, const CHECKED: bool, P1, P2> BarnesHutInteraction<&'a P1, &'a P2> for Acceleration<CHECKED>
where
    P1: Position + ?Sized,
    P2: Position<Vector = P1::Vector> + ?Sized,
    P1::Vector: Distance,
    Acceleration<CHECKED>: Interaction<Between<&'a P1, &'a P2>>,
{
    type Distance = <<P1 as Position>::Vector as Norm>::Output;

    #[inline]
    fn distance_squared(affected: &'a P1, affecting: &'a P2) -> Self::Distance {
        // Important: we call `distance_squared` on the affecting particle's position because this
        // allows the compiler to optimise subsequent distance calculations when computing the
        // interaction of these particles because that is how it is implemented.
        affecting.position().distance_squared(affected.position())
    }
}

impl<const CHECKED: bool, P> TreeInteraction<P> for Acceleration<CHECKED>
where
    P: TreeData + Position + Clone,
    P::Vector: IntoArray,
{
    type Coordinates = <P::Vector as IntoArray>::Array;

    type TreeData = P::Data;

    #[inline]
    fn coordinates(p: &P) -> Self::Coordinates {
        p.position().into()
    }

    #[inline]
    fn compute_data(particles: &[P]) -> Self::TreeData {
        P::centre_of_mass(particles.iter().cloned())
    }
}

#[cfg(feature = "gpu")]
impl<const CHECKED: bool, P1, P2> InteractionShader<P1, P2> for Acceleration<CHECKED>
where
    P1: Position,
    P2: AccelerationGPU<CHECKED, Vector = P1::Vector>,
{
    type Output = P2::Output;

    const AFFECTED_SIZE: u64 = std::mem::size_of::<P2::GPUVector>() as u64;

    const AFFECTING_SIZE: u64 = std::mem::size_of::<P2::GPUAffecting>() as u64;

    const INTERACTION_SIZE: u64 = std::mem::size_of::<P2::GPUOutput>() as u64;

    const SHADER: &'static str = P2::SOURCE;

    #[inline]
    fn write_affected(affected: &[P1], mut view: wgpu::QueueWriteBufferView) {
        view.chunks_exact_mut(std::mem::size_of::<P2::GPUVector>())
            .zip(affected.iter())
            .for_each(|(chunk, particle)| {
                chunk.copy_from_slice(bytemuck::bytes_of(&P2::to_gpu_position(
                    &particle.position(),
                )));
            });
    }

    #[inline]
    fn write_affecting(affecting: &[P2], mut view: wgpu::QueueWriteBufferView) {
        view.chunks_exact_mut(std::mem::size_of::<P2::GPUAffecting>())
            .zip(affecting.iter())
            .for_each(|(chunk, particle)| {
                chunk.copy_from_slice(bytemuck::bytes_of(&particle.to_gpu_particle()));
            });
    }

    #[inline]
    fn read_interactions(view: &wgpu::BufferView) -> Vec<Self::Output> {
        bytemuck::cast_slice(view)
            .iter()
            .map(P2::to_cpu_output)
            .collect()
    }
}
