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

/// Algorithm to compute the gravitational acceleration between two point-masses using Newton's law
/// of universal gravitation with a softening parameter.
///
/// If the positions of the affected and affecting particles are guaranteed to be different, this
/// computation can be more efficient with `CHECKED` set to false.
#[derive(Clone, Copy, Debug)]
pub struct AccelerationSoftened<S, const CHECKED: bool> {
    /// Softening parameter to avoid singularities.
    pub softening: S,
}

impl<S: Default> Default for AccelerationSoftened<S, true> {
    #[inline]
    fn default() -> Self {
        Self::checked(Default::default())
    }
}

impl<S> AccelerationSoftened<S, true> {
    /// Creates a new [`AccelerationSoftened`] that checks if particles share their positions with
    /// the given softening parameter.
    #[inline]
    pub const fn checked(softening: S) -> Self {
        Self { softening }
    }
}

impl<S> AccelerationSoftened<S, false> {
    /// Creates a new [`AccelerationSoftened`] that does not check if particles share their
    /// positions with the given softening parameter.
    ///
    /// Unless the affected and affecting particles (which can overlap or be the same) are
    /// guaranteed to have different positions, use [`AccelerationSoftened::checked`] instead.
    #[inline]
    pub const fn unchecked(softening: S) -> Self {
        Self { softening }
    }
}

impl<const CHECKED: bool, S, P1, P2> Interaction<Between<&P1, &P2>>
    for AccelerationSoftened<S, CHECKED>
where
    P1: Position + ?Sized,
    P2: AccelerationAt<Softening = S, Vector = P1::Vector> + ?Sized,
{
    type Output = P2::Output;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&P1, &P2>) -> Self::Output {
        affecting.acceleration_at::<CHECKED>(&affected.position(), &self.softening)
    }
}

impl<const CHECKED: bool, S, const L: usize, P> SimdInteraction<L, P>
    for AccelerationSoftened<S, CHECKED>
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

impl<const CHECKED: bool, S, Storage, U> ReduceSimdInteraction<Storage>
    for AccelerationSoftened<S, CHECKED>
where
    U: Reduce,
    AccelerationSoftened<S, CHECKED>: Interaction<Storage, Output = U>,
{
    type Reduced = U::Output;

    #[inline]
    fn reduce_sum(output: <Self as Interaction<Storage>>::Output) -> Self::Reduced {
        output.reduce_sum()
    }
}

impl<const CHECKED: bool, S, P> InteractionPair<&P> for AccelerationSoftened<S, CHECKED>
where
    P: AccelerationPaired<Softening = S> + ?Sized,
{
    type Output = P::Output;

    #[inline]
    fn compute_pair(
        &mut self,
        Between(affected, affecting): Between<&P, &P>,
    ) -> (P::Output, P::Output) {
        affected.acceleration_paired(affecting, &self.softening)
    }
}

impl<'a, const CHECKED: bool, S, P1, P2> BarnesHutInteraction<&'a P1, &'a P2>
    for AccelerationSoftened<S, CHECKED>
where
    P1: Position + ?Sized,
    P2: Position<Vector = P1::Vector> + ?Sized,
    P1::Vector: Distance,
    AccelerationSoftened<S, CHECKED>: Interaction<Between<&'a P1, &'a P2>>,
{
    type Distance = <<P1 as Position>::Vector as Norm>::Output;

    #[inline]
    fn distance_squared(affected: &'a P1, affecting: &'a P2) -> Self::Distance {
        // Important note: we call `distance_squared` on the affecting particle's position because
        // this allows the compiler to optimise subsequent distance calculations when computing
        // the acceleration of these particles because that is how it is implemented.
        affecting.position().distance_squared(affected.position())
    }
}

impl<const CHECKED: bool, S, P> TreeInteraction<P> for AccelerationSoftened<S, CHECKED>
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
impl<const CHECKED: bool, S, P1, P2> InteractionShader<P1, P2> for AccelerationSoftened<S, CHECKED>
where
    P1: Position,
    P2: AccelerationGPU<CHECKED, Vector = P1::Vector>,
    S: bytemuck::NoUninit,
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

    #[inline]
    fn init_push_constants(&self) -> &[wgpu::PushConstantRange] {
        &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..((std::mem::size_of::<S>() + 3) & !3) as u32,
        }]
    }

    #[inline]
    fn push_constants_data(&self) -> &[u8] {
        bytemuck::cast_slice(std::slice::from_ref(&self.softening))
    }
}
