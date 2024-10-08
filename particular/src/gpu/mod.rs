/// Simple abstraction over `wgpu` types to perform computations on the GPU.
pub mod resources;

pub use crate::gpu::resources::MemoryStrategy;

use crate::{
    gpu::resources::WgpuResources,
    storage::{Ordered, Reordered},
    Between, Interaction,
};

/// Trait to compute the interaction between particles using different parallel algorithms.
pub trait GpuCompute<T>: Sized {
    /// Returns the interaction(s) between these particles using a brute-force algorithm performed
    /// on the GPU.
    ///
    /// Refer to [`BruteForce`] for more information.
    #[inline]
    fn gpu_brute_force<'a>(
        self,
        resources: &'a mut GpuResources,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        interaction: T,
    ) -> <BruteForce<'a, T> as Interaction<Self>>::Output
    where
        BruteForce<'a, T>: Interaction<Self>,
    {
        BruteForce::new(resources, device, queue, interaction).compute(self)
    }
}

// Manual implementations for better linting.
impl<T, P> GpuCompute<T> for &[P] {}
impl<T, P> GpuCompute<T> for &Ordered<P> {}
impl<T, P, F> GpuCompute<T> for &Reordered<'_, P, F> {}
impl<T, S1, S2> GpuCompute<T> for Between<S1, S2> {}

/// Trait to compute an interaction between two particles on the GPU.
pub trait InteractionShader<P1, P2> {
    /// The computed interaction.
    type Output;

    /// The shader code for the interaction between two particles. This shader should define three
    /// types: `Affected`, `Affecting` and `Interaction`. They should then be used to define a
    /// `compute` function with the following signature:
    /// ```wgsl
    /// fn compute(p1: Affected, p2: Affecting, out: ptr<function, Interaction>)
    /// ```
    /// where `out` is a pointer to write the interaction to.
    const SHADER: &'static str;

    /// The size of the affected particle on the GPU.
    const AFFECTED_SIZE: u64;

    /// The size of the affected particle on the GPU.
    const AFFECTING_SIZE: u64;

    /// The size of the interaction on the GPU.
    const INTERACTION_SIZE: u64;

    /// Writes the affected particles to the GPU buffer.
    fn write_affected(affected: &[P1], view: wgpu::QueueWriteBufferView);

    /// Writes the affecting particles to the GPU buffer.
    fn write_affecting(affecting: &[P2], view: wgpu::QueueWriteBufferView);

    /// Reads the interactions from the GPU buffer.
    fn read_interactions(view: &wgpu::BufferView) -> Vec<Self::Output>;

    /// Initialises the push constants for the shader.
    #[inline]
    fn init_push_constants(&self) -> &[wgpu::PushConstantRange] {
        &[]
    }

    /// Sets the push constants for the shader.
    #[inline]
    fn push_constants_data(&self) -> &[u8] {
        &[]
    }
}

enum GpuResourcesState {
    New(MemoryStrategy),
    Init(WgpuResources),
}

/// Resources used by `wgpu` for computing on the GPU.
pub struct GpuResources(GpuResourcesState);

impl GpuResources {
    /// Creates a new [`GpuResources`] instance.
    #[inline]
    pub fn new(shader_type: MemoryStrategy) -> Self {
        Self(GpuResourcesState::New(shader_type))
    }

    /// Creates a new [`GpuResources`] instance with initialised buffers and pipeline.
    #[inline]
    pub fn init<T, P1, P2>(interaction: &T, strategy: MemoryStrategy, device: &wgpu::Device) -> Self
    where
        T: InteractionShader<P1, P2>,
    {
        Self(GpuResourcesState::Init(WgpuResources::new(
            T::SHADER,
            device,
            strategy,
            interaction.init_push_constants(),
            T::AFFECTED_SIZE,
            T::AFFECTING_SIZE,
            T::INTERACTION_SIZE,
        )))
    }

    /// Returns a mutable reference to the [`WgpuResources`] if it has been initialised.
    #[inline]
    pub fn get(&mut self) -> Option<&mut WgpuResources> {
        match &mut self.0 {
            GpuResourcesState::Init(resources) => Some(resources),
            _ => None,
        }
    }

    /// Returns a mutable reference to the [`WgpuResources`] or initialises it.
    #[inline]
    pub fn get_or_init<T, P1, P2>(
        &mut self,
        interaction: &T,
        device: &wgpu::Device,
    ) -> &mut WgpuResources
    where
        T: InteractionShader<P1, P2>,
    {
        if let GpuResourcesState::New(strategy) = &mut self.0 {
            *self = GpuResources::init(interaction, *strategy, device);
        }

        // This won't panic because the state is guaranteed to be `Init` after the above block.
        self.get().unwrap()
    }
}

/// Brute-force algorithm using the GPU with [wgpu](https://github.com/gfx-rs/wgpu).
///
/// To use particles `P1` and `P2` with this algorithm, the interaction `M` should implement
/// [`InteractionShader<P1, P2>`].
pub struct BruteForce<'a, T> {
    /// Instanced resources used for the computation. It **should not** be recreated for every
    /// iteration. Doing so can result in significantly reduced performance.
    pub resources: &'a mut GpuResources,
    /// [`wgpu::Device`] used for the computation.
    pub device: &'a wgpu::Device,
    /// [`wgpu::Queue`] used for the computation.
    pub queue: &'a wgpu::Queue,
    /// Interaction to compute between two particles.
    pub interaction: T,
}

impl<'a, T> BruteForce<'a, T> {
    /// Creates a new [`BruteForce`] instance.
    #[inline]
    pub fn new(
        resources: &'a mut GpuResources,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        interaction: T,
    ) -> Self {
        Self {
            resources,
            device,
            queue,
            interaction,
        }
    }
}

impl<P1, P2, T> Interaction<Between<&[P1], &[P2]>> for BruteForce<'_, T>
where
    T: InteractionShader<P1, P2>,
    T::Output: Default,
{
    type Output = std::vec::IntoIter<T::Output>;

    #[inline]
    fn compute(&mut self, Between(affected, affecting): Between<&[P1], &[P2]>) -> Self::Output {
        let gpu_data = self.resources.get_or_init(&self.interaction, self.device);

        T::write_affected(
            affected,
            gpu_data.write_affected_with(self.device, self.queue, affected.len() as u64),
        );

        T::write_affecting(
            affecting,
            gpu_data.write_affecting_with(self.device, self.queue, affecting.len() as u64),
        );

        pollster::block_on(gpu_data.compute(
            self.device,
            self.queue,
            self.interaction.push_constants_data(),
            T::read_interactions,
        ))
        .into_iter()
    }
}
