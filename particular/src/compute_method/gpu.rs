use {
    crate::compute_method::{
        gpu_compute::WgpuResources,
        storage::{ParticleSliceSystem, PointMass},
        ComputeMethod,
    },
    ultraviolet::Vec3,
};

enum GpuResourcesState {
    New(u32),
    Init(WgpuResources),
}

impl GpuResourcesState {
    /// Returns a mutable reference to the [`WgpuResources`] if it is initialised.
    #[inline]
    pub fn get_or_init(&mut self, device: &wgpu::Device) -> &mut WgpuResources {
        if let Self::New(workgroup_size) = self {
            *self = Self::Init(WgpuResources::new(device, *workgroup_size));
        }

        match self {
            Self::Init(resources) => resources,
            _ => unreachable!(),
        }
    }
}

/// Initialised data used by `wgpu` for computing on the GPU.
pub struct GpuData(GpuResourcesState);

impl GpuData {
    /// Creates a new [`GpuData`] instance.
    #[inline]
    pub fn new(workgroup_size: u32) -> Self {
        Self(GpuResourcesState::New(workgroup_size))
    }

    /// Creates a new [`GpuData`] instance with initialised buffers and pipeline.
    #[inline]
    pub fn init(workgroup_size: u32, device: &wgpu::Device) -> Self {
        Self(GpuResourcesState::Init(WgpuResources::new(
            device,
            workgroup_size,
        )))
    }

    /// Returns a mutable reference to the [`WgpuResources`] if it is initialised.
    #[inline]
    pub fn get_or_init(&mut self, device: &wgpu::Device) -> &mut WgpuResources {
        self.0.get_or_init(device)
    }
}

impl Default for GpuData {
    #[inline]
    fn default() -> Self {
        Self(GpuResourcesState::New(256))
    }
}

/// Brute-force [`ComputeMethod`] using the GPU with [wgpu](https://github.com/gfx-rs/wgpu).
///
/// Currently only implemented for 3D f32 vectors. You can still use it in 2D by converting your 2D
/// f32 vectors to 3D f32 vectors. Does not work on WASM.
pub struct BruteForce<'a> {
    /// Instanced resources used for the computation. It **should not** be recreated for every
    /// iteration. Doing so can result in significantly reduced performance.
    pub gpu_data: &'a mut GpuData,
    /// [`wgpu::Device`] used for the computation.
    pub device: &'a wgpu::Device,
    /// [`wgpu::Queue`] used for the computation.
    pub queue: &'a wgpu::Queue,
}

impl<'a> BruteForce<'a> {
    /// Creates a new [`BruteForce`] instance.
    #[inline]
    pub fn new(
        gpu_data: &'a mut GpuData,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
    ) -> Self {
        Self {
            gpu_data,
            device,
            queue,
        }
    }
}

impl ComputeMethod<ParticleSliceSystem<'_, Vec3, f32>> for BruteForce<'_> {
    type Output = Vec<Vec3>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<Vec3, f32>) -> Self::Output {
        let gpu_data = self.gpu_data.get_or_init(self.device);

        gpu_data.write_particle_data(system.affected, system.massive, self.device, self.queue);
        pollster::block_on(gpu_data.compute(self.device, self.queue))
    }
}

unsafe impl<V: bytemuck::Zeroable, S: bytemuck::Zeroable> bytemuck::Zeroable for PointMass<V, S> {}
unsafe impl<V: bytemuck::NoUninit, S: bytemuck::NoUninit> bytemuck::NoUninit for PointMass<V, S> {}

/// Helper function to get default values for [`wgpu::Device`] and [`wgpu::Queue`].
pub async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .unwrap();

    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        let (device, queue) = &pollster::block_on(setup_wgpu());
        let mut gpu_data = GpuData::default();
        tests::acceleration_error(BruteForce::new(&mut gpu_data, device, queue), 1e-2);
        tests::circular_orbit_stability(BruteForce::new(&mut gpu_data, device, queue), 100, 1e-2);
    }
}
