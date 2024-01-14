use crate::compute_method::{
    gpu_compute,
    math::Zero,
    storage::{ParticleSliceSystem, PointMass},
    ComputeMethod,
};
use ultraviolet::Vec3;

/// Initialized data used by `wgpu` for computing on the GPU.
pub struct GpuData {
    affected_count: usize,
    massive_count: usize,
    resources: Option<gpu_compute::WgpuResources>,
}

impl GpuData {
    /// Creates a new [`GpuData`] instance.
    #[inline]
    pub fn new() -> Self {
        Self {
            affected_count: 0,
            massive_count: 0,
            resources: None,
        }
    }

    /// Creates a new [`BruteForce`] instance with initialized buffers and pipeline.
    #[inline]
    pub fn new_init(device: &wgpu::Device, affected_count: usize, massive_count: usize) -> Self {
        Self {
            affected_count,
            massive_count,
            resources: Some(gpu_compute::WgpuResources::init(
                device,
                affected_count,
                massive_count,
            )),
        }
    }

    #[inline]
    fn get_or_init(&mut self, device: &wgpu::Device) -> &mut gpu_compute::WgpuResources {
        self.resources.get_or_insert_with(|| {
            gpu_compute::WgpuResources::init(device, self.affected_count, self.massive_count)
        })
    }

    #[inline]
    fn update(&mut self, affected_count: usize, massive_count: usize) {
        if self.affected_count != affected_count || self.massive_count != massive_count {
            self.affected_count = affected_count;
            self.massive_count = massive_count;
            self.resources = None;
        }
    }
}

impl Default for GpuData {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Brute-force [`ComputeMethod`] using the GPU with [wgpu](https://github.com/gfx-rs/wgpu).
///
/// Currently only available for 3D f32 vectors. You can still use it in 2D by converting your 2D
/// f32 vectors to 3D f32 vectors.
pub struct BruteForce<'a> {
    /// Instanced [`GpuData`] used for the computation. It **should not** be recreated for every
    /// iteration. Doing so would result in significantly reduced performance.
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
            device,
            queue,
            gpu_data,
        }
    }
}

impl ComputeMethod<ParticleSliceSystem<'_, Vec3, f32>> for BruteForce<'_> {
    type Output = Vec<Vec3>;

    #[inline]
    fn compute(&mut self, system: ParticleSliceSystem<'_, Vec3, f32>) -> Self::Output {
        let affected_count = system.affected.len();
        let massive_count = system.massive.len();

        if massive_count == 0 {
            return system.affected.iter().map(|_| Vec3::ZERO).collect();
        }

        let gpu_data = {
            self.gpu_data.update(affected_count, massive_count);
            self.gpu_data.get_or_init(self.device)
        };

        gpu_data.write_particle_data(system.affected, system.massive, self.queue);
        gpu_data.compute_pass(
            self.device,
            self.queue,
            (affected_count as f32 / 256.0).ceil() as u32,
        );

        pollster::block_on(gpu_data.read_accelerations(self.device))
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
        let (d, q) = &pollster::block_on(setup_wgpu());
        tests::acceleration_error(BruteForce::new(&mut GpuData::new(), d, q), 1e-2);
        tests::circular_orbit_stability(BruteForce::new(&mut GpuData::new(), d, q), 100, 1e-2);
    }
}
