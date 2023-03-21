use glam::Vec3A;

use super::wgpu::{setup_wgpu, WgpuData};

/// A brute-force [`ComputeMethod`](super::ComputeMethod) using the GPU with [wgpu](https://github.com/gfx-rs/wgpu).
///
/// This struct should not be recreated every iteration in order to maintain performance as it holds initialized data used by WGPU for computing on the GPU.
/// 
/// Currently only available for 3D f32 vectors. You can still use it by converting your 2D f32 vectors to 3D f32 vectors until this is fixed.
pub struct BruteForce {
    device: wgpu::Device,
    queue: wgpu::Queue,
    wgpu_data: Option<WgpuData>,
}

impl super::ComputeMethod<Vec3A, f32> for BruteForce {
    #[inline]
    fn compute(&mut self, particles: &[(Vec3A, f32)]) -> Vec<Vec3A> {
        let massive: Vec<_> = particles
            .iter()
            .copied()
            .filter(|(_, mu)| *mu != 0.0)
            .collect();

        let particles_len = particles.len() as u64;
        let massive_len = massive.len() as u64;

        if massive_len == 0 {
            return Vec::new();
        }

        if let Some(wgpu_data) = &self.wgpu_data {
            if wgpu_data.particle_count != particles_len {
                self.wgpu_data = None;
            }
        }

        self.wgpu_data
            .get_or_insert_with(|| WgpuData::init(particles_len, massive_len, &self.device))
            .write_particle_data(particles, &massive, &self.queue)
            .compute_pass(&self.device, &self.queue)
            .read_accelerations(&self.device)
    }
}

impl BruteForce {
    /// Create a new [`BruteForce`] instance with initialized buffers and pipeline.
    pub fn new_init(massive_count: usize, massless_count: usize) -> Self {
        let (device, queue) = pollster::block_on(setup_wgpu());

        let wgpu_data = Some(WgpuData::init(
            massive_count as u64,
            massless_count as u64,
            &device,
        ));

        Self {
            device,
            queue,
            wgpu_data,
        }
    }
}

impl Default for BruteForce {
    fn default() -> Self {
        let (device, queue) = pollster::block_on(setup_wgpu());

        Self {
            device,
            queue,
            wgpu_data: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(BruteForce::default());
    }
}
