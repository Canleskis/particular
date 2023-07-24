use crate::{
    algorithms::{
        vector,
        wgpu_data::{setup_wgpu, WgpuData},
        MassiveAffectedArray, PointMass,
    },
    compute_method::ComputeMethod,
};

const PARTICLE_SIZE: u64 = std::mem::size_of::<PointMass<[f32; 3], f32>>() as u64;

/// Brute-force [`ComputeMethod`] using the GPU with [wgpu](https://github.com/gfx-rs/wgpu).
///
/// This struct should not be recreated every iteration for performance reasons as it holds initialized data used by WGPU for computing on the GPU.
///
/// Currently only available for 3D f32 vectors. You can still use it in 2D by converting your 2D f32 vectors to 3D f32 vectors.
pub struct BruteForce {
    wgpu_data: Option<WgpuData>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<V> ComputeMethod<MassiveAffectedArray<3, f32, V>, V> for &mut BruteForce
where
    V: vector::ConvertArray<3, f32, Array = [f32; 3]>,
{
    type Output = Vec<V>;

    #[inline]
    fn compute(self, storage: &MassiveAffectedArray<3, f32, V>) -> Self::Output {
        let storage = &storage.0;
        let particles_len = storage.affected.len() as u64;
        let massive_len = storage.massive.len() as u64;

        if massive_len == 0 {
            return storage.affected.iter().map(|_| V::from([0.0; 3])).collect();
        }

        if let Some(wgpu_data) = &self.wgpu_data {
            if wgpu_data.particle_count != particles_len {
                self.wgpu_data = None;
            }
        }

        let wgpu_data = self.wgpu_data.get_or_insert_with(|| {
            WgpuData::init(PARTICLE_SIZE, particles_len, massive_len, &self.device)
        });

        wgpu_data.write_particle_data(&storage.affected, &storage.massive, &self.queue);
        wgpu_data.compute_pass(&self.device, &self.queue);

        wgpu_data
            .read_accelerations(&self.device)
            .iter()
            // 1 byte padding between each vec3<f32>.
            .map(|acc: &[f32; 4]| V::from([acc[0], acc[1], acc[2]]))
            .collect()
    }
}

impl BruteForce {
    /// Create a new [`BruteForce`] instance.
    pub fn new() -> Self {
        let (device, queue) = pollster::block_on(setup_wgpu());

        Self {
            wgpu_data: None,
            device,
            queue,
        }
    }

    /// Create a new [`BruteForce`] instance with initialized buffers and pipeline.
    pub fn new_init(particle_count: usize, massive_count: usize) -> Self {
        let (device, queue) = pollster::block_on(setup_wgpu());

        let wgpu_data = Some(WgpuData::init(
            PARTICLE_SIZE,
            particle_count as u64,
            massive_count as u64,
            &device,
        ));

        Self {
            wgpu_data,
            device,
            queue,
        }
    }
}

impl Default for BruteForce {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<S, const D: usize> bytemuck::Zeroable for super::PointMass<[S; D], S> {}
unsafe impl<S: bytemuck::Pod, const D: usize> bytemuck::Pod for super::PointMass<[S; D], S> {}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(BruteForce::new(), 1e-2);
        tests::circular_orbit_stability(BruteForce::new(), 100, 1e-2);
    }
}
