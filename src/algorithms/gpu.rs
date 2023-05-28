use crate::{
    algorithms::{
        wgpu_data::{setup_wgpu, WgpuData},
        FromMassive, PointMass, Scalar,
    },
    compute_method::{ComputeMethod, Storage},
};

const PARTICLE_SIZE: u64 = std::mem::size_of::<PointMass<[f32; 3], f32>>() as u64;

/// A brute-force [`ComputeMethod`](ComputeMethod) using the GPU with [wgpu](https://github.com/gfx-rs/wgpu).
///
/// This struct should not be recreated every iteration in order to maintain performance as it holds initialized data used by WGPU for computing on the GPU.
///
/// Currently only available for 3D f32 vectors. You can still use it in 2D by converting your 2D f32 vectors to 3D f32 vectors until this is fixed.
pub struct BruteForce {
    device: ::wgpu::Device,
    queue: ::wgpu::Queue,
    wgpu_data: Option<WgpuData>,
}

impl<V> ComputeMethod<FromMassive<[f32; 3], f32>, V> for &mut BruteForce
where
    V: From<[f32; 3]> + 'static,
{
    type Output = Box<dyn Iterator<Item = V>>;

    #[inline]
    fn compute(self, storage: FromMassive<[f32; 3], f32>) -> Self::Output {
        let particles_len = storage.affected.len() as u64;
        let massive_len = storage.massive.len() as u64;

        if massive_len == 0 {
            return Box::new(storage.affected.into_iter().map(|_| V::from([0.0; 3])));
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

        Box::new(
            wgpu_data
                .read_accelerations(&self.device)
                .into_iter()
                // 1 byte padding between each vec3<f32>.
                .map(|slice: [f32; 4]| V::from([slice[0], slice[1], slice[2]])),
        )
    }
}

impl BruteForce {
    /// Create a new [`BruteForce`] instance.
    pub fn new() -> Self {
        let (device, queue) = pollster::block_on(setup_wgpu());

        Self {
            device,
            queue,
            wgpu_data: None,
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
            device,
            queue,
            wgpu_data,
        }
    }
}

impl Default for BruteForce {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, S, const DIM: usize> Storage<PointMass<V, S>> for FromMassive<[S; DIM], S>
where
    V: Into<[S; DIM]> + 'static,
    S: Scalar + 'static,
{
    #[inline]
    fn new(input: impl Iterator<Item = PointMass<V, S>>) -> Self {
        Self::from(input.map(PointMass::into))
    }
}

unsafe impl<S, const DIM: usize> bytemuck::Zeroable for super::PointMass<[S; DIM], S> {}
unsafe impl<S: bytemuck::Pod, const DIM: usize> bytemuck::Pod for super::PointMass<[S; DIM], S> {}

#[cfg(test)]
mod tests {
    use super::super::tests;
    use super::*;

    #[test]
    fn brute_force() {
        tests::acceleration_computation(&mut BruteForce::default());
    }
}
