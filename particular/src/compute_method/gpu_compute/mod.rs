use ultraviolet::{Vec3, Vec4};

type PointMass = crate::compute_method::storage::PointMass<Vec3, f32>;

/// All the `wgpu` resources needed to perform the computation on the GPU.
pub struct WgpuResources {
    bind_group: wgpu::BindGroup,
    buffer_affected: wgpu::Buffer,
    buffer_massive: wgpu::Buffer,
    buffer_accelerations: wgpu::Buffer,
    buffer_staging: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

impl WgpuResources {
    /// Creates a new [`WgpuResources`] with the given [`wgpu::Device`] and count of affected and massive particles.
    #[inline]
    pub fn init(device: &wgpu::Device, affected_count: usize, massive_count: usize) -> Self {
        let affected_size = (std::mem::size_of::<PointMass>() * affected_count) as u64;
        let massive_size = (std::mem::size_of::<PointMass>() * massive_count) as u64;

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(affected_size),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(massive_size),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(affected_size),
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        let buffer_affected = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Affected buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size: affected_size,
            mapped_at_creation: false,
        });

        let buffer_massive = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Massive buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size: massive_size,
            mapped_at_creation: false,
        });

        let buffer_accelerations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accelerations buffer"),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            size: affected_size,
            mapped_at_creation: false,
        });

        let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: affected_size,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_affected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_massive.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer_accelerations.as_entire_binding(),
                },
            ],
            label: None,
        });

        WgpuResources {
            bind_group,
            buffer_affected,
            buffer_massive,
            buffer_accelerations,
            buffer_staging,
            pipeline,
        }
    }

    /// Performs the computation on the GPU.
    #[inline]
    pub fn compute_pass(&self, device: &wgpu::Device, queue: &wgpu::Queue, work_group_count: u32) {
        let encoder_descriptor = wgpu::CommandEncoderDescriptor { label: None };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);

        encoder.push_debug_group("Compute accelerations");
        {
            let compute_pass_descriptor = wgpu::ComputePassDescriptor { label: None };
            let mut compute_pass = encoder.begin_compute_pass(&compute_pass_descriptor);

            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(work_group_count, 1, 1);
        }
        encoder.pop_debug_group();

        encoder.copy_buffer_to_buffer(
            &self.buffer_accelerations,
            0,
            &self.buffer_staging,
            0,
            self.buffer_accelerations.size(),
        );

        queue.submit([encoder.finish()]);
    }

    /// Write the given affected and massive particles to GPU buffers.
    #[inline]
    pub fn write_particle_data(
        &self,
        affected: &[PointMass],
        massive: &[PointMass],
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(&self.buffer_affected, 0, bytemuck::cast_slice(affected));
        queue.write_buffer(&self.buffer_massive, 0, bytemuck::cast_slice(massive));
    }

    /// Read the accelerations from the corresponding buffer.
    #[inline]
    pub async fn read_accelerations(&self, device: &wgpu::Device) -> Vec<Vec3> {
        let (sender, receiver) = flume::bounded(1);

        let buffer = self.buffer_staging.slice(..);
        buffer.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());

        device.poll(wgpu::Maintain::Wait);
        receiver.recv_async().await.unwrap().unwrap();

        let view = buffer.get_mapped_range();
        // vec3<f32> is 16 byte aligned so we need to cast to a slice of `Vec4`.
        let accelerations = bytemuck::cast_slice(&view)
            .iter()
            .map(Vec4::truncated)
            .collect();

        drop(view);
        self.buffer_staging.unmap();

        accelerations
    }
}
