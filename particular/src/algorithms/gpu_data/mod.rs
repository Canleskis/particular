pub struct WgpuResources {
    bind_group: wgpu::BindGroup,
    buffer_affected: wgpu::Buffer,
    buffer_massive: wgpu::Buffer,
    buffer_accelerations: wgpu::Buffer,
    buffer_staging: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
}

impl WgpuResources {
    #[inline]
    pub fn init(affected_size: u64, massive_size: u64, device: &wgpu::Device) -> Self {
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

        queue.submit(Some(encoder.finish()));
    }

    #[inline]
    pub fn write_particle_data<T>(&self, affected: &[T], massive: &[T], queue: &wgpu::Queue)
    where
        T: bytemuck::NoUninit,
    {
        queue.write_buffer(&self.buffer_affected, 0, bytemuck::cast_slice(affected));
        queue.write_buffer(&self.buffer_massive, 0, bytemuck::cast_slice(massive));
    }

    #[inline]
    pub fn read_accelerations<T>(&self, device: &wgpu::Device) -> Vec<T>
    where
        T: bytemuck::AnyBitPattern,
    {
        let buffer = self.buffer_staging.slice(..);
        buffer.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = buffer.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.buffer_staging.unmap();

        result
    }
}
