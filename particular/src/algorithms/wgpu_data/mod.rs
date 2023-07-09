pub(crate) struct WgpuData {
    bind_group: wgpu::BindGroup,
    buffer_particles: wgpu::Buffer,
    buffer_massive: wgpu::Buffer,
    buffer_accelerations: wgpu::Buffer,
    buffer_staging: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    size_buffer_particles: u64,
    pub particle_count: u64,
    pub work_group_count: u32,
}

impl WgpuData {
    #[inline]
    pub fn init(
        particle_size: u64,
        particle_count: u64,
        massive_count: u64,
        device: &wgpu::Device,
    ) -> Self {
        let size_buffer_particles = particle_count * particle_size;
        let size_buffer_massive = massive_count * particle_size;

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_buffer_particles),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_buffer_massive),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(size_buffer_particles),
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        let buffer_particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size: size_buffer_particles,
            mapped_at_creation: false,
        });

        let buffer_massive = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Massive particle buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size: size_buffer_massive,
            mapped_at_creation: false,
        });

        let buffer_accelerations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accelerations buffer"),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            size: size_buffer_particles,
            mapped_at_creation: false,
        });

        let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: size_buffer_particles,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer_particles.as_entire_binding(),
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

        let work_group_count = ((particle_count as f32) / 256.0).ceil() as u32;

        WgpuData {
            bind_group,
            buffer_particles,
            buffer_massive,
            buffer_accelerations,
            buffer_staging,
            compute_pipeline,
            size_buffer_particles,
            particle_count,
            work_group_count,
        }
    }

    #[inline]
    pub fn compute_pass(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let encoder_descriptor = wgpu::CommandEncoderDescriptor { label: None };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);

        encoder.push_debug_group("Compute accelerations");
        {
            let compute_pass_descriptor = wgpu::ComputePassDescriptor { label: None };
            let mut compute_pass = encoder.begin_compute_pass(&compute_pass_descriptor);

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(self.work_group_count, 1, 1);
        }
        encoder.pop_debug_group();

        // TODO: Potentially wrong buffer size.
        encoder.copy_buffer_to_buffer(
            &self.buffer_accelerations,
            0,
            &self.buffer_staging,
            0,
            self.size_buffer_particles,
        );

        queue.submit(Some(encoder.finish()));
    }

    #[inline]
    pub fn write_particle_data<T: bytemuck::Pod>(
        &self,
        particles: &[T],
        massive: &[T],
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(&self.buffer_massive, 0, bytemuck::cast_slice(massive));
        queue.write_buffer(&self.buffer_particles, 0, bytemuck::cast_slice(particles));
    }

    #[inline]
    pub fn read_accelerations<T: bytemuck::Pod>(&self, device: &wgpu::Device) -> Vec<T> {
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

pub(crate) async fn setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
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
