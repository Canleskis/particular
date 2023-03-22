use std::borrow::Cow;

use glam::Vec3A;

pub(crate) struct WgpuData {
    bind_group: wgpu::BindGroup,
    buffer_particles: wgpu::Buffer,
    buffer_massive_particles: wgpu::Buffer,
    buffer_accelerations: wgpu::Buffer,
    buffer_staging: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    pub work_group_count: u32,
    pub particle_count: u64,
}

impl WgpuData {
    #[inline]
    pub fn init(particle_count: u64, massive_count: u64, device: &wgpu::Device) -> Self {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
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
                            min_binding_size: wgpu::BufferSize::new(particle_count * 4 * 4),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(massive_count * 4 * 4),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(particle_count * 4 * 4),
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
            size: particle_count * 4 * 4,
            mapped_at_creation: false,
        });

        let buffer_massive_particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Massive particle buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            size: massive_count * 4 * 4,
            mapped_at_creation: false,
        });

        let buffer_accelerations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Accelerations buffer"),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            size: particle_count * 4 * 4,
            mapped_at_creation: false,
        });

        let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: particle_count * 4 * 4,
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
                    resource: buffer_massive_particles.as_entire_binding(),
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
            buffer_massive_particles,
            buffer_accelerations,
            buffer_staging,
            compute_pipeline,
            work_group_count,
            particle_count,
        }
    }

    #[inline]
    pub fn compute_pass(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> &Self {
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

        encoder.copy_buffer_to_buffer(
            &self.buffer_accelerations,
            0,
            &self.buffer_staging,
            0,
            self.particle_count * 4 * 4,
        );

        queue.submit(Some(encoder.finish()));

        self
    }

    #[inline]
    pub fn write_particle_data(
        &self,
        particles: &[(Vec3A, f32)],
        massive: &[(Vec3A, f32)],
        queue: &wgpu::Queue,
    ) -> &Self {
        let massive_data: Vec<f32> = massive
            .iter()
            .flat_map(|point_mass| {
                let [x, y, z]: [f32; 3] = point_mass.0.into();
                [x, y, z, point_mass.1]
            })
            .collect();

        let particles_data: Vec<f32> = particles
            .iter()
            .flat_map(|point_mass| {
                let [x, y, z]: [f32; 3] = point_mass.0.into();
                [x, y, z, point_mass.1]
            })
            .collect();

        queue.write_buffer(
            &self.buffer_massive_particles,
            0,
            bytemuck::cast_slice(&massive_data),
        );

        queue.write_buffer(
            &self.buffer_particles,
            0,
            bytemuck::cast_slice(&particles_data),
        );

        self
    }

    #[inline]
    pub fn read_accelerations(&self, device: &wgpu::Device) -> Vec<Vec3A> {
        let buffer = self.buffer_staging.slice(..);
        buffer.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = buffer.get_mapped_range();

        // Because of alignment rules each vec3<f32> in array is 16 bytes (even though they technically are 12 bytes, 4 bytes for the 3 f32s).
        // We discard the extra value.
        let result = bytemuck::cast_slice(&data)
            .chunks(4)
            .map(|slice| Vec3A::new(slice[0], slice[1], slice[2]))
            .collect();

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
