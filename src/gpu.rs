use crate::vector::SIMD;
use std::borrow::Cow;

// number of boid particles to simulate
// const NUM_PARTICLES: u32 = 1001;

// number of single-particle calculations (invocations) in each gpu work group
const PARTICLES_PER_GROUP: u32 = 128;

async fn _setup_wgpu() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::Backends::all());

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

pub struct Wgpu {
    particle_bind_group: wgpu::BindGroup,
    particle_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    compute_pipeline: wgpu::ComputePipeline,
    work_group_count: u32,
    size: u64,
}

impl Wgpu {
    pub fn init(particle_count: usize, device: &wgpu::Device) -> Self {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
        });

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new((particle_count * 8) as _),
                    },
                    count: None,
                }],
                label: None,
            });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        let size = (particle_count * 16) as _;

        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Buffer"),
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            size,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size,
            mapped_at_creation: false,
        });

        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: particle_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let work_group_count =
            ((particle_count as f32) / (PARTICLES_PER_GROUP as f32)).ceil() as u32;

        Wgpu {
            particle_bind_group,
            particle_buffer,
            staging_buffer,
            compute_pipeline,
            work_group_count,
            size,
        }
    }

    pub fn update(&self, particles: Vec<(SIMD, f32)>, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<[f32; 3]> {
        self.write_back(queue, particles);

        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.push_debug_group("compute accelerations");
        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.particle_bind_group, &[]);
            cpass.dispatch_workgroups(self.work_group_count, 1, 1);
        }
        encoder.pop_debug_group();

        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &self.staging_buffer, 0, self.size);

        queue.submit(Some(encoder.finish()));

        self.read(device)
    }

    fn write_back(&self, queue: &wgpu::Queue, particles: Vec<(SIMD, f32)>) {
        let buffer_data: Vec<f32> = particles
            .iter()
            .flat_map(|particle| {
                let array: [f32; 3] = particle.0.into();
                [array[0], array[1], array[2], particle.1]
            })
            .collect();

        queue.write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(&buffer_data));
    }

    fn read(&self, device: &wgpu::Device) -> Vec<[f32; 3]> {
        let buffer = self.staging_buffer.slice(..);
        buffer.map_async(wgpu::MapMode::Read, |_| {});

        device.poll(wgpu::Maintain::Wait);

        let data = buffer.get_mapped_range();
        let result = bytemuck::cast_slice(&data)
            .chunks(4)
            .map(|r| [r[0], r[1], r[2]])
            .collect();

        drop(data);
        self.staging_buffer.unmap();

        result
    }
}
