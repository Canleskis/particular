use ultraviolet::{Vec3, Vec4};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

struct DynamicBuffer {
    buffer: wgpu::Buffer,
    label: Option<String>,
}

impl DynamicBuffer {
    fn new(device: &wgpu::Device, descriptor: &wgpu::BufferDescriptor) -> Self {
        DynamicBuffer {
            buffer: device.create_buffer(descriptor),
            label: descriptor.label.map(String::from),
        }
    }

    fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, contents: &[u8]) {
        if self.buffer.size() == contents.len() as wgpu::BufferAddress {
            queue.write_buffer(&self.buffer, 0, contents);
        } else {
            self.buffer = device.create_buffer_init(&BufferInitDescriptor {
                contents,
                label: self.label.as_deref(),
                usage: self.buffer.usage(),
            });
        }
    }

    fn resize(&mut self, device: &wgpu::Device, size: wgpu::BufferAddress) {
        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size,
            mapped_at_creation: false,
            label: self.label.as_deref(),
            usage: self.buffer.usage(),
        });
    }

    fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    fn size(&self) -> wgpu::BufferAddress {
        self.buffer.size()
    }
}

type PointMass = crate::compute_method::storage::PointMass<Vec3, f32>;

const PARTICLE_SIZE: u64 = std::mem::size_of::<PointMass>() as u64;

/// All the `wgpu` resources needed to perform the computation of accelerations on the GPU.
pub struct WgpuResources {
    bind_group_layout: wgpu::BindGroupLayout,
    buffer_affected: DynamicBuffer,
    buffer_massive: DynamicBuffer,
    buffer_accelerations: DynamicBuffer,
    pipeline: wgpu::ComputePipeline,
    workgroup_size: u32,
}

impl WgpuResources {
    /// Creates a new [`WgpuResources`] with the given [`wgpu::Device`].
    #[inline]
    pub fn new(device: &wgpu::Device, workgroup_size: u32) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let buffer_affected = DynamicBuffer::new(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Affected buffer"),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size: 0,
                mapped_at_creation: false,
            },
        );

        let buffer_massive = DynamicBuffer::new(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Massive buffer"),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size: 0,
                mapped_at_creation: false,
            },
        );

        let buffer_accelerations = DynamicBuffer::new(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Accelerations buffer"),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                size: 0,
                mapped_at_creation: false,
            },
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                include_str!("compute.wgsl")
                    .replace("R_WORKGROUP_SIZE", &(workgroup_size.to_string() + "u"))
                    .into(),
            ),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        WgpuResources {
            bind_group_layout,
            buffer_affected,
            buffer_massive,
            buffer_accelerations,
            pipeline,
            workgroup_size,
        }
    }

    /// Write the given affected and massive particles to GPU buffers.
    #[inline]
    pub fn write_particle_data(
        &mut self,
        affected: &[PointMass],
        massive: &[PointMass],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let affected = bytemuck::cast_slice(affected);
        self.buffer_affected.write(device, queue, affected);

        let massive = bytemuck::cast_slice(massive);
        self.buffer_massive.write(device, queue, massive);

        let size = affected.len() as wgpu::BufferAddress;
        self.buffer_accelerations.resize(device, size);
    }

    /// Returns the computed accelerations on the GPU.
    #[inline]
    pub async fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Vec3> {
        let affected_count = self.buffer_affected.size() / PARTICLE_SIZE;
        let massive_count = self.buffer_massive.size() / PARTICLE_SIZE;

        if affected_count == 0 {
            return Vec::new();
        }

        if massive_count == 0 {
            return vec![Vec3::zero(); affected_count as usize];
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer_affected.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffer_massive.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffer_accelerations.buffer().as_entire_binding(),
                },
            ],
            label: None,
        });

        let encoder_descriptor = wgpu::CommandEncoderDescriptor { label: None };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);

        encoder.push_debug_group("Compute accelerations");
        {
            let workgroups = (affected_count as f32 / self.workgroup_size as f32).ceil() as u32;
            let compute_pass_descriptor = wgpu::ComputePassDescriptor { label: None };
            let mut compute_pass = encoder.begin_compute_pass(&compute_pass_descriptor);
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.pop_debug_group();

        let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: self.buffer_accelerations.size(),
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            self.buffer_accelerations.buffer(),
            0,
            &buffer_staging,
            0,
            self.buffer_accelerations.size(),
        );

        queue.submit([encoder.finish()]);

        let (sender, receiver) = flume::bounded(1);

        let buffer = buffer_staging.slice(..);
        buffer.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());

        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv_async()
            .await
            .unwrap()
            .expect("Could not read buffer");

        let view = buffer.get_mapped_range();
        // vec3<f32> is 16 byte aligned so we need to cast to a slice of `Vec4`.
        let accelerations = bytemuck::cast_slice(&view)
            .iter()
            .map(Vec4::truncated)
            .collect();

        drop(view);
        buffer_staging.unmap();

        accelerations
    }
}
