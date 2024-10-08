struct DynamicBuffer {
    buffer: wgpu::Buffer,
    label: Option<String>,
    item_size: u64,
}

impl DynamicBuffer {
    #[inline]
    fn new(device: &wgpu::Device, descriptor: &wgpu::BufferDescriptor, item_size: u64) -> Self {
        Self {
            buffer: device.create_buffer(descriptor),
            label: descriptor.label.map(String::from),
            item_size,
        }
    }

    #[inline]
    fn write_with<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &'a wgpu::Queue,
        len: u64,
    ) -> wgpu::QueueWriteBufferView {
        assert_ne!(len, 0, "Cannot write with a length of 0");

        let size = len * self.item_size;
        if self.buffer.size() != size {
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                size,
                label: self.label.as_deref(),
                usage: self.buffer.usage(),
                mapped_at_creation: false,
            });
        }

        // The buffer is guaranteed to be the correct size at this point.
        queue
            .write_buffer_with(&self.buffer, 0, size.try_into().unwrap())
            .unwrap()
    }

    #[inline]
    fn resize(&mut self, device: &wgpu::Device, len: u64) {
        let size = len * self.item_size;
        if self.buffer.size() == size {
            return;
        }

        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            size,
            mapped_at_creation: false,
            label: self.label.as_deref(),
            usage: self.buffer.usage(),
        });
    }

    #[inline]
    const fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    #[inline]
    fn size(&self) -> wgpu::BufferAddress {
        self.buffer.size()
    }

    #[inline]
    fn len(&self) -> u64 {
        self.size() / self.item_size
    }
}

/// Defines the way memory for affecting particles is accessed in the compute shader.
#[derive(Debug, Clone, Copy)]
pub enum MemoryStrategy {
    /// Uses shared memory to store the affecting particles and speed up memory access within one
    /// workgroup. Based on <https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda>.
    Shared(u32),
    /// Uses global memory to store and access the affecting particles.
    Global(u32),
}

impl Default for MemoryStrategy {
    #[inline]
    fn default() -> Self {
        Self::Global(256)
    }
}

impl MemoryStrategy {
    /// Returns the processed shader as an allocated string for the given [`MemoryStrategy`].
    #[inline]
    pub fn as_shader(&self) -> String {
        let (concat, workgroup_size) = match self {
            Self::Shared(workgroup_size) => {
                (include_str!("bruteforce_shared.wgsl"), workgroup_size)
            }
            Self::Global(workgroup_size) => (include_str!("bruteforce.wgsl"), workgroup_size),
        };

        concat.replace("#WORKGROUP_SIZE", &(workgroup_size.to_string() + "u"))
    }

    /// Returns the workgroup size for the shader of this [`MemoryStrategy`].
    #[inline]
    pub const fn workgroup_size(&self) -> u32 {
        match self {
            Self::Global(workgroup_size) | Self::Shared(workgroup_size) => *workgroup_size,
        }
    }
}

/// All the `wgpu` resources needed to perform the computation on the GPU.
pub struct WgpuResources {
    buffer_affected: DynamicBuffer,
    buffer_affecting: DynamicBuffer,
    buffer_output: DynamicBuffer,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    workgroup_size: u32,
}

impl WgpuResources {
    /// Creates a new [`WgpuResources`] with the given [`wgpu::Device`].
    #[inline]
    pub fn new(
        shader: &str,
        device: &wgpu::Device,
        shader_type: MemoryStrategy,
        push_constant_ranges: &[wgpu::PushConstantRange],
        affected_size: u64,
        affecting_size: u64,
        output_size: u64,
    ) -> Self {
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
            affected_size,
        );

        let buffer_affecting = DynamicBuffer::new(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Affecting buffer"),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
                size: 0,
                mapped_at_creation: false,
            },
            affecting_size,
        );

        let buffer_output = DynamicBuffer::new(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Output buffer"),
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                size: 0,
                mapped_at_creation: false,
            },
            output_size,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges,
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl((shader_type.as_shader() + shader).into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let workgroup_size = shader_type.workgroup_size();

        Self {
            buffer_affected,
            buffer_affecting,
            buffer_output,
            bind_group_layout,
            pipeline,
            workgroup_size,
        }
    }

    /// Write the affected particles to the affected buffer.
    #[inline]
    pub fn write_affected_with<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &'a wgpu::Queue,
        len: u64,
    ) -> wgpu::QueueWriteBufferView {
        self.buffer_output.resize(device, len);
        self.buffer_affected.write_with(device, queue, len)
    }

    /// Write the affecting particles to the affecting buffer.
    #[inline]
    pub fn write_affecting_with<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &'a wgpu::Queue,
        len: u64,
    ) -> wgpu::QueueWriteBufferView {
        self.buffer_affecting.write_with(device, queue, len)
    }

    /// Returns the computed values.
    #[inline]
    pub async fn compute<Output>(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        push_data: &[u8],
        mut read_buffer: impl FnMut(&wgpu::BufferView) -> Vec<Output>,
    ) -> Vec<Output>
    where
        Output: Default,
    {
        let affected_count = self.buffer_affected.len();

        if affected_count == 0 {
            return Vec::new();
        }

        if self.buffer_affecting.size() == 0 {
            return std::iter::repeat_with(Default::default)
                .take(affected_count as usize)
                .collect();
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
                    resource: self.buffer_affecting.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buffer_output.buffer().as_entire_binding(),
                },
            ],
            label: None,
        });

        let encoder_descriptor = wgpu::CommandEncoderDescriptor { label: None };
        let mut encoder = device.create_command_encoder(&encoder_descriptor);

        encoder.push_debug_group("Compute interactions");
        {
            let workgroups = (affected_count as f32 / self.workgroup_size as f32).ceil() as u32;
            let compute_pass_descriptor = wgpu::ComputePassDescriptor::default();
            let mut compute_pass = encoder.begin_compute_pass(&compute_pass_descriptor);
            compute_pass.set_pipeline(&self.pipeline);
            if !push_data.is_empty() {
                compute_pass.set_push_constants(0, push_data);
            }
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.pop_debug_group();

        let buffer_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            size: self.buffer_output.size(),
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            self.buffer_output.buffer(),
            0,
            &buffer_staging,
            0,
            self.buffer_output.size(),
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

        let output = read_buffer(&view);

        drop(view);
        buffer_staging.unmap();

        output
    }
}
