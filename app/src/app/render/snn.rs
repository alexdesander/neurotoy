use bytemuck::{Pod, Zeroable};
use snn::Model;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    *,
};

use crate::app::render::core::RenderCore;

/// A struct that holds a cpu and gpu side buffer and synchronizes them.
pub(super) struct SyncedBuffer<T: Zeroable + Pod> {
    cpu_buf: Vec<T>,
    gpu_buf: Buffer,
    /// In size_of::<T>()
    gpu_len: usize,
    /// In size_of::<T>()
    gpu_cap: usize,
    needs_sync: bool,
    buffer_usages: BufferUsages,
}

impl<T: Zeroable + Pod> SyncedBuffer<T> {
    pub fn new(device: &Device, buffer_usages: BufferUsages) -> Self {
        Self {
            cpu_buf: Vec::new(),
            gpu_buf: device.create_buffer(&BufferDescriptor {
                label: None,
                size: 0,
                usage: buffer_usages | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            gpu_len: 0,
            gpu_cap: 0,
            needs_sync: false,
            buffer_usages,
        }
    }

    pub fn cpu_mut(&mut self) -> &mut Vec<T> {
        self.needs_sync = true;
        &mut self.cpu_buf
    }

    pub fn gpu_maybe_sync(&mut self, device: &Device, queue: &Queue) -> (&Buffer, usize) {
        if self.needs_sync {
            self.sync(device, queue);
        }
        (&self.gpu_buf, self.gpu_len)
    }

    fn sync(&mut self, device: &Device, queue: &Queue) {
        if self.gpu_cap < self.cpu_buf.len() {
            self.gpu_cap = self.cpu_buf.len().next_power_of_two();
            self.gpu_buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size: self.gpu_cap as u64 * std::mem::size_of::<T>() as u64,
                usage: self.buffer_usages | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.gpu_buf, 0, bytemuck::cast_slice(&self.cpu_buf));
        self.gpu_len = self.cpu_buf.len();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub(super) struct Vertex([f32; 2]);

impl Vertex {
    pub const QUAD_VERTICES: [Vertex; 4] = [
        Vertex([-1.0, -1.0]),
        Vertex([1.0, -1.0]),
        Vertex([1.0, 1.0]),
        Vertex([-1.0, 1.0]),
    ];
    pub const QUAD_INDICES: [u16; 6] = [0, 1, 2, 0, 2, 3];

    pub const ATTRIBS: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x2];
    pub fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub(super) struct NeuronInstance {
    pub center: [f32; 2],
    pub radius: f32,
    pub _pad: f32,
}

impl NeuronInstance {
    pub const ATTRIBS: [VertexAttribute; 3] =
        vertex_attr_array![1 => Float32x2, 2 => Float32, 3 => Float32];
    pub fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub(super) struct SynapseInstance {
    pub end1: [f32; 2],
    pub end2: [f32; 2],
}

impl SynapseInstance {
    pub const ATTRIBS: [VertexAttribute; 2] = vertex_attr_array![1 => Float32x2, 2 => Float32x2];
    pub fn desc() -> VertexBufferLayout<'static> {
        use std::mem;
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct ModelRenderer {
    core: RenderCore,

    quad_vertices: Buffer,
    quad_indices: Buffer,

    neurons: SyncedBuffer<NeuronInstance>,
    neuron_vs: SyncedBuffer<f32>,
    neuron_vs_bind_group_layout: BindGroupLayout,
    neuron_vs_bind_group: BindGroup,
    neuron_render_pipeline: RenderPipeline,

    synapses: SyncedBuffer<SynapseInstance>,
    synapse_states: SyncedBuffer<u32>,
    synapse_states_bind_group_layout: BindGroupLayout,
    synapse_states_bind_group: BindGroup,
    synapse_render_pipeline: RenderPipeline,
}

impl ModelRenderer {
    pub fn new(
        core: RenderCore,
        surface_format: TextureFormat,
        window_bind_group_layout: &BindGroupLayout,
        camera_bind_group_layout: &BindGroupLayout,
        model: &Model,
    ) -> Self {
        let quad_vertices = core.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&Vertex::QUAD_VERTICES),
            usage: BufferUsages::VERTEX,
        });
        let quad_indices = core.device().create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&Vertex::QUAD_INDICES),
            usage: BufferUsages::INDEX,
        });

        let neurons = SyncedBuffer::<NeuronInstance>::new(core.device(), BufferUsages::VERTEX);
        let mut neuron_vs = SyncedBuffer::<f32>::new(core.device(), BufferUsages::STORAGE);
        neuron_vs.cpu_mut().push(0.0);
        let neuron_vs_bind_group_layout =
            core.device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Neuron Vertex Shader Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let neuron_vs_bind_group = core.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Neuron Vertex Shader Bind Group"),
            layout: &neuron_vs_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: neuron_vs
                    .gpu_maybe_sync(core.device(), core.queue())
                    .0
                    .as_entire_binding(),
            }],
        });
        let neuron_shader = core
            .device()
            .create_shader_module(wgpu::include_wgsl!("shaders/neuron.wgsl"));
        let neuron_render_pipeline_layout =
            core.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        &window_bind_group_layout,
                        &camera_bind_group_layout,
                        &neuron_vs_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                    ..Default::default()
                });
        let neuron_render_pipeline =
            core.device()
                .create_render_pipeline(&RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&neuron_render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &neuron_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Vertex::desc(), NeuronInstance::desc()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &neuron_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: surface_format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Cw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        let synapses = SyncedBuffer::<SynapseInstance>::new(core.device(), BufferUsages::VERTEX);
        let mut synapse_states = SyncedBuffer::<u32>::new(core.device(), BufferUsages::STORAGE);
        synapse_states.cpu_mut().push(0);
        let synapse_states_bind_group_layout =
            core.device()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Synapse Vertex Shader Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let synapse_states_bind_group = core.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Synapse Vertex Shader Bind Group"),
            layout: &synapse_states_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: synapse_states
                    .gpu_maybe_sync(core.device(), core.queue())
                    .0
                    .as_entire_binding(),
            }],
        });
        let synapse_shader = core
            .device()
            .create_shader_module(wgpu::include_wgsl!("shaders/synapse.wgsl"));
        let synapse_render_pipeline_layout =
            core.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: &[
                        &window_bind_group_layout,
                        &camera_bind_group_layout,
                        &synapse_states_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                    ..Default::default()
                });
        let synapse_render_pipeline =
            core.device()
                .create_render_pipeline(&RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&synapse_render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &synapse_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[Vertex::desc(), SynapseInstance::desc()],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &synapse_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: surface_format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Cw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        let mut s = Self {
            core,
            quad_vertices,
            quad_indices,
            neurons,
            neuron_vs,
            neuron_vs_bind_group_layout,
            neuron_vs_bind_group,
            neuron_render_pipeline,
            synapses,
            synapse_states,
            synapse_states_bind_group_layout,
            synapse_states_bind_group,
            synapse_render_pipeline,
        };
        s.relayout(model);
        s
    }

    pub fn update_model(&mut self, model: &Model) {
        let neuron_vs = self.neuron_vs.cpu_mut();
        neuron_vs.clear();
        neuron_vs.extend(model.neuron_vs());
        self.neuron_vs.sync(self.core.device(), self.core.queue());

        let synapse_states = self.synapse_states.cpu_mut();
        synapse_states.clear();
        synapse_states.extend(model.synapse_states());
        self.synapse_states
            .sync(self.core.device(), self.core.queue());
    }

    pub fn relayout(&mut self, model: &Model) {
        let (layout_neurons, layout_synapses) = snn::render::layout_graph(model);
        let neurons = self.neurons.cpu_mut();
        neurons.clear();
        for neuron in layout_neurons {
            neurons.push(NeuronInstance {
                center: neuron.center,
                radius: neuron.radius,
                _pad: 0.0,
            });
        }
        let synapses = self.synapses.cpu_mut();
        synapses.clear();
        for synapse in layout_synapses {
            synapses.push(SynapseInstance {
                end1: synapse.end1,
                end2: synapse.end2,
            });
        }

        // Rebuild neuron charges storage buffer
        let neuron_vs = self.neuron_vs.cpu_mut();
        neuron_vs.clear();
        neuron_vs.extend(model.neuron_vs());
        self.neuron_vs_bind_group = self.core.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Neuron vs bind group"),
            layout: &self.neuron_vs_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: self
                        .neuron_vs
                        .gpu_maybe_sync(self.core.device(), self.core.queue())
                        .0,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // Rebuild synapse states bind group
        let synapse_states = self.synapse_states.cpu_mut();
        synapse_states.clear();
        synapse_states.extend(model.synapse_states());
        self.synapse_states_bind_group =
            self.core.device().create_bind_group(&BindGroupDescriptor {
                label: Some("Synapse states bind group"),
                layout: &self.synapse_states_bind_group_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: self
                            .synapse_states
                            .gpu_maybe_sync(self.core.device(), self.core.queue())
                            .0,
                        offset: 0,
                        size: None,
                    }),
                }],
            });
    }

    pub fn render(
        &mut self,
        window_bind_group: &BindGroup,
        camera_bind_group: &BindGroup,
        rp: &mut RenderPass,
    ) {
        // Render synapses
        let (synapse_buffer, synapse_count) = self
            .synapses
            .gpu_maybe_sync(self.core.device(), self.core.queue());
        if synapse_count > 0 {
            rp.set_pipeline(&self.synapse_render_pipeline);
            rp.set_bind_group(0, window_bind_group, &[]);
            rp.set_bind_group(1, camera_bind_group, &[]);
            rp.set_bind_group(2, &self.synapse_states_bind_group, &[]);
            rp.set_vertex_buffer(0, self.quad_vertices.slice(..));
            rp.set_vertex_buffer(1, synapse_buffer.slice(..));
            rp.set_index_buffer(self.quad_indices.slice(..), IndexFormat::Uint16);
            rp.draw_indexed(
                0..Vertex::QUAD_INDICES.len() as u32,
                0,
                0..synapse_count as u32,
            );
        }

        // Render neurons
        let (neuron_buffer, neuron_count) = self
            .neurons
            .gpu_maybe_sync(self.core.device(), self.core.queue());
        if neuron_count > 0 {
            rp.set_pipeline(&self.neuron_render_pipeline);
            rp.set_bind_group(0, window_bind_group, &[]);
            rp.set_bind_group(1, camera_bind_group, &[]);
            rp.set_bind_group(2, &self.neuron_vs_bind_group, &[]);
            rp.set_vertex_buffer(0, self.quad_vertices.slice(..));
            rp.set_vertex_buffer(1, neuron_buffer.slice(..));
            rp.set_index_buffer(self.quad_indices.slice(..), IndexFormat::Uint16);
            rp.draw_indexed(
                0..Vertex::QUAD_INDICES.len() as u32,
                0,
                0..neuron_count as u32,
            );
        }
    }
}
