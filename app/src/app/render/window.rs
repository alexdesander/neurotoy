use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroupLayout, Device};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(super) struct WindowUniform {
    pub width: f32,
    pub height: f32,
}

impl WindowUniform {
    pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: None,
        })
    }
}
