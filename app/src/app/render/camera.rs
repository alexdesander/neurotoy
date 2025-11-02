use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroupLayout, Device};

#[derive(Clone, Copy)]
pub struct Camera {
    pub center_x: f32,
    pub center_y: f32,
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            center_x: 0.0,
            center_y: 0.0,
            zoom: 20.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct CameraUniform {
    pub center_x: f32,
    pub center_y: f32,
    pub zoom: f32,
    pub _pad: f32,
}

impl From<Camera> for CameraUniform {
    fn from(camera: Camera) -> Self {
        CameraUniform {
            center_x: camera.center_x,
            center_y: camera.center_y,
            zoom: camera.zoom,
            _pad: 0.0,
        }
    }
}

impl CameraUniform {
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
