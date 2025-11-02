use std::sync::Arc;

use ::snn::Model;
use wgpu::{
    rwh::{HasDisplayHandle, HasWindowHandle},
    util::DeviceExt,
    *,
};
use winit::window::Window;

use crate::app::render::{
    camera::{Camera, CameraUniform},
    core::RenderCore,
    snn::ModelRenderer,
    window::WindowUniform,
};

pub mod camera;
pub mod core;
pub mod snn;
pub mod window;

pub struct Renderer {
    model_renderer: Option<ModelRenderer>,
    camera_bind_group: BindGroup,
    camera_bind_group_layout: BindGroupLayout,
    camera_buffer: Buffer,
    camera: Camera,
    window_bind_group: BindGroup,
    window_bind_group_layout: BindGroupLayout,
    window_buffer: Buffer,
    surface_config: SurfaceConfiguration,
    surface: Surface<'static>,
    core: RenderCore,
    window: Arc<Window>,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let core = RenderCore::new();

        let surface = unsafe {
            core.instance()
                .create_surface_unsafe(SurfaceTargetUnsafe::RawHandle {
                    raw_display_handle: window.display_handle().unwrap().as_raw(),
                    raw_window_handle: window.window_handle().unwrap().as_raw(),
                })
                .unwrap()
        };
        let surface_caps = surface.get_capabilities(core.adapter());
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f, TextureFormat::Rgba8UnormSrgb))
            .or_else(|| surface_caps.formats.get(0).copied())
            .unwrap_or(TextureFormat::Bgra8UnormSrgb);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::AutoVsync,
            alpha_mode: CompositeAlphaMode::Auto,
            view_formats: vec![surface_format],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(core.device(), &surface_config);

        let window_uniform = WindowUniform {
            width: size.width as f32,
            height: size.height as f32,
        };
        let window_buffer = core
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Window Uniform Buffer"),
                contents: bytemuck::cast_slice(&[window_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let window_bind_group_layout = WindowUniform::bind_group_layout(core.device());
        let window_bind_group = core.device().create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &window_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: window_buffer.as_entire_binding(),
            }],
            label: Some("Window Bind Group"),
        });

        let camera = Camera::default();
        let camera_buffer = core
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[CameraUniform::from(camera)]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let camera_bind_group_layout = CameraUniform::bind_group_layout(core.device());
        let camera_bind_group = core.device().create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        Self {
            window,
            core,
            surface,
            surface_config,
            camera,
            camera_buffer,
            camera_bind_group_layout,
            camera_bind_group,
            window_buffer,
            window_bind_group_layout,
            window_bind_group,
            model_renderer: None,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface
                .configure(self.core.device(), &self.surface_config);
            self.core.queue().write_buffer(
                &self.window_buffer,
                0,
                bytemuck::cast_slice(&[WindowUniform {
                    width: width as f32,
                    height: height as f32,
                }]),
            );
        }
    }

    pub fn set_model(&mut self, model: &Model) {
        if self.model_renderer.is_none() {
            self.model_renderer = Some(ModelRenderer::new(
                self.core.clone(),
                self.surface_config.format,
                &self.window_bind_group_layout,
                &self.camera_bind_group_layout,
                model,
            ));
        } else {
            self.model_renderer.as_mut().unwrap().relayout(model);
        }
    }

    pub fn update_model(&mut self, model: &Model) {
        self.model_renderer.as_mut().unwrap().update_model(model);
    }

    pub fn update(&mut self) {
        // Reupload camera every update cause why not
        self.core.queue().write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[CameraUniform::from(self.camera)]),
        );
    }

    pub fn render(&mut self) {
        self.update();
        match self.render_inner() {
            Ok(_) => {}
            // Reconfigure the surface if it's lost or outdated
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = self.window.inner_size();
                self.resize(size.width, size.height);
            }
            Err(e) => {
                eprintln!("Unable to render {}", e);
            }
        }
    }

    fn render_inner(&mut self) -> Result<(), SurfaceError> {
        self.window.request_redraw();
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());
        let mut encoder = self
            .core
            .device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            if let Some(model_renderer) = &mut self.model_renderer {
                model_renderer.render(
                    &self.window_bind_group,
                    &self.camera_bind_group,
                    &mut render_pass,
                );
            }
        }

        self.core.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
}
