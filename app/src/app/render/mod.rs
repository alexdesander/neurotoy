use std::sync::Arc;

use wgpu::{
    rwh::{HasDisplayHandle, HasWindowHandle},
    *,
};
use winit::window::Window;

use crate::app::render::core::RenderCore;

pub mod core;

pub struct Renderer {
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

        Self {
            window,
            core,
            surface,
            surface_config,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;
            self.surface
                .configure(self.core.device(), &self.surface_config);
        }
    }

    pub fn render(&mut self) {
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
            let _render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
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
        }

        self.core.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
