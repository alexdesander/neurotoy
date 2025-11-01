use std::sync::Arc;

use winit::{
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes, WindowId},
};

use crate::app::render::Renderer;

pub mod render;

pub struct App {
    renderer: Renderer,
}

impl App {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(Self::create_window(event_loop));
        let renderer = Renderer::new(window);
        Self { renderer }
    }

    pub fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.renderer.resize(new_size.width, new_size.height);
            }
            WindowEvent::RedrawRequested => {
                self.renderer.render();
            }
            _ => {}
        }
    }

    fn create_window(event_loop: &ActiveEventLoop) -> Window {
        event_loop
            .create_window(WindowAttributes::default().with_title("Neurotoy"))
            .unwrap()
    }
}
