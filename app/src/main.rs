use mimalloc::MiMalloc;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

use crate::app::App;

mod app;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = WinitWrapper::default();
    event_loop.run_app(&mut app).unwrap();
}

#[derive(Default)]
struct WinitWrapper {
    app: Option<App>,
}

impl ApplicationHandler for WinitWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_none() {
            self.app = Some(App::new(event_loop));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(app) = self.app.as_mut() {
            app.window_event(event_loop, window_id, event);
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(app) = self.app.as_mut() {
            app.device_event(event_loop, device_id, event);
        }
    }
}
