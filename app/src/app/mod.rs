use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use snn::Model;
use winit::{
    event::{DeviceEvent, DeviceId, MouseScrollDelta, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::Key,
    window::{Window, WindowAttributes, WindowId},
};
use winit_input_manager::{InputManager, LogicalEvent};

use crate::app::render::Renderer;

pub mod render;

pub struct App {
    last_update: Instant,
    input: InputManager,
    renderer: Renderer,

    model: Model,
    last_model_tick: Instant,
}

impl App {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(Self::create_window(event_loop));
        let mut renderer = Renderer::new(window);
        let input = InputManager::new();

        // TEMPORARY
        let model = Model::grid(20, 20);
        renderer.set_model(&model);

        Self {
            last_update: Instant::now(),
            input,
            renderer,
            model,

            last_model_tick: Instant::now(),
        }
    }

    pub fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.input.process_window_event(&event);
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                self.renderer.resize(new_size.width, new_size.height);
            }
            WindowEvent::RedrawRequested => {
                self.update();
                self.renderer.render();
            }
            _ => {}
        }
    }

    pub fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.input.process_device_event(&event);
    }

    fn create_window(event_loop: &ActiveEventLoop) -> Window {
        event_loop
            .create_window(WindowAttributes::default().with_title("Neurotoy"))
            .unwrap()
    }

    fn update(&mut self) {
        const ZOOM_FACTOR_PER_LINE_SCROLLED: f32 = 0.1;
        const CAMERA_MOVE_SPEED: f32 = 400.0;
        const MODEL_TICK_INTERVAL: Duration = Duration::from_millis(500);

        let dt = self.last_update.elapsed().as_secs_f32();
        self.last_update = Instant::now();

        let is_key_down =
            |s: &'static str| self.input.is_logical_key_pressed(Key::Character(s.into()));

        // Process held
        if is_key_down("a") {
            let camera = self.renderer.camera_mut();
            camera.center_x -= CAMERA_MOVE_SPEED * dt / camera.zoom;
        }
        if is_key_down("d") {
            let camera = self.renderer.camera_mut();
            camera.center_x += CAMERA_MOVE_SPEED * dt / camera.zoom;
        }
        if is_key_down("w") {
            let camera = self.renderer.camera_mut();
            camera.center_y -= CAMERA_MOVE_SPEED * dt / camera.zoom;
        }
        if is_key_down("s") {
            let camera = self.renderer.camera_mut();
            camera.center_y += CAMERA_MOVE_SPEED * dt / camera.zoom;
        }

        // Process events
        for event in self.input.iter_logical_events() {
            match event.event {
                LogicalEvent::MouseWheelScrolled(delta) => match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.renderer.camera_mut().zoom *=
                            1.0 - ZOOM_FACTOR_PER_LINE_SCROLLED * -y as f32;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        for event in self.input.iter_physical_events() {
            match event.event {
                _ => {}
            }
        }

        if self.last_model_tick.elapsed() >= MODEL_TICK_INTERVAL {
            self.last_model_tick = Instant::now();
            self.tick_model();
        }
    }

    fn tick_model(&mut self) {
        println!("Tick model");
        self.model.tick();
        self.renderer.update_model(&self.model);
    }
}
