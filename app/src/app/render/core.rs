#![allow(unused)]
use std::sync::Arc;

use wgpu::*;

pub struct RenderCore {
    inner: Arc<RenderCoreInner>,
}

impl RenderCore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RenderCoreInner::new()),
        }
    }

    pub fn instance(&self) -> &Instance {
        &self.inner.instance
    }

    pub fn adapter(&self) -> &Adapter {
        &self.inner.adapter
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn queue(&self) -> &Queue {
        &self.inner.queue
    }
}

struct RenderCoreInner {
    queue: Queue,
    device: Device,
    adapter: Adapter,
    instance: Instance,
}

impl RenderCoreInner {
    pub fn new() -> Self {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        Self {
            queue,
            device,
            adapter,
            instance,
        }
    }
}
