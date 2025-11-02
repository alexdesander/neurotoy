struct Window {
    width: f32,
    height: f32,
};
@group(0) @binding(0)
var<uniform> window: Window;

struct Camera {
    pos: vec2<f32>,
    zoom: f32,
    _pad: f32,
};
@group(1) @binding(0)
var<uniform> camera: Camera;

// new: synapse states
@group(2) @binding(0)
var<storage, read> syn_states: array<u32>;

struct VertexInput {
    @builtin(instance_index) idx: u32,
    // quad vertex in clip-like space
    @location(0) quad_vertex: vec2<f32>,
    // endpoints in world space
    @location(1) end1: vec2<f32>,
    @location(2) end2: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let seg = in.end2 - in.end1;
    let seg_len = length(seg);
    let dir = select(vec2<f32>(1.0, 0.0), seg / seg_len, seg_len > 0.0001);
    let normal = vec2<f32>(-dir.y, dir.x);

    let t = 0.5 * (in.quad_vertex.x + 1.0);
    let half_thickness = 0.0025;

    let base_pos = in.end1 + dir * (t * seg_len);
    let world_pos = base_pos + normal * in.quad_vertex.y * half_thickness;

    let view_pos = (world_pos - camera.pos) * camera.zoom;

    let half_size = vec2<f32>(window.width * 0.5, window.height * 0.5);
    var ndc = view_pos / half_size;
    ndc.y = -ndc.y;

    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);

    // color from synaptic state, same channel for RGB
    let s = syn_states[in.idx];
    if s == 0 {
        out.color = vec4<f32>(0.13, 0.13, 0.13, 0.7);
    } else {
        out.color = vec4<f32>(1.0, 0.0, 0.0, 0.8);
    };
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
