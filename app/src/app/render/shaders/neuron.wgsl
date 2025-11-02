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


@group(2) @binding(0)
var<storage, read> neuron_vs: array<f32>;

struct VertexInput {
    @builtin(instance_index) idx: u32,
    @location(0) quad_vertex: vec2<f32>,
    @location(1) neuron_center: vec2<f32>,
    @location(2) neuron_radius: f32,
    @location(3) _pad: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) circle_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

// ----------- COLOR CALCULATION
const COLOR_MIN_VAL : f32 = -1.0;
const COLOR_MID_VAL : f32 =  0.0;
const COLOR_MAX_VAL : f32 =  1.0;

const COLOR_MIN_RGB : vec3<f32> = vec3<f32>(0.0, 0.0, 1.0); // blue
const COLOR_MID_RGB : vec3<f32> = vec3<f32>(0.13, 0.13, 0.13); // gray
const COLOR_MAX_RGB : vec3<f32> = vec3<f32>(1.0, 0.0, 0.0); // red

const INV_MIN_MID : f32 = 1.0 / (COLOR_MID_VAL - COLOR_MIN_VAL); // 1/3
const INV_MID_MAX : f32 = 1.0 / (COLOR_MAX_VAL - COLOR_MID_VAL); // 1/5

fn value_to_color(v: f32) -> vec4<f32> {
    // Branchless selection: s = 0 for v<=0, 1 for v>0
    let s = step(0.0, v);

    // t0: for [-3,0]
    let t0 = clamp((v - COLOR_MIN_VAL) * INV_MIN_MID, 0.0, 1.0);
    let col0 = mix(COLOR_MIN_RGB, COLOR_MID_RGB, t0);

    // t1: for [0,5]
    let t1 = clamp((v - COLOR_MID_VAL) * INV_MID_MAX, 0.0, 1.0);
    let col1 = mix(COLOR_MID_RGB, COLOR_MAX_RGB, t1);

    // Pick col0 when v<=0, col1 when v>0
    let rgb = mix(col0, col1, s);

    return vec4<f32>(rgb, 1.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // circle in world space
    let world_pos = in.neuron_center + in.quad_vertex * in.neuron_radius;

    // camera: translate then scale
    let view_pos = (world_pos - camera.pos) * camera.zoom;

    // pixel -> NDC
    let half_size = vec2<f32>(window.width * 0.5, window.height * 0.5);
    var ndc = view_pos / half_size;
    // wgpu surface has origin at top-left, NDC has +y up
    ndc.y = -ndc.y;

    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    out.circle_pos = in.quad_vertex; // length(local) == 1.0 on circle
    out.color = value_to_color(neuron_vs[in.idx]);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let d = length(in.circle_pos);
    if d > 1.0 {
        discard;
    }

    return in.color;
}
