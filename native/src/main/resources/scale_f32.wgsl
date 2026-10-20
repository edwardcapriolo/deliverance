struct Params {
    factor: f32,
    offset: u32,
    length: u32,
    columns: u32,
};

@group(0) @binding(0)
var<storage, read_write> data: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let scaled_index = id.x;
    if (params.offset == 0u && params.length == params.columns) {
        data[scaled_index] = data[scaled_index] * params.factor;
        return;
    }
    let row = scaled_index / params.length;
    let column = params.offset + scaled_index % params.length;
    let element = row * params.columns + column;
    data[element] = data[element] * params.factor;
}
