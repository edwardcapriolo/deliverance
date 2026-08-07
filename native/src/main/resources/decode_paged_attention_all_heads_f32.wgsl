struct Params {
    rows: u32,
    head_size: u32,
    number_of_heads: u32,
    number_of_kv_heads: u32,
    kv_length: u32,
    key_stride: u32,
    value_stride: u32,
    reset: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> Empty: array<f32>;
@group(0) @binding(4) var<storage, read_write> State: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let head = workgroup_id.x;
    if (head >= params.number_of_heads) {
        return;
    }

    let state_width = 2u + params.head_size;
    let state_base = head * state_width;
    let q_base = head * params.head_size;
    let head_group_size = params.number_of_heads / params.number_of_kv_heads;
    let kv_head = head / head_group_size;
    let kv_offset = kv_head * params.head_size;

    if (params.reset != 0u) {
        State[state_base] = -3.4028234663852886e38;
        State[state_base + 1u] = 0.0;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            State[state_base + 2u + col] = 0.0;
        }
    }

    var max_val = State[state_base];
    var sum_val = State[state_base + 1u];

    for (var row = 0u; row < params.rows; row = row + 1u) {
        let key_base = row * params.key_stride + kv_offset;
        var score = 0.0;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            score = score + Q[q_base + col] * K[key_base + col];
        }
        score = score * params.scale;

        let next_max = max(max_val, score);
        var old_scale = 0.0;
        if (max_val > -3.0e38) {
            old_scale = exp(max_val - next_max);
        }
        let weight = exp(score - next_max);

        let value_base = row * params.value_stride + kv_offset;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            let out_idx = state_base + 2u + col;
            State[out_idx] = State[out_idx] * old_scale + weight * V[value_base + col];
        }

        sum_val = sum_val * old_scale + weight;
        max_val = next_max;
    }

    State[state_base] = max_val;
    State[state_base + 1u] = sum_val;
}
