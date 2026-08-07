struct Params {
    rows: u32,
    head_size: u32,
    kv_offset: u32,
    key_stride: u32,
    value_stride: u32,
    reset: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> Empty: array<f32>;
@group(0) @binding(4) var<storage, read_write> State: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main() {
    if (params.reset != 0u) {
        State[0] = -3.4028234663852886e38;
        State[1] = 0.0;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            State[2u + col] = 0.0;
        }
    }

    var max_val = State[0];
    var sum_val = State[1];

    for (var row = 0u; row < params.rows; row = row + 1u) {
        let key_base = row * params.key_stride + params.kv_offset;
        var score = 0.0;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            score = score + Q[col] * K[key_base + col];
        }
        score = score * params.scale;

        let next_max = max(max_val, score);
        var old_scale = 0.0;
        if (max_val > -3.0e38) {
            old_scale = exp(max_val - next_max);
        }
        let weight = exp(score - next_max);

        let value_base = row * params.value_stride + params.kv_offset;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            let out_idx = 2u + col;
            State[out_idx] = State[out_idx] * old_scale + weight * V[value_base + col];
        }

        sum_val = sum_val * old_scale + weight;
        max_val = next_max;
    }

    State[0] = max_val;
    State[1] = sum_val;
}
