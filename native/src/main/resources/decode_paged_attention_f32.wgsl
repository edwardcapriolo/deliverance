struct Params {
    rows: u32,
    head_size: u32,
    number_of_heads: u32,
    number_of_kv_heads: u32,
    kv_length: u32,
    key_base: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> QK: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read> EmptyB: array<f32>;
@group(0) @binding(3) var<storage, read> EmptyB2: array<f32>;
@group(0) @binding(4) var<storage, read_write> Out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let head = workgroup_id.x;
    if (head >= params.number_of_heads) {
        return;
    }
    let head_group_size = params.number_of_heads / params.number_of_kv_heads;
    let kv_head = head / head_group_size;
    let q_base = head * params.head_size;
    let kv_offset = kv_head * params.head_size;
    let out_base = head * params.head_size;

    var max_val = -3.4028234663852886e38;
    var sum_val = 0.0;

    for (var col = 0u; col < params.head_size; col = col + 1u) {
        Out[out_base + col] = 0.0;
    }

    for (var row = 0u; row < params.rows; row = row + 1u) {
        let row_base = params.key_base + row * params.kv_length + kv_offset;
        var score = 0.0;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            score = score + QK[q_base + col] * QK[row_base + col];
        }
        score = score * params.scale;

        let next_max = max(max_val, score);
        var old_scale = 0.0;
        if (max_val > -3.0e38) {
            old_scale = exp(max_val - next_max);
        }
        let weight = exp(score - next_max);

        let v_base = row * params.kv_length + kv_offset;
        for (var col = 0u; col < params.head_size; col = col + 1u) {
            let out_idx = out_base + col;
            Out[out_idx] = Out[out_idx] * old_scale + weight * V[v_base + col];
        }
        sum_val = sum_val * old_scale + weight;
        max_val = next_max;
    }

    for (var col = 0u; col < params.head_size; col = col + 1u) {
        let out_idx = out_base + col;
        Out[out_idx] = Out[out_idx] / sum_val;
    }
}
