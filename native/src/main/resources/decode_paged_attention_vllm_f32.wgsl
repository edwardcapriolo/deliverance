struct Params {
    visible_rows: u32,
    block_size: u32,
    number_of_heads: u32,
    number_of_kv_heads: u32,
    head_size: u32,
    max_blocks_per_seq: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> KCache: array<f32>;
@group(0) @binding(2) var<storage, read> VCache: array<f32>;
@group(0) @binding(3) var<storage, read> BlockTable: array<u32>;
@group(0) @binding(4) var<storage, read_write> State: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> partial: array<f32, 128>;
var<workgroup> scale_weight: array<f32, 3>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head = workgroup_id.x;
    let lane = local_id.x;
    if (head >= params.number_of_heads) {
        return;
    }

    let state_width = 2u + params.head_size;
    let state_base = head * state_width;
    let q_base = head * params.head_size;
    let head_group_size = params.number_of_heads / params.number_of_kv_heads;
    let kv_head = head / head_group_size;

    if (lane == 0u) {
        State[state_base] = -3.4028234663852886e38;
        State[state_base + 1u] = 0.0;
    }
    if (lane < params.head_size) {
        State[state_base + 2u + lane] = 0.0;
    }
    workgroupBarrier();

    var max_val = State[state_base];
    var sum_val = State[state_base + 1u];

    for (var logical_row = 0u; logical_row < params.visible_rows; logical_row = logical_row + 1u) {
        let logical_block = logical_row / params.block_size;
        let block_offset = logical_row % params.block_size;
        let physical_block = BlockTable[logical_block];
        let kv_base = (((physical_block * params.block_size + block_offset) * params.number_of_kv_heads + kv_head) * params.head_size);

        if (lane < params.head_size) {
            partial[lane] = Q[q_base + lane] * KCache[kv_base + lane];
        } else {
            partial[lane] = 0.0;
        }
        workgroupBarrier();

        var stride = 64u;
        loop {
            if (lane < stride && lane + stride < params.head_size) {
                partial[lane] = partial[lane] + partial[lane + stride];
            }
            workgroupBarrier();
            if (stride == 1u) {
                break;
            }
            stride = stride / 2u;
        }

        if (lane == 0u) {
            let score = partial[0] * params.scale;
            let next_max = max(max_val, score);
            var old_scale = 0.0;
            if (max_val > -3.0e38) {
                old_scale = exp(max_val - next_max);
            }
            let weight = exp(score - next_max);
            sum_val = sum_val * old_scale + weight;
            max_val = next_max;
            State[state_base] = max_val;
            State[state_base + 1u] = sum_val;
            scale_weight[0] = old_scale;
            scale_weight[1] = weight;
            scale_weight[2] = sum_val;
        }
        workgroupBarrier();

        if (lane < params.head_size) {
            let out_idx = state_base + 2u + lane;
            State[out_idx] = State[out_idx] * scale_weight[0] + scale_weight[1] * VCache[kv_base + lane];
        }
        workgroupBarrier();
    }
}
