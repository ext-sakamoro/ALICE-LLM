// KV-outer sparse attention forward — one workgroup per (edge × qhead lane)
// partial. Mirrors the CPU `process_work_unit` in
// `src/sparse_attention/forward.rs` and matches its numerical semantics
// (natural-log-space online softmax, right-aligned causal is unimplemented
// in this MVP — CPU host must set `causal=0`).
//
// Layout expectations:
//   * `edge_meta[i]` is 5 i32s per absolute edge:
//       [q_idx, h, blk, batch, block_len]
//     `block_len` is precomputed on CPU from `used_kv_lens` clamped by the
//     block, so the shader does not need a separate `used_kv_lens` binding.
//   * `block_tables[b, p]` = i32 physical page id (or -1 for unmapped)
//   * `k_pages` / `v_pages` layout = [num_pages, hkv, page_size, head_dim]
//   * `q_buf` layout = [Tq, hq, head_dim]
//
// Output partial layout matches `ForwardPartials`:
//   * `o_partial[partial_idx * head_dim + d]`
//   * `m_partial[partial_idx]`
//   * `l_partial[partial_idx]`
// where `partial_idx = abs_edge * qhead + qh`.

struct Params {
    total_partials: u32,   // total_edges * qhead
    qhead: u32,
    head_dim: u32,
    block_size: u32,
    page_size: u32,
    pages_per_block: u32,
    hkv: u32,
    hq: u32,
    msb: u32,
    max_pages: u32,
    softmax_scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_buf: array<f32>;
@group(0) @binding(2) var<storage, read> k_pages: array<f32>;
@group(0) @binding(3) var<storage, read> v_pages: array<f32>;
@group(0) @binding(4) var<storage, read> edge_meta: array<i32>;
@group(0) @binding(5) var<storage, read> block_tables: array<i32>;

@group(0) @binding(6) var<storage, read_write> o_partial: array<f32>;
@group(0) @binding(7) var<storage, read_write> m_partial: array<f32>;
@group(0) @binding(8) var<storage, read_write> l_partial: array<f32>;

const WG_SIZE: u32 = 64u;
const MAX_BLOCK_SIZE: u32 = 128u;
const NEG_LARGE: f32 = -1.0e30;

var<workgroup> scores: array<f32, MAX_BLOCK_SIZE>;
var<workgroup> reduce_buf: array<f32, WG_SIZE>;
var<workgroup> broadcast_m: f32;
var<workgroup> broadcast_l: f32;

@compute @workgroup_size(64)
fn sparse_forward(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let partial_idx = wg.x;
    let tid = lid.x;

    if (partial_idx >= params.total_partials) {
        return;
    }

    let abs_edge = partial_idx / params.qhead;
    let qh = partial_idx % params.qhead;

    // Read 5-i32 edge metadata.
    let meta_off = abs_edge * 5u;
    let q_idx = u32(edge_meta[meta_off + 0u]);
    let h = u32(edge_meta[meta_off + 1u]);
    let blk = u32(edge_meta[meta_off + 2u]);
    let batch = u32(edge_meta[meta_off + 3u]);
    let block_len_i = edge_meta[meta_off + 4u];
    let block_len: u32 = select(u32(0), u32(block_len_i), block_len_i > 0);

    let hq_abs = h * params.qhead + qh;
    let q_base = (q_idx * params.hq + hq_abs) * params.head_dim;

    // --- Init scores[0..MAX_BLOCK_SIZE] with NEG_LARGE so the reduction
    //     never trips on stale values from a previous dispatch. ------------
    var init_i = tid;
    while (init_i < MAX_BLOCK_SIZE) {
        scores[init_i] = NEG_LARGE;
        init_i = init_i + WG_SIZE;
    }
    workgroupBarrier();

    // Early bail: block is entirely masked. Thread 0 writes sentinel partial.
    if (block_len == 0u) {
        if (tid == 0u) {
            m_partial[partial_idx] = NEG_LARGE;
            l_partial[partial_idx] = 0.0;
            var d0: u32 = 0u;
            while (d0 < params.head_dim) {
                o_partial[partial_idx * params.head_dim + d0] = 0.0;
                d0 = d0 + 1u;
            }
        }
        return;
    }

    // --- Phase 1: compute Q · K per position (stride WG_SIZE) ------------
    var t = tid;
    while (t < block_len) {
        let page_local = t / params.page_size;
        let pos_in_page = t - page_local * params.page_size;
        let page_slot = blk * params.pages_per_block + page_local;
        let page_id_i = block_tables[batch * params.max_pages + page_slot];
        var s: f32 = NEG_LARGE;
        if (page_id_i >= 0) {
            let page_id = u32(page_id_i);
            let page_base = page_id * params.hkv * params.page_size * params.head_dim
                          + h * params.page_size * params.head_dim
                          + pos_in_page * params.head_dim;
            var dot_val: f32 = 0.0;
            for (var d = 0u; d < params.head_dim; d = d + 1u) {
                dot_val = dot_val + q_buf[q_base + d] * k_pages[page_base + d];
            }
            s = dot_val * params.softmax_scale;
        }
        scores[t] = s;
        t = t + WG_SIZE;
    }
    workgroupBarrier();

    // --- Phase 2a: parallel max reduction over scores[0..block_len] ------
    var local_max: f32 = NEG_LARGE;
    var mt = tid;
    while (mt < block_len) {
        let cand = scores[mt];
        if (cand > local_max) {
            local_max = cand;
        }
        mt = mt + WG_SIZE;
    }
    reduce_buf[tid] = local_max;
    workgroupBarrier();
    var step: u32 = WG_SIZE / 2u;
    while (step > 0u) {
        if (tid < step) {
            let other = reduce_buf[tid + step];
            if (other > reduce_buf[tid]) {
                reduce_buf[tid] = other;
            }
        }
        workgroupBarrier();
        step = step >> 1u;
    }
    if (tid == 0u) {
        broadcast_m = reduce_buf[0];
    }
    workgroupBarrier();
    let m_val = broadcast_m;

    // --- Phase 2b: softmax weights + parallel sum reduction --------------
    var local_sum: f32 = 0.0;
    var st = tid;
    while (st < block_len) {
        var p: f32 = 0.0;
        let raw = scores[st];
        if (raw > NEG_LARGE * 0.5) {
            p = exp(raw - m_val);
        }
        scores[st] = p;      // repurpose scores as softmax weights
        local_sum = local_sum + p;
        st = st + WG_SIZE;
    }
    reduce_buf[tid] = local_sum;
    workgroupBarrier();
    step = WG_SIZE / 2u;
    while (step > 0u) {
        if (tid < step) {
            reduce_buf[tid] = reduce_buf[tid] + reduce_buf[tid + step];
        }
        workgroupBarrier();
        step = step >> 1u;
    }
    if (tid == 0u) {
        broadcast_l = reduce_buf[0];
    }
    workgroupBarrier();
    let l_val = broadcast_l;

    // --- Phase 3: weighted sum V per output dim (stride WG_SIZE) ---------
    var d = tid;
    while (d < params.head_dim) {
        var acc: f32 = 0.0;
        for (var t3 = 0u; t3 < block_len; t3 = t3 + 1u) {
            let w = scores[t3];
            if (w == 0.0) {
                continue;
            }
            let page_local = t3 / params.page_size;
            let pos_in_page = t3 - page_local * params.page_size;
            let page_slot = blk * params.pages_per_block + page_local;
            let page_id_i = block_tables[batch * params.max_pages + page_slot];
            if (page_id_i >= 0) {
                let page_id = u32(page_id_i);
                let page_base = page_id * params.hkv * params.page_size * params.head_dim
                              + h * params.page_size * params.head_dim
                              + pos_in_page * params.head_dim;
                acc = acc + w * v_pages[page_base + d];
            }
        }
        o_partial[partial_idx * params.head_dim + d] = acc;
        d = d + WG_SIZE;
    }

    if (tid == 0u) {
        m_partial[partial_idx] = m_val;
        l_partial[partial_idx] = l_val;
    }
}
