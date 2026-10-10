package io.teknek.deliverance.tensor2;

import java.util.concurrent.ForkJoinPool;

/** One-token causal attention operation over page-backed KV tensors. */
public record DecodePagedAttention(TensorRef output, TensorRef query, TensorRef[] keyPages,
        TensorRef[] valuePages, int visibleRows, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
        float scale, Float softcap, ForkJoinPool pool, int headSplitSize) {
}
