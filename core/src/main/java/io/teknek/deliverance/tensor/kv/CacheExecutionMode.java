package io.teknek.deliverance.tensor.kv;

/** Explicit cache mutation mode for a model forward phase. */
public enum CacheExecutionMode {
    PREFILL_UPDATE_CACHE,
    DECODE_UPDATE_CACHE,
    READ_PREFIX_NO_UPDATE,
    DENOISE_BLOCK_NO_UPDATE,
    VERIFY_AND_UPDATE_CACHE
}
