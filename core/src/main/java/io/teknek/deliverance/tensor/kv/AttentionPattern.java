package io.teknek.deliverance.tensor.kv;

/** Visibility pattern requested by attention when reading from a KV cache session. */
public enum AttentionPattern {
    CAUSAL,
    BIDIRECTIONAL,
    PREFIX_CAUSAL_PLUS_BIDIRECTIONAL_BLOCK
}
