package io.teknek.deliverance.model;

/** Selects the one-token decode attention algorithm. */
public enum DecodeAttentionMode {
    /** Existing staged path: QK scores, materialized softmax row, then V accumulation. */
    STAGED,

    /** Decode-only online-softmax path that avoids materializing the full attention row. */
    FLASH_DECODE
}
