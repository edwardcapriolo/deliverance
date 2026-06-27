package io.teknek.deliverance.model;

/** Logical names for tensor operation implementations that may be available to a model. */
public enum TensorProviderKind {
    GPU,
    SIMD,
    PANAMA,
    NAIVE
}
