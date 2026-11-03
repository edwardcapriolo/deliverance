package io.teknek.reckoning.tensor2;

/** Selects the provider implementation family for a batch dot product. */
public enum GemmKernel {
    BATCH,
    ROW,
    CHUNK
}
