package io.teknek.reckoning.tensor2;

/** Selects who, if anyone, schedules independent GEMM work. */
public enum Threading {
    NONE,
    PROVIDER
}
