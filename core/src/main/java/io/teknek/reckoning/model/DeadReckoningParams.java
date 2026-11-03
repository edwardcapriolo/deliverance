package io.teknek.reckoning.model;

import java.nio.file.Path;
import java.util.Objects;
import java.util.List;

/** Collect-all runtime tracing configuration for Dead Reckoning diagnostics. */
public record DeadReckoningParams(Path output, int traceMaxValues, List<DeadReckoningSelector> selectors) {
    public DeadReckoningParams(Path output, int traceMaxValues) {
        this(output, traceMaxValues, List.of());
    }

    public DeadReckoningParams {
        Objects.requireNonNull(output, "output");
        selectors = List.copyOf(Objects.requireNonNull(selectors, "selectors"));
        if (traceMaxValues < 0) {
            throw new IllegalArgumentException("traceMaxValues must be >= 0");
        }
    }
}
