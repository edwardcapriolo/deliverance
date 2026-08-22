package io.teknek.deliverance.model.tensorparallel;

import java.util.Locale;

public enum TensorParallelAssignmentMode {
    AUTOMATIC,
    MANUAL;

    public static TensorParallelAssignmentMode fromString(String value) {
        if (value == null || value.isBlank()) {
            return AUTOMATIC;
        }
        return TensorParallelAssignmentMode.valueOf(value.trim().toUpperCase(Locale.ROOT));
    }
}
