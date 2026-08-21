package io.teknek.deliverance.model.tensorparallel;

import java.time.Duration;
import java.util.Objects;

public record TensorParallelTimeoutSettings(
        Duration rankConnectTimeout,
        Duration rankRequestTimeout,
        Duration rankOperationTimeout,
        Duration rankCloseTimeout) {
    public static final TensorParallelTimeoutSettings DEFAULT = new TensorParallelTimeoutSettings(
            Duration.ofSeconds(5), Duration.ofSeconds(30), Duration.ofSeconds(60), Duration.ofSeconds(10));

    public TensorParallelTimeoutSettings {
        rankConnectTimeout = requirePositive(rankConnectTimeout, "rankConnectTimeout");
        rankRequestTimeout = requirePositive(rankRequestTimeout, "rankRequestTimeout");
        rankOperationTimeout = requirePositive(rankOperationTimeout, "rankOperationTimeout");
        rankCloseTimeout = requirePositive(rankCloseTimeout, "rankCloseTimeout");
    }

    public static TensorParallelTimeoutSettings ofSeconds(long rankConnectTimeoutSeconds,
            long rankRequestTimeoutSeconds, long rankOperationTimeoutSeconds, long rankCloseTimeoutSeconds) {
        return new TensorParallelTimeoutSettings(Duration.ofSeconds(rankConnectTimeoutSeconds),
                Duration.ofSeconds(rankRequestTimeoutSeconds), Duration.ofSeconds(rankOperationTimeoutSeconds),
                Duration.ofSeconds(rankCloseTimeoutSeconds));
    }

    private static Duration requirePositive(Duration duration, String name) {
        Objects.requireNonNull(duration, name);
        if (duration.isZero() || duration.isNegative()) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return duration;
    }
}
