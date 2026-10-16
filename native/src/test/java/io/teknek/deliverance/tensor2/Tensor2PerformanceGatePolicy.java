package io.teknek.deliverance.tensor2;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Deterministic policy used by the tensor2 performance-gate behavior tests. */
final class Tensor2PerformanceGatePolicy {
    static final double FIRST_PASS_ALLOWANCE = 1.10;
    static final int MAX_RETRIES = 10;

    private Tensor2PerformanceGatePolicy() {
    }

    static Decision evaluate(Map<String, List<Long>> history, Map<String, Long> current,
            Map<String, List<Long>> retries) {
        List<Suspect> suspects = new ArrayList<>();
        for (Map.Entry<String, Long> entry : current.entrySet()) {
            List<Long> oldValues = history.get(entry.getKey());
            if (oldValues == null || oldValues.isEmpty()) {
                suspects.add(new Suspect(entry.getKey(), List.of(), entry.getValue(), 0));
                continue;
            }
            long maximum = oldValues.stream().mapToLong(Long::longValue).max().orElseThrow();
            if (entry.getValue() > maximum * FIRST_PASS_ALLOWANCE) {
                suspects.add(new Suspect(entry.getKey(), oldValues, entry.getValue(), maximum));
            }
        }

        List<Suspect> failures = new ArrayList<>();
        for (Suspect suspect : suspects) {
            List<Long> attempts = retries.getOrDefault(suspect.key(), List.of());
            boolean recovered = attempts.stream().limit(MAX_RETRIES)
                    .anyMatch(value -> value <= suspect.historicalMaximum());
            if (!recovered) {
                failures.add(suspect.withRetries(attempts));
            }
        }
        return new Decision(suspects, failures);
    }

    record Decision(List<Suspect> suspects, List<Suspect> failures) {
        boolean passed() {
            return failures.isEmpty();
        }
    }

    record Suspect(String key, List<Long> historicalValues, long currentValue, long historicalMaximum,
            List<Long> retryValues) {
        Suspect(String key, List<Long> historicalValues, long currentValue, long historicalMaximum) {
            this(key, historicalValues, currentValue, historicalMaximum, List.of());
        }

        Suspect withRetries(List<Long> values) {
            return new Suspect(key, historicalValues, currentValue, historicalMaximum,
                    values.stream().limit(MAX_RETRIES).toList());
        }
    }
}
