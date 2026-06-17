package io.teknek.deliverance.model;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.LongAdder;
import java.util.function.Supplier;

/** Lightweight accumulated stage profiler for inference hot paths. */
public final class InferenceProfiler {
    private static final ConcurrentMap<String, Stats> STATS = new ConcurrentHashMap<>();
    private static volatile boolean enabled = Boolean.getBoolean("deliverance.profile.stages");

    private InferenceProfiler() {
    }

    public static void setEnabled(boolean enabled) {
        InferenceProfiler.enabled = enabled;
    }

    public static boolean isEnabled() {
        return enabled;
    }

    public static void reset() {
        STATS.clear();
    }

    public static <T> T time(String name, Supplier<T> supplier) {
        if (!enabled) {
            return supplier.get();
        }
        long start = System.nanoTime();
        try {
            return supplier.get();
        } finally {
            record(name, System.nanoTime() - start);
        }
    }

    public static void record(String name, long nanos) {
        Stats stats = STATS.computeIfAbsent(name, ignored -> new Stats());
        stats.count.increment();
        stats.nanos.add(nanos);
    }

    public static List<Snapshot> snapshotsByTotalTime() {
        return STATS.entrySet().stream()
                .map(entry -> new Snapshot(entry.getKey(), entry.getValue().count.sum(), entry.getValue().nanos.sum()))
                .sorted(Comparator.comparingLong(Snapshot::totalNanos).reversed())
                .toList();
    }

    public static void printSummary(String label, int maxRows) {
        if (!enabled) {
            return;
        }
        System.out.println("[profile] " + label);
        snapshotsByTotalTime().stream().limit(maxRows).forEach(snapshot -> {
            double totalMs = snapshot.totalNanos / 1_000_000.0;
            double meanUs = snapshot.count == 0 ? 0.0 : snapshot.totalNanos / 1_000.0 / snapshot.count;
            System.out.printf(java.util.Locale.ROOT,
                    "[profile] %-45s count=%8d total_ms=%10.3f mean_us=%10.3f%n",
                    snapshot.name, snapshot.count, totalMs, meanUs);
        });
    }

    public static boolean shouldPrintCounter(String name) {
        return name.endsWith(".input_dtype.F32")
                || name.endsWith(".input_dtype.I8")
                || name.endsWith(".input_dtype.BF16");
    }

    private static final class Stats {
        private final LongAdder count = new LongAdder();
        private final LongAdder nanos = new LongAdder();
    }

    public record Snapshot(String name, long count, long totalNanos) {
    }
}
