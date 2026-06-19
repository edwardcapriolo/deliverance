package io.teknek.deliverance.model;

import com.codahale.metrics.Counter;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;

import java.util.Comparator;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Registry for Dropwizard metrics that should appear in benchmark profile summaries.
 *
 * <p>This class does not time code itself. Hot paths should use the returned Dropwizard {@link Timer} or {@link Counter}
 * directly. The helper only records which metric names are part of the focused inference profile so benchmark output does
 * not have to dump every metric in the registry.</p>
 */
public final class InferenceProfiler {
    private static final Set<String> TIMER_NAMES = ConcurrentHashMap.newKeySet();
    private static final Set<String> COUNTER_NAMES = ConcurrentHashMap.newKeySet();
    private static final ConcurrentMap<String, Set<Timer>> TIMERS = new ConcurrentHashMap<>();
    private static final ConcurrentMap<String, Set<Counter>> COUNTERS = new ConcurrentHashMap<>();
    private static final ConcurrentMap<String, TimerBaseline> TIMER_BASELINES = new ConcurrentHashMap<>();
    private static final ConcurrentMap<String, Long> COUNTER_BASELINES = new ConcurrentHashMap<>();
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
        TIMER_NAMES.forEach(name -> TIMER_BASELINES.put(name,
                TimerBaseline.fromTimers(TIMERS.getOrDefault(name, Set.of()))));
        COUNTER_NAMES.forEach(name -> COUNTER_BASELINES.put(name,
                COUNTERS.getOrDefault(name, Set.of()).stream().mapToLong(Counter::getCount).sum()));
    }

    public static Timer timer(MetricRegistry metricRegistry, String name) {
        Timer timer = metricRegistry.timer(name);
        TIMER_NAMES.add(name);
        TIMERS.computeIfAbsent(name, ignored -> Collections.newSetFromMap(new ConcurrentHashMap<>())).add(timer);
        return timer;
    }

    public static Counter counter(MetricRegistry metricRegistry, String name) {
        Counter counter = metricRegistry.counter(name);
        COUNTER_NAMES.add(name);
        COUNTERS.computeIfAbsent(name, ignored -> Collections.newSetFromMap(new ConcurrentHashMap<>())).add(counter);
        return counter;
    }

    public static void printSummary(String label, int maxRows) {
        if (!enabled) {
            return;
        }
        System.out.println("[profile] " + label);
        TIMER_NAMES.stream()
                .map(name -> new TimerSnapshot(name, TIMERS.getOrDefault(name, Set.of())))
                .map(snapshot -> snapshot.minus(TIMER_BASELINES.getOrDefault(snapshot.name(), TimerBaseline.ZERO)))
                .filter(snapshot -> snapshot.count() > 0)
                .sorted(Comparator.comparingDouble(TimerDelta::estimatedTotalNanos).reversed())
                .limit(maxRows)
                .forEach(snapshot -> {
            long count = snapshot.count();
            double meanNanos = snapshot.meanNanos();
            double totalMs = (count * meanNanos) / 1_000_000.0;
            double meanUs = meanNanos / 1_000.0;
            System.out.printf(java.util.Locale.ROOT,
                    "[profile] %-45s count=%8d total_ms=%10.3f mean_us=%10.3f%n",
                    snapshot.name(), count, totalMs, meanUs);
        });
    }

    public static long counterValue(String name) {
        long current = COUNTERS.getOrDefault(name, Set.of()).stream().mapToLong(Counter::getCount).sum();
        return current - COUNTER_BASELINES.getOrDefault(name, 0L);
    }

    public static boolean shouldPrintCounter(String name) {
        return COUNTER_NAMES.contains(name);
    }

    private record TimerSnapshot(String name, Set<Timer> timers) {
        private TimerDelta minus(TimerBaseline baseline) {
            long count = Math.max(0, count() - baseline.count());
            double totalNanos = Math.max(0.0, estimatedTotalNanos() - baseline.estimatedTotalNanos());
            return new TimerDelta(name, count, totalNanos);
        }

        private double estimatedTotalNanos() {
            return timers.stream().mapToDouble(timer -> timer.getCount() * timer.getSnapshot().getMean()).sum();
        }

        private long count() {
            return timers.stream().mapToLong(Timer::getCount).sum();
        }

        private double meanNanos() {
            long count = count();
            return count == 0 ? 0.0 : estimatedTotalNanos() / count;
        }
    }

    private record TimerDelta(String name, long count, double estimatedTotalNanos) {
        private double meanNanos() {
            return count == 0 ? 0.0 : estimatedTotalNanos / count;
        }
    }

    private record TimerBaseline(long count, double estimatedTotalNanos) {
        private static final TimerBaseline ZERO = new TimerBaseline(0, 0.0);

        private static TimerBaseline fromTimers(Set<Timer> timers) {
            long count = timers.stream().mapToLong(Timer::getCount).sum();
            double totalNanos = timers.stream().mapToDouble(timer -> timer.getCount() * timer.getSnapshot().getMean()).sum();
            return new TimerBaseline(count, totalNanos);
        }
    }
}
