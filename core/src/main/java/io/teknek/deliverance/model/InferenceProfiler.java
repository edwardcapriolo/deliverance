package io.teknek.deliverance.model;

import com.codahale.metrics.Counter;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Snapshot;
import com.codahale.metrics.Timer;
import com.codahale.metrics.ExponentiallyDecayingReservoir;

import java.util.Comparator;
import java.util.Collections;
import java.util.concurrent.TimeUnit;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Registry for Dropwizard metrics that should appear in benchmark profile summaries.
 *
 * <p>Hot paths should use the returned Dropwizard-compatible {@link Timer} or {@link Counter} directly. Timers are also
 * recorded into exact cumulative count/nanosecond totals so benchmark profile summaries can print per-reset deltas without
 * trying to reset Dropwizard process-lifetime metrics.</p>
 */
public final class InferenceProfiler {
    private static final Set<String> TIMER_NAMES = ConcurrentHashMap.newKeySet();
    private static final Set<String> COUNTER_NAMES = ConcurrentHashMap.newKeySet();
    private static final ConcurrentMap<String, Set<Counter>> COUNTERS = new ConcurrentHashMap<>();
    private static final ConcurrentMap<String, TimingTotals> TIMER_TOTALS = new ConcurrentHashMap<>();
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
        TIMER_NAMES.forEach(name -> TIMER_BASELINES.put(name, baseline(name)));
        COUNTER_NAMES.forEach(name -> COUNTER_BASELINES.put(name,
                COUNTERS.getOrDefault(name, Set.of()).stream().mapToLong(Counter::getCount).sum()));
    }

    public static Timer timer(MetricRegistry metricRegistry, String name) {
        TIMER_NAMES.add(name);
        TimingTotals totals = TIMER_TOTALS.computeIfAbsent(name, ignored -> new TimingTotals());
        return new ProfilingTimer(metricRegistry.timer(name), totals);
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
                .map(name -> timerDelta(name, TIMER_BASELINES.getOrDefault(name, TimerBaseline.ZERO)))
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

    private record TimerDelta(String name, long count, double estimatedTotalNanos) {
        private double meanNanos() {
            return count == 0 ? 0.0 : estimatedTotalNanos / count;
        }
    }

    private record TimerBaseline(long count, double estimatedTotalNanos) {
        private static final TimerBaseline ZERO = new TimerBaseline(0, 0.0);
    }

    private static TimerBaseline baseline(String name) {
        TimingTotals totals = TIMER_TOTALS.get(name);
        return totals == null ? TimerBaseline.ZERO : new TimerBaseline(totals.count(), totals.totalNanos());
    }

    private static TimerDelta timerDelta(String name, TimerBaseline baseline) {
        TimingTotals totals = TIMER_TOTALS.get(name);
        if (totals == null) {
            return new TimerDelta(name, 0, 0.0);
        }
        return new TimerDelta(name, Math.max(0, totals.count() - baseline.count()),
                Math.max(0.0, totals.totalNanos() - baseline.estimatedTotalNanos()));
    }

    private static final class TimingTotals {
        private final LongAdder count = new LongAdder();
        private final LongAdder totalNanos = new LongAdder();

        private void update(long duration, TimeUnit unit) {
            count.increment();
            totalNanos.add(unit.toNanos(duration));
        }

        private long count() {
            return count.sum();
        }

        private long totalNanos() {
            return totalNanos.sum();
        }
    }

    private static final class ProfilingTimer extends Timer {
        private final Timer delegate;
        private final TimingTotals totals;

        private ProfilingTimer(Timer delegate, TimingTotals totals) {
            super(new ExponentiallyDecayingReservoir());
            this.delegate = delegate;
            this.totals = totals;
        }

        @Override
        public void update(long duration, TimeUnit unit) {
            delegate.update(duration, unit);
            totals.update(duration, unit);
        }

        @Override
        public long getCount() {
            return delegate.getCount();
        }

        @Override
        public Snapshot getSnapshot() {
            return delegate.getSnapshot();
        }
    }
}
