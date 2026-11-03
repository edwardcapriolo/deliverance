package io.teknek.reckoning.model;

import java.util.Objects;
import java.util.Set;
import java.util.EnumSet;
import java.util.regex.Pattern;

/** Selects runtime trace events and controls how tensor values are captured. */
public record DeadReckoningSelector(Pattern kind, Pattern name, Pattern phase, Pattern position,
        Capture capture, int maxValues, Set<Statistic> statistics) {
    public enum Capture { FIRST_VALUES, FULL, STATS, HASH }
    public enum Statistic {
        SHAPE, DTYPE, COUNT, MIN, MAX, MEAN, SUM, MEAN_ABS, MAX_ABS, L2_NORM, RMS, STDDEV,
        ZERO_COUNT, NAN_COUNT, POSITIVE_INFINITY_COUNT, NEGATIVE_INFINITY_COUNT, HASH, FIRST_VALUES
    }

    public DeadReckoningSelector {
        capture = Objects.requireNonNull(capture, "capture");
        statistics = Set.copyOf(Objects.requireNonNull(statistics, "statistics"));
        if (maxValues < 0) throw new IllegalArgumentException("maxValues must be >= 0");
    }

    public static DeadReckoningSelector regex(String kind, String name, String phase, String position,
            Capture capture, int maxValues) {
        return regex(kind, name, phase, position, capture, maxValues, EnumSet.allOf(Statistic.class));
    }

    public static DeadReckoningSelector regex(String kind, String name, String phase, String position,
            Capture capture, int maxValues, Set<Statistic> statistics) {
        return new DeadReckoningSelector(pattern(kind), pattern(name), pattern(phase), pattern(position), capture,
                maxValues, statistics);
    }

    public static DeadReckoningSelector exact(String kind, String name, String phase, String position,
            Capture capture, int maxValues) {
        return regex(quote(kind), quote(name), quote(phase), quote(position), capture, maxValues);
    }

    boolean matches(String eventKind, String eventName, String eventPhase, int eventPosition) {
        return matches(kind, eventKind) && matches(name, eventName) && matches(phase, eventPhase)
                && matches(position, Integer.toString(eventPosition));
    }

    private static boolean matches(Pattern pattern, String value) {
        return pattern == null || pattern.matcher(value).matches();
    }

    private static Pattern pattern(String value) {
        return value == null ? null : Pattern.compile(value);
    }

    private static String quote(String value) {
        return value == null ? null : Pattern.quote(value);
    }
}
