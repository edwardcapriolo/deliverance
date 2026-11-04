package io.teknek.reckoning.model;

import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.reckoning.JsonUtils;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

/** Exact semantic equality checks for newline-delimited Dead Reckoning traces. */
public final class DeadReckoningTraceCheck {
    private static final int MAX_REPORTED_DIFFERENCES = 12;

    private DeadReckoningTraceCheck() {
    }

    /** Compares two traces, printing a concise difference report when they are not identical. */
    public static boolean same(File expected, File actual) {
        return same(expected.toPath(), actual.toPath());
    }

    /** Compares two traces, printing a concise difference report when they are not identical. */
    public static boolean same(Path expected, Path actual) {
        TraceFiles left = read(expected);
        TraceFiles right = read(actual);
        List<String> differences = differences(left.records, right.records);
        if (!differences.isEmpty()) {
            System.out.println(summary(expected, actual, left.size(), right.size(), differences));
            return false;
        }
        return true;
    }

    /** Compares a trace with the newest older trace for the same model and hostname directory. */
    public static boolean sameAsPrevious(File current) {
        return sameAsPrevious(current.toPath());
    }

    /** Compares a trace with the newest older trace for the same model and hostname directory. */
    public static boolean sameAsPrevious(Path current) {
        Path previous = previousTrace(current);
        if (previous == null) {
            throw new IllegalArgumentException("No previous Dead Reckoning trace found for " + current);
        }
        return same(previous, current);
    }

    /** Returns the difference report without printing it. */
    public static String explain(Path expected, Path actual) {
        TraceFiles left = read(expected);
        TraceFiles right = read(actual);
        List<String> differences = differences(left.records, right.records);
        return differences.isEmpty()
                ? "Dead Reckoning traces are identical: " + expected + " and " + actual
                : summary(expected, actual, left.records.size(), right.records.size(), differences);
    }

    private static Path previousTrace(Path current) {
        Path absolute = current.toAbsolutePath();
        Path directory = absolute.getParent();
        if (directory == null) return null;
        String fileName = absolute.getFileName().toString();
        int timestampSeparator = fileName.lastIndexOf('-');
        if (timestampSeparator < 0) return null;
        String prefix = fileName.substring(0, timestampSeparator + 1);
        try (Stream<Path> paths = Files.list(directory)) {
            return paths.filter(Files::isRegularFile)
                    .filter(path -> !path.equals(absolute))
                    .filter(path -> path.getFileName().toString().startsWith(prefix))
                    .filter(path -> path.getFileName().toString().endsWith(".jsonl"))
                    .max(Comparator.comparing(Path::getFileName))
                    .orElse(null);
        } catch (IOException e) {
            throw new IllegalStateException("Unable to find previous Dead Reckoning trace near " + current, e);
        }
    }

    private static TraceFiles read(Path path) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Dead Reckoning trace does not exist: " + path);
        }
        Map<String, List<JsonNode>> records = new LinkedHashMap<>();
        try {
            for (String line : Files.readAllLines(path)) {
                if (line.isBlank()) continue;
                JsonNode record = JsonUtils.om.readTree(line);
                records.computeIfAbsent(key(record), ignored -> new ArrayList<>()).add(record);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read Dead Reckoning trace " + path, e);
        }
        records.values().forEach(values -> values.sort(Comparator.comparing(JsonNode::toString)));
        return new TraceFiles(records);
    }

    private static List<String> differences(Map<String, List<JsonNode>> expected,
            Map<String, List<JsonNode>> actual) {
        List<String> differences = new ArrayList<>();
        expected.keySet().stream().sorted().forEach(key -> {
            if (!actual.containsKey(key)) {
                differences.add("missing record: " + key);
            } else {
                compareRecords(key, expected.get(key), actual.get(key), differences);
            }
        });
        actual.keySet().stream().sorted()
                .filter(key -> !expected.containsKey(key))
                .forEach(key -> differences.add("unexpected record: " + key));
        return differences;
    }

    private static void compareRecords(String key, List<JsonNode> expected, List<JsonNode> actual,
            List<String> differences) {
        if (expected.size() != actual.size()) {
            differences.add("record count changed: " + key + " expected=" + expected.size()
                    + " actual=" + actual.size());
        }
        int count = Math.min(expected.size(), actual.size());
        for (int i = 0; i < count; i++) {
            JsonNode expectedRecord = expected.get(i);
            JsonNode actualRecord = actual.get(i);
            if (expectedRecord.equals(actualRecord)) continue;
            expectedRecord.fieldNames().forEachRemaining(field -> {
                JsonNode left = expectedRecord.get(field);
                JsonNode right = actualRecord.get(field);
                if (!Objects.equals(left, right) && differences.size() < MAX_REPORTED_DIFFERENCES) {
                    differences.add("changed " + key + " field=" + field
                            + " expected=" + printable(left) + " actual=" + printable(right));
                }
            });
            actualRecord.fieldNames().forEachRemaining(field -> {
                if (!expectedRecord.has(field) && differences.size() < MAX_REPORTED_DIFFERENCES) {
                    differences.add("added " + key + " field=" + field
                            + " actual=" + printable(actualRecord.get(field)));
                }
            });
        }
    }

    private static String key(JsonNode record) {
        StringBuilder key = new StringBuilder(record.path("kind").asText());
        appendKeyPart(key, record, "phase");
        appendKeyPart(key, record, "position");
        appendKeyPart(key, record, "layer");
        appendKeyPart(key, record, "stage");
        appendKeyPart(key, record, "name");
        return key.toString();
    }

    private static void appendKeyPart(StringBuilder key, JsonNode record, String field) {
        if (record.has(field)) key.append('|').append(field).append('=').append(record.get(field).asText());
    }

    private static String printable(JsonNode value) {
        return value == null ? "<missing>" : value.toString();
    }

    private static String summary(Path expected, Path actual, int expectedCount, int actualCount,
            List<String> differences) {
        StringBuilder summary = new StringBuilder()
                .append("Dead Reckoning traces differ\n")
                .append("expected=").append(expected).append('\n')
                .append("actual=").append(actual).append('\n')
                .append("records expected=").append(expectedCount)
                .append(" actual=").append(actualCount).append('\n');
        differences.stream().limit(MAX_REPORTED_DIFFERENCES)
                .forEach(difference -> summary.append("- ").append(difference).append('\n'));
        if (differences.size() > MAX_REPORTED_DIFFERENCES) {
            summary.append("- ... ").append(differences.size() - MAX_REPORTED_DIFFERENCES)
                    .append(" more differences\n");
        }
        return summary.toString();
    }

    private record TraceFiles(Map<String, List<JsonNode>> records) {
        private int size() {
            return records.values().stream().mapToInt(List::size).sum();
        }
    }
}
