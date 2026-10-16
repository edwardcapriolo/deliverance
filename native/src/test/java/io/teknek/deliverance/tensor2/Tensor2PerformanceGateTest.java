package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.Test;
import org.junit.platform.engine.TestDescriptor;
import org.junit.platform.launcher.Launcher;
import org.junit.platform.launcher.LauncherDiscoveryRequest;
import org.junit.platform.launcher.TestIdentifier;
import org.junit.platform.launcher.TestPlan;
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder;
import org.junit.platform.launcher.core.LauncherFactory;
import org.junit.platform.launcher.listeners.SummaryGeneratingListener;
import org.junit.platform.launcher.listeners.TestExecutionSummary;
import org.junit.platform.engine.discovery.DiscoverySelectors;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.fail;

/** Persistent, hostname-local performance gate for the tensor2 fuzz invocations. */
class Tensor2PerformanceGateTest {
    private static final String TIMING_ENABLED = "deliverance.tensor2.timing.csv";
    private static final String TIMING_PATH = "deliverance.tensor2.timing.csv.path";
    private static final String ROOT_PROPERTY = "deliverance.tensor2.performance.gate.directory";
    private static final int HISTORY_SIZE = 5;
    private static final int WARMUP_PERCENT = 5;
    private static final int RETRY_COUNT = 10;
    private static final long RETRY_DELAY_MILLIS = 1;
    private static final String FUZZ_PACKAGE = "io.teknek.deliverance.tensor2";

    @Test
    void tensorFuzzPerformanceGate() throws Exception {
        Path hostRoot = hostRoot();
        Files.createDirectories(hostRoot);
        try (FileChannel lockChannel = FileChannel.open(hostRoot.resolve("gate.lock"),
                StandardOpenOption.CREATE, StandardOpenOption.WRITE);
             FileLock ignored = lockChannel.lock()) {
            Files.createDirectories(historyDirectory(hostRoot));
            Files.createDirectories(failureDirectory(hostRoot));

            List<TimingRun> history = loadHistory(hostRoot);
            List<String> invocations = invocations();
            if (history.isEmpty()) {
                history = bootstrap(hostRoot, invocations);
            }

            Path current = hostRoot.resolve("current-" + timestamp() + ".csv");
            List<Failure> failures = new ArrayList<>();
            try {
                runWarmup();
                for (String uniqueId : invocations) {
                    failures.addAll(runInvocation(hostRoot, current, uniqueId, history));
                }
            } catch (Exception | AssertionError failure) {
                retainFailure(hostRoot, current);
                throw failure;
            }
            if (!failures.isEmpty()) {
                Path retained = retainFailure(hostRoot, current);
                fail(formatFailures(retained, failures));
            }
            rotatePassingCurrent(hostRoot, current);
        }
    }

    private static List<TimingRun> bootstrap(Path hostRoot, List<String> invocations) throws Exception {
        List<TimingRun> baselines = new ArrayList<>();
        for (int run = 0; run < HISTORY_SIZE; run++) {
            Path output = hostRoot.resolve("bootstrap-" + timestamp() + "-" + run + ".csv");
            TimingRun baseline = new TimingRun();
            runWarmup();
            for (String uniqueId : invocations) {
                Invocation invocation = executeInvocation(uniqueId);
                assertNestedRunSucceeded("bootstrap", invocation.summary());
                baseline.addAll(invocation.run());
            }
            baseline.write(output);
            Files.move(output, historyDirectory(hostRoot).resolve(output.getFileName()),
                    StandardCopyOption.ATOMIC_MOVE);
            baselines.add(baseline);
        }
        return baselines;
    }

    private static List<Failure> runInvocation(Path hostRoot, Path current, String uniqueId,
            List<TimingRun> history) throws Exception {
        Invocation first = executeInvocation(uniqueId);
        assertNestedRunSucceeded("measured", first.summary());
        List<Failure> suspects = compare(first.run(), history);
        TimingRun finalRun = first.run();
        Map<String, List<Long>> retryValues = new HashMap<>();
        for (int attempt = 0; attempt < RETRY_COUNT && !suspects.isEmpty(); attempt++) {
            Thread.sleep(RETRY_DELAY_MILLIS);
            Invocation retry = executeInvocation(uniqueId);
            assertNestedRunSucceeded("retry", retry.summary());
            TimingRun retryRun = retry.run();
            finalRun = retryRun;
            for (Failure suspect : suspects) {
                Measurement measurement = retryRun.measurements().get(suspect.key());
                if (measurement != null) {
                    retryValues.computeIfAbsent(suspect.key(), ignored -> new ArrayList<>())
                            .add(measurement.elapsedNanos());
                }
            }
            suspects = suspects.stream().filter(suspect -> !recovered(retryRun, suspect)).toList();
        }
        // Rows are committed only after all retries for this invocation have finished.
        finalRun.appendTo(current);
        return suspects.stream().map(suspect -> suspect.withRetries(retryValues.getOrDefault(suspect.key(), List.of())))
                .toList();
    }

    private static boolean recovered(TimingRun run, Failure suspect) {
        Measurement measurement = run.measurements().get(suspect.key());
        return measurement != null && measurement.elapsedNanos() <= suspect.maximum();
    }

    private static List<Failure> compare(TimingRun current, List<TimingRun> history) {
        List<Failure> failures = new ArrayList<>();
        for (Map.Entry<String, Measurement> entry : current.measurements().entrySet()) {
            List<Long> oldValues = new ArrayList<>();
            for (TimingRun run : history) {
                Measurement measurement = run.measurements().get(entry.getKey());
                if (measurement != null) {
                    oldValues.add(measurement.elapsedNanos());
                }
            }
            if (oldValues.size() < 2) {
                continue;
            }
            long maximum = oldValues.stream().mapToLong(Long::longValue).max().orElseThrow();
            if (entry.getValue().elapsedNanos() > maximum * 1.10) {
                long minimum = oldValues.stream().mapToLong(Long::longValue).min().orElseThrow();
                failures.add(new Failure(entry.getKey(), oldValues, entry.getValue().elapsedNanos(), minimum, maximum,
                        List.of()));
            }
        }
        return failures;
    }

    private static Invocation executeInvocation(String uniqueId) {
        TimingRun run = new TimingRun();
        String oldEnabled = System.getProperty(TIMING_ENABLED);
        String oldPath = System.getProperty(TIMING_PATH);
        try {
            System.setProperty(TIMING_ENABLED, "true");
            System.setProperty(TIMING_PATH, "target/tensor2-performance-gate-unused.csv");
            Tensor2TimingCsv.installInMemorySink(run::accept);
            SummaryGeneratingListener summary = new SummaryGeneratingListener();
            Launcher launcher = LauncherFactory.create();
            launcher.registerTestExecutionListeners(summary);
            launcher.execute(request(uniqueId));
            return new Invocation(run, summary.getSummary());
        } finally {
            Tensor2TimingCsv.clearInMemorySink();
            restoreProperty(TIMING_ENABLED, oldEnabled);
            restoreProperty(TIMING_PATH, oldPath);
        }
    }

    private static void runWarmup() {
        String oldEnabled = System.getProperty(TIMING_ENABLED);
        try {
            System.setProperty(TIMING_ENABLED, "false");
            Launcher launcher = LauncherFactory.create();
            launcher.execute(warmupRequest());
        } finally {
            restoreProperty(TIMING_ENABLED, oldEnabled);
        }
    }

    private static List<String> invocations() {
        Launcher launcher = LauncherFactory.create();
        TestPlan plan = launcher.discover(baseRequest());
        List<String> result = new ArrayList<>();
        collectInvocations(plan, plan.getRoots(), result);
        return result;
    }

    private static void collectInvocations(TestPlan plan, java.util.Set<TestIdentifier> identifiers,
            List<String> result) {
        for (TestIdentifier identifier : identifiers) {
            if (identifier.isTest()) {
                result.add(identifier.getUniqueId());
            }
            collectInvocations(plan, plan.getChildren(identifier), result);
        }
    }

    private static LauncherDiscoveryRequest baseRequest() {
        return LauncherDiscoveryRequestBuilder.request()
                .selectors(DiscoverySelectors.selectPackage(FUZZ_PACKAGE))
                .filters(org.junit.platform.launcher.EngineFilter.includeEngines("junit-jupiter"))
                .filters(org.junit.platform.engine.discovery.ClassNameFilter.includeClassNamePatterns(".*FuzzTest"))
                .build();
    }

    private static LauncherDiscoveryRequest warmupRequest() {
        return LauncherDiscoveryRequestBuilder.request()
                .selectors(DiscoverySelectors.selectPackage(FUZZ_PACKAGE))
                .filters(org.junit.platform.launcher.EngineFilter.includeEngines("junit-jupiter"))
                .filters(org.junit.platform.engine.discovery.ClassNameFilter.includeClassNamePatterns(".*FuzzTest"))
                .filters(new FivePercentFilter()).build();
    }

    private static LauncherDiscoveryRequest request(String uniqueId) {
        return LauncherDiscoveryRequestBuilder.request().selectors(DiscoverySelectors.selectUniqueId(uniqueId))
                .filters(org.junit.platform.launcher.EngineFilter.includeEngines("junit-jupiter")).build();
    }

    private static void assertNestedRunSucceeded(String phase, TestExecutionSummary summary) {
        if (summary.getFailures().isEmpty()) {
            return;
        }
        fail("Nested fuzz " + phase + " failed:\n" + summary.getFailures().stream()
                .map(failure -> phase + " failure: " + failure.getTestIdentifier().getDisplayName()
                        + " -> " + failure.getException()).collect(Collectors.joining("\n")));
    }

    private static List<TimingRun> loadHistory(Path hostRoot) {
        return historyFiles(hostRoot).stream().map(Tensor2PerformanceGateTest::readCsv).toList();
    }

    private static TimingRun readCsv(Path path) {
        try {
            List<String> rows = Files.readAllLines(path);
            TimingRun run = new TimingRun();
            for (int i = 1; i < rows.size(); i++) {
                run.accept(rows.get(i) + "\n");
            }
            return run;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read timing CSV: " + path, e);
        }
    }

    private static void rotatePassingCurrent(Path hostRoot, Path current) throws IOException {
        Files.move(current, historyDirectory(hostRoot).resolve("run-" + timestamp() + ".csv"),
                StandardCopyOption.ATOMIC_MOVE);
        List<Path> all = historyFiles(hostRoot);
        for (int i = HISTORY_SIZE; i < all.size(); i++) {
            Files.deleteIfExists(all.get(i));
        }
    }

    private static Path retainFailure(Path hostRoot, Path current) throws IOException {
        if (!Files.isRegularFile(current)) {
            return current;
        }
        Path destination = failureDirectory(hostRoot).resolve(current.getFileName());
        Files.move(current, destination, StandardCopyOption.ATOMIC_MOVE);
        return destination;
    }

    private static List<Path> historyFiles(Path hostRoot) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(historyDirectory(hostRoot), "*.csv")) {
            List<Path> result = new ArrayList<>();
            stream.forEach(result::add);
            result.sort(Comparator.comparing(Path::getFileName).reversed());
            return result;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to list performance history", e);
        }
    }

    private static Path hostRoot() {
        try {
            return Path.of(System.getProperty(ROOT_PROPERTY, "target/tensor2-performance-gate"),
                    java.net.InetAddress.getLocalHost().getHostName());
        } catch (IOException e) {
            throw new IllegalStateException("Unable to determine hostname", e);
        }
    }

    private static Path historyDirectory(Path root) { return root.resolve("history"); }
    private static Path failureDirectory(Path root) { return root.resolve("failures"); }
    private static String timestamp() { return Long.toString(Instant.now().toEpochMilli()); }

    private static void restoreProperty(String name, String value) {
        if (value == null) System.clearProperty(name); else System.setProperty(name, value);
    }

    private record Invocation(TimingRun run, TestExecutionSummary summary) { }
    private record Measurement(long elapsedNanos) { }
    private record Failure(String key, List<Long> oldValues, long newValue, long minimum, long maximum,
            List<Long> retries) {
        Failure withRetries(List<Long> values) { return new Failure(key, oldValues, newValue, minimum, maximum, values); }
    }

    private static final class TimingRun {
        private final List<String> rows = new ArrayList<>();
        private final Map<String, Measurement> measurements = new LinkedHashMap<>();
        private final Map<String, Integer> occurrences = new HashMap<>();
        private final Map<Integer, String> columns = columns(Tensor2TimingCsv.header());

        void accept(String row) {
            rows.add(row);
            String[] values = row.stripTrailing().split(",", -1);
            int elapsed = indexOf(Tensor2TimingCsv.header().split(",", -1), "elapsed_ns");
            String base = columns.entrySet().stream().map(e -> e.getValue() + "=" + values[e.getKey()])
                    .collect(Collectors.joining("|"));
            String key = base + "|occurrence=" + occurrences.merge(base, 1, Integer::sum);
            measurements.put(key, new Measurement(Long.parseLong(values[elapsed])));
        }

        void addAll(TimingRun other) { other.rows.forEach(this::accept); }
        Map<String, Measurement> measurements() { return measurements; }

        void appendTo(Path output) throws IOException {
            Files.createDirectories(output.toAbsolutePath().getParent());
            if (Files.notExists(output)) Files.writeString(output, Tensor2TimingCsv.header());
            Files.writeString(output, String.join("", rows), StandardOpenOption.APPEND);
        }

        void write(Path output) throws IOException {
            Files.createDirectories(output.toAbsolutePath().getParent());
            Files.writeString(output, Tensor2TimingCsv.header() + String.join("", rows),
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        }

        private static Map<Integer, String> columns(String headerLine) {
            String[] header = headerLine.split(",", -1);
            int elapsed = indexOf(header, "elapsed_ns");
            Map<Integer, String> result = new LinkedHashMap<>();
            for (int i = 0; i < header.length; i++) {
                if (i != elapsed && !header[i].equals("mean_ms") && !header[i].equals("gflops")) result.put(i, header[i]);
            }
            return result;
        }
    }

    private static int indexOf(String[] values, String wanted) {
        for (int i = 0; i < values.length; i++) if (values[i].equals(wanted)) return i;
        throw new IllegalStateException("Timing CSV has no " + wanted + " column");
    }

    private static String formatFailures(Path current, List<Failure> failures) {
        StringBuilder message = new StringBuilder("Tensor2 performance gate failed for ")
                .append(failures.size()).append(" case(s).\nCandidate retained at: ").append(current).append('\n');
        for (Failure failure : failures) {
            message.append("case=").append(failure.key()).append(" old_elapsed_ns=").append(failure.oldValues())
                    .append(" new_elapsed_ns=").append(failure.newValue()).append(" known_range_ns=")
                    .append(failure.minimum()).append("..").append(failure.maximum())
                    .append(" retry_elapsed_ns=").append(failure.retries()).append('\n');
        }
        return message.toString();
    }

    private static final class FivePercentFilter implements org.junit.platform.launcher.PostDiscoveryFilter {
        @Override public org.junit.platform.engine.FilterResult apply(TestDescriptor descriptor) {
            if (descriptor.getType() != TestDescriptor.Type.TEST) return org.junit.platform.engine.FilterResult.included("container");
            return Math.floorMod(descriptor.getDisplayName().hashCode(), 100) < WARMUP_PERCENT
                    ? org.junit.platform.engine.FilterResult.included("warm-up sample")
                    : org.junit.platform.engine.FilterResult.excluded("outside warm-up sample");
        }
    }
}
