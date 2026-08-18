package io.teknek.deliverance.embedding;

import com.codahale.metrics.Counter;
import com.codahale.metrics.Histogram;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Snapshot;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/** Deliverance-only embedding benchmark that prints InferenceProfiler and Dropwizard metric summaries. */
class EmbeddingInferenceProfilerBenchmarkTest {
    private static final int DEFAULT_MAX_FILES = 24;
    private static final int DEFAULT_MAX_CHARS = 4_000;
    private static final int DEFAULT_MAX_TOKENS = 128;
    private static final int DEFAULT_MAX_METRIC_ROWS = 40;

    @Test
    void profileJavaFileEmbeddingsWithInferenceProfiler() throws Exception {
        MetricRegistry metrics = new MetricRegistry();
        int maxFiles = Integer.getInteger("deliverance.embedding.profile.maxFiles", DEFAULT_MAX_FILES);
        int maxChars = Integer.getInteger("deliverance.embedding.profile.maxChars", DEFAULT_MAX_CHARS);
        int maxTokens = Integer.getInteger("deliverance.embedding.profile.maxTokens", DEFAULT_MAX_TOKENS);
        int maxMetricRows = Integer.getInteger("deliverance.embedding.profile.maxMetricRows", DEFAULT_MAX_METRIC_ROWS);
        Path root = Path.of(System.getProperty("deliverance.embedding.profile.root", "."))
                .toAbsolutePath().normalize();

        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        boolean previousProfilerState = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(64));
             AbstractModel model = buildDeliveranceModel(metrics, allocator, pool)) {
            List<CorpusEntry> corpus = corpus(root, maxFiles, maxChars, maxTokens, model);
            assertFalse(corpus.isEmpty(), "profile corpus must not be empty under " + root);

            model.embed(corpus.get(0).text(), PoolingType.AVG);
            Timer full = metrics.timer("embedding.profile.deliverance.full");
            var chars = metrics.histogram("embedding.profile.input.chars");
            var tokens = metrics.histogram("embedding.profile.input.tokens");

            for (CorpusEntry entry : corpus) {
                chars.update(entry.text().length());
                tokens.update(entry.tokens());
                try (Timer.Context ignored = full.time()) {
                    float[] embedding = model.embed(entry.text(), PoolingType.AVG);
                    assertEquals(384, embedding.length, "Deliverance embedding dimension");
                }
            }

            System.out.printf("%nEmbedding profiler corpus root=%s files=%d provider=%s maxTokens=%d quantizeOnDemand=Q4%n",
                    root, corpus.size(), model.primaryTensorOperations().name(), maxTokens);
            printTimer("embedding.profile.deliverance.full", full);
            printHistogram("embedding.profile.input.chars", chars.getSnapshot(), 1.0);
            printHistogram("embedding.profile.input.tokens", tokens.getSnapshot(), 1.0);
            InferenceProfiler.printSummary("embedding.profile.inference_profiler", maxMetricRows);
            printMetricRegistrySummary(metrics, maxMetricRows);
        } finally {
            InferenceProfiler.setEnabled(previousProfilerState);
        }
    }

    private static AbstractModel buildDeliveranceModel(MetricRegistry metrics, ArrayQueueTensorAllocator allocator,
            WrappedForkJoinPool pool) {
        AutoModelForEmbeddings.Builder builder = AutoModelForEmbeddings.newBuilder(
                new ModelFetcher("sentence-transformers", "all-MiniLM-L6-v2"));
        builder.quantizeOnDemand(DType.Q4, "sentence-transformers", "all-MiniLM-L6-v2-JQ4");
        builder.withWorkingMemoryType(DType.F32);
        builder.withWorkingQuantType(DType.F32);
        builder.withMetricRegistry(metrics);
        builder.withTensorAllocator(allocator);
        builder.withKvBufferCacheSettings(new KvBufferCacheSettings(true));
        builder.withWrappedForkJoinPool(pool);
        return builder.buildLocalEmbeddingModel();
    }

    private static List<CorpusEntry> corpus(Path root, int maxFiles, int maxChars, int maxTokens, AbstractModel model)
            throws IOException {
        List<Path> files;
        try (var stream = Files.walk(root)) {
            files = stream
                    .filter(Files::isRegularFile)
                    .filter(path -> path.toString().endsWith(".java"))
                    .filter(path -> !path.toString().contains("/target/"))
                    .sorted(Comparator.comparing(Path::toString))
                    .limit(maxFiles * 4L)
                    .toList();
        }
        List<CorpusEntry> corpus = new ArrayList<>();
        for (Path file : files) {
            String text = Files.readString(file);
            text = trimToTokenLimit(text, maxChars, maxTokens, model);
            if (text.isBlank()) {
                continue;
            }
            int tokens = model.getTokenizer().encode(text).length();
            if (tokens <= maxTokens) {
                corpus.add(new CorpusEntry(file, text, tokens));
            }
            if (corpus.size() >= maxFiles) {
                break;
            }
        }
        return corpus;
    }

    private static String trimToTokenLimit(String text, int maxChars, int maxTokens, AbstractModel model) {
        String trimmed = text.length() <= maxChars ? text : text.substring(0, maxChars);
        while (trimmed.length() > 256) {
            Encoding encoding = model.getTokenizer().encode(trimmed);
            if (encoding.length() <= maxTokens) {
                return trimmed;
            }
            trimmed = trimmed.substring(0, Math.max(256, (int) (trimmed.length() * 0.8)));
        }
        return trimmed;
    }

    private static void printMetricRegistrySummary(MetricRegistry metrics, int maxRows) {
        System.out.println("[metrics] timers");
        metrics.getTimers().entrySet().stream()
                .sorted(Comparator.comparingDouble(entry -> -estimatedTotalMillis(entry.getValue())))
                .limit(maxRows)
                .forEach(entry -> printTimer("[metrics] " + entry.getKey(), entry.getValue()));

        System.out.println("[metrics] histograms");
        metrics.getHistograms().entrySet().stream()
                .filter(entry -> entry.getValue().getCount() > 0)
                .sorted(Comparator.comparingLong((Map.Entry<String, Histogram> entry) -> entry.getValue().getCount())
                        .reversed())
                .limit(maxRows)
                .forEach(entry -> printHistogram("[metrics] " + entry.getKey(), entry.getValue().getSnapshot(), 1.0));

        System.out.println("[metrics] counters");
        metrics.getCounters().entrySet().stream()
                .filter(entry -> entry.getValue().getCount() != 0 || InferenceProfiler.shouldPrintCounter(entry.getKey()))
                .sorted(Comparator.comparingLong((Map.Entry<String, Counter> entry) -> Math.abs(entry.getValue().getCount()))
                        .reversed())
                .limit(maxRows)
                .forEach(entry -> System.out.printf("[metrics] %-55s count=%d%n",
                        entry.getKey(), entry.getValue().getCount()));
    }

    private static double estimatedTotalMillis(Timer timer) {
        return nanosToMillis(timer.getSnapshot().getMean() * timer.getCount());
    }

    private static void printTimer(String name, Timer timer) {
        Snapshot snapshot = timer.getSnapshot();
        double totalMillis = estimatedTotalMillis(timer);
        double throughput = totalMillis == 0.0 ? Double.NaN : timer.getCount() / (totalMillis / 1000.0);
        System.out.printf("%s count=%d total_ms=%.3f mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f throughput_s=%.3f%n",
                name,
                timer.getCount(),
                totalMillis,
                nanosToMillis(snapshot.getMean()),
                nanosToMillis(snapshot.getMedian()),
                nanosToMillis(snapshot.get95thPercentile()),
                nanosToMillis(snapshot.get99thPercentile()),
                throughput);
    }

    private static void printHistogram(String name, Snapshot snapshot, double scale) {
        System.out.printf("%s min=%.3f p05=%.3f mean=%.3f p50=%.3f p95=%.3f max=%.3f%n",
                name,
                snapshot.getMin() / scale,
                snapshot.getValue(0.05) / scale,
                snapshot.getMean() / scale,
                snapshot.getMedian() / scale,
                snapshot.get95thPercentile() / scale,
                snapshot.getMax() / scale);
    }

    private static double nanosToMillis(double nanos) {
        return nanos / TimeUnit.MILLISECONDS.toNanos(1);
    }

    private record CorpusEntry(Path path, String text, int tokens) {
    }
}
