package io.teknek.deliverance.embedding;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Snapshot;
import com.codahale.metrics.Timer;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2q.AllMiniLmL6V2QuantizedEmbeddingModel;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;
import org.opentest4j.TestAbortedException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/** Test-scope benchmark comparing Deliverance embeddings with LangChain4j's ONNX MiniLM model. */
class OnnxDeliveranceEmbeddingBenchmarkTest {
    private static final int DEFAULT_MAX_FILES = 24;
    private static final int DEFAULT_MAX_CHARS = 4_000;
    // LangChain4j's MiniLM ONNX wrapper effectively behaves like a 128-token encoder; use the same cap by default
    // so the benchmark compares the same input text instead of Deliverance embedding a longer prefix.
    private static final int DEFAULT_MAX_TOKENS = 128;

    @Test
    void benchmarkJavaFilesOnnxVsDeliverance() throws Exception {
        MetricRegistry metrics = new MetricRegistry();
        int maxFiles = Integer.getInteger("deliverance.embedding.benchmark.maxFiles", DEFAULT_MAX_FILES);
        int maxChars = Integer.getInteger("deliverance.embedding.benchmark.maxChars", DEFAULT_MAX_CHARS);
        int maxTokens = Integer.getInteger("deliverance.embedding.benchmark.maxTokens", DEFAULT_MAX_TOKENS);
        Path root = Path.of(System.getProperty("deliverance.embedding.benchmark.root", "."))
                .toAbsolutePath().normalize();

        Timer deliveranceFull = metrics.timer("embedding.deliverance.full");
        Timer onnxFull = metrics.timer("embedding.onnx.full");
        var chars = metrics.histogram("embedding.input.chars");
        var tokens = metrics.histogram("embedding.input.tokens");
        var cosine = metrics.histogram("embedding.output.cosine_vs_onnx.ppm");
        String onnxModel = System.getProperty("deliverance.embedding.benchmark.onnxModel", "fp32");

        List<CorpusEntry> corpus;
        List<float[]> deliveranceVectors = new ArrayList<>();
        String deliveranceProvider;
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel deliverance = buildDeliveranceModel(metrics, allocator, pool)) {
            corpus = corpus(root, maxFiles, maxChars, maxTokens, deliverance);
            assertFalse(corpus.isEmpty(), "benchmark corpus must not be empty under " + root);
            deliveranceProvider = deliverance.primaryTensorOperations().name();
            deliverance.embed(corpus.get(0).text(), PoolingType.AVG);

            for (CorpusEntry entry : corpus) {
                chars.update(entry.text().length());
                tokens.update(entry.tokens());
                try (Timer.Context ignored = deliveranceFull.time()) {
                    deliveranceVectors.add(deliverance.embed(entry.text(), PoolingType.AVG));
                }
            }
        }

        EmbeddingModel onnx = buildOnnxModelOrSkip(onnxModel);
        onnx.embed(TextSegment.from(corpus.get(0).text()));

        double cosineSum = 0.0;
        List<Result> results = new ArrayList<>();
        for (int i = 0; i < corpus.size(); i++) {
            CorpusEntry entry = corpus.get(i);
            float[] deliveranceVector = deliveranceVectors.get(i);
                float[] onnxVector;
                try (Timer.Context ignored = onnxFull.time()) {
                    onnxVector = onnx.embed(TextSegment.from(entry.text())).content().vector();
                }

                assertEquals(384, deliveranceVector.length, "Deliverance embedding dimension");
                assertEquals(384, onnxVector.length, "ONNX embedding dimension");
                float similarity = VectorMathUtils.cosineSimilarity(deliveranceVector, onnxVector);
                cosine.update(Math.round(similarity * 1_000_000.0f));
                cosineSum += similarity;
                results.add(new Result(entry.path(), entry.text().length(), entry.tokens(), similarity));
        }

            printReport(root, onnxModel, deliveranceProvider, maxTokens, corpus,
                deliveranceFull, onnxFull, chars.getSnapshot(), tokens.getSnapshot(), cosine.getSnapshot(),
                cosineSum / corpus.size(), results);
    }

    private static AbstractModel buildDeliveranceModel(MetricRegistry metrics, ArrayQueueTensorAllocator allocator,
            WrappedForkJoinPool pool) {
        AutoModelForEmbeddings.Builder builder = AutoModelForEmbeddings.newBuilder(
                new ModelFetcher("edwardcapriolo", "all-MiniLM-L6-v2-JQ4"));
        builder.withWorkingMemoryType(DType.F32);
        builder.withWorkingQuantType(DType.F32);
        builder.withMetricRegistry(metrics);
        builder.withTensorAllocator(allocator);
        builder.withKvBufferCacheSettings(new KvBufferCacheSettings(true));
        builder.withWrappedForkJoinPool(pool);
        return builder.buildLocalEmbeddingModel();
    }

    private static EmbeddingModel buildOnnxModelOrSkip(String model) {
        try {
            return switch (model) {
                case "quantized", "q" -> new AllMiniLmL6V2QuantizedEmbeddingModel();
                case "fp32", "default" -> new AllMiniLmL6V2EmbeddingModel();
                default -> throw new IllegalArgumentException("Unsupported ONNX benchmark model: " + model
                        + " (expected fp32 or quantized)");
            };
        } catch (UnsatisfiedLinkError | ExceptionInInitializerError e) {
            throw new TestAbortedException("ONNX Runtime native library is not available on this machine", e);
        }
    }

    private static void warmup(AbstractModel deliverance, EmbeddingModel onnx, String text) {
        deliverance.embed(text, PoolingType.AVG);
        onnx.embed(TextSegment.from(text));
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

    private static void printReport(Path root, String onnxModel, String deliveranceProvider, int maxTokens,
            List<CorpusEntry> corpus, Timer deliveranceFull, Timer onnxFull, Snapshot chars, Snapshot tokens,
            Snapshot cosine, double meanCosine, List<Result> results) {
        System.out.printf("%nEmbedding benchmark corpus root=%s files=%d onnxModel=%s deliveranceProvider=%s maxTokens=%d residency=sequential%n",
                root, corpus.size(), onnxModel, deliveranceProvider, maxTokens);
        printTimer("embedding.deliverance.full", deliveranceFull);
        printTimer("embedding.onnx.full", onnxFull);
        System.out.printf("embedding.onnx_vs_deliverance.speedup=%.3fx%n", throughput(onnxFull) / throughput(deliveranceFull));
        printHistogram("embedding.input.chars", chars, 1.0);
        printHistogram("embedding.input.tokens", tokens, 1.0);
        printHistogram("embedding.output.cosine_vs_onnx", cosine, 1_000_000.0);
        System.out.printf("embedding.output.cosine_vs_onnx.mean=%.6f%n", meanCosine);
        printCosineBuckets(results);
        printWorstCosines(results);
    }

    private static void printTimer(String name, Timer timer) {
        Snapshot snapshot = timer.getSnapshot();
        double totalMillis = nanosToMillis(snapshot.getMean() * timer.getCount());
        System.out.printf("%s count=%d total_ms=%.3f mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f throughput_s=%.3f%n",
                name,
                timer.getCount(),
                totalMillis,
                nanosToMillis(snapshot.getMean()),
                nanosToMillis(snapshot.getMedian()),
                nanosToMillis(snapshot.get95thPercentile()),
                nanosToMillis(snapshot.get99thPercentile()),
                throughput(timer));
    }

    private static double throughput(Timer timer) {
        double totalMillis = nanosToMillis(timer.getSnapshot().getMean() * timer.getCount());
        return totalMillis == 0.0 ? Double.NaN : timer.getCount() / (totalMillis / 1000.0);
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

    private static void printWorstCosines(List<Result> results) {
        System.out.println("embedding.output.cosine_vs_onnx.worst:");
        results.stream()
                .sorted(Comparator.comparingDouble(Result::cosine))
                .limit(Math.min(8, results.size()))
                .forEach(result -> System.out.printf("  cosine=%.6f tokens=%d chars=%d file=%s%n",
                        result.cosine(), result.tokens(), result.chars(), result.path()));
    }

    private static void printCosineBuckets(List<Result> results) {
        System.out.println("embedding.output.cosine_vs_onnx.by_tokens:");
        printBucket(results, 0, 64);
        printBucket(results, 65, 128);
        printBucket(results, 129, 256);
        printBucket(results, 257, Integer.MAX_VALUE);
    }

    private static void printBucket(List<Result> results, int minInclusive, int maxInclusive) {
        List<Result> bucket = results.stream()
                .filter(result -> result.tokens() >= minInclusive && result.tokens() <= maxInclusive)
                .toList();
        if (bucket.isEmpty()) {
            return;
        }
        double mean = bucket.stream().mapToDouble(Result::cosine).average().orElse(Double.NaN);
        double min = bucket.stream().mapToDouble(Result::cosine).min().orElse(Double.NaN);
        double max = bucket.stream().mapToDouble(Result::cosine).max().orElse(Double.NaN);
        String label = maxInclusive == Integer.MAX_VALUE ? minInclusive + "+" : minInclusive + "-" + maxInclusive;
        System.out.printf("  tokens=%s count=%d mean=%.6f min=%.6f max=%.6f%n",
                label, bucket.size(), mean, min, max);
    }

    private static double nanosToMillis(double nanos) {
        return nanos / TimeUnit.MILLISECONDS.toNanos(1);
    }

    private record CorpusEntry(Path path, String text, int tokens) {
    }

    private record Result(Path path, int chars, int tokens, float cosine) {
    }
}
