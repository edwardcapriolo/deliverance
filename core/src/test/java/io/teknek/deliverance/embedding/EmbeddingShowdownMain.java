package io.teknek.deliverance.embedding;

import com.fasterxml.jackson.databind.JsonNode;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForEmbeddings;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;

/** Runs the Java side of the reproducible EmbeddingGemma 2 parity benchmark. */
public final class EmbeddingShowdownMain {
    private EmbeddingShowdownMain() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        Files.createDirectories(options.output.toAbsolutePath().getParent());
        List<Case> cases = readCases(options.corpus);
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        List<Result> results;
        long[] samples = new long[cases.size() * options.iterations];
        int sampleIndex = 0;
        long tokenCount = 0;
        System.out.printf("java phase=model_load cases=%d warmup=%d iterations=%d pool_size=%d%n",
                cases.size(), options.warmup, options.iterations, options.poolSize);
        long modelLoadStart = System.nanoTime();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(options.poolSize));
             AbstractModel model = AutoModelForEmbeddings.newBuilder(options.modelDirectory)
                     .buildLocalEmbeddingModel()) {
            System.out.printf(Locale.ROOT, "java phase=model_loaded elapsed_s=%.3f%n",
                    elapsedSeconds(modelLoadStart));
            results = new ArrayList<>(cases.size());
            System.out.println("java phase=reference_vectors");
            long referenceStart = System.nanoTime();
            for (Case value : cases) {
                int[] tokenIds = model.getTokenizer().encode(value.text()).inputIds();
                float[] embedding = model.embed(value.text(), PoolingType.AVG);
                results.add(new Result(value.id(), tokenIds, embedding, sha256Floats(embedding)));
            }
            System.out.printf(Locale.ROOT, "java phase=reference_vectors_complete elapsed_s=%.3f%n",
                    elapsedSeconds(referenceStart));
            for (int warmup = 0; warmup < options.warmup; warmup++) {
                System.out.printf("java phase=warmup iteration=%d/%d%n", warmup + 1, options.warmup);
                long warmupStart = System.nanoTime();
                for (Case value : cases) {
                    model.embed(value.text(), PoolingType.AVG);
                }
                System.out.printf(Locale.ROOT, "java phase=warmup_complete iteration=%d/%d elapsed_s=%.3f%n",
                        warmup + 1, options.warmup, elapsedSeconds(warmupStart));
            }
            System.out.println("java phase=timed_iterations");
            for (int iteration = 0; iteration < options.iterations; iteration++) {
                System.out.printf("java phase=timed_iteration iteration=%d/%d%n", iteration + 1, options.iterations);
                long iterationStart = System.nanoTime();
                for (Case value : cases) {
                    long start = System.nanoTime();
                    model.embed(value.text(), PoolingType.AVG);
                    samples[sampleIndex++] = System.nanoTime() - start;
                    tokenCount += model.getTokenizer().encode(value.text()).length();
                }
                System.out.printf(Locale.ROOT, "java phase=timed_iteration_complete iteration=%d/%d elapsed_s=%.3f%n",
                        iteration + 1, options.iterations, elapsedSeconds(iterationStart));
            }
            System.out.println("java phase=timed_iterations_complete");
        }

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("engine", "deliverance-java");
        report.put("model", options.modelDirectory.toString());
        report.put("corpus", options.corpus.toString());
        report.put("cases", results);
        report.put("iterations", options.iterations);
        report.put("warmup", options.warmup);
        report.put("stats", stats(samples, tokenCount));
        JsonUtils.om.writeValue(options.output.toFile(), report);
        printSummary("java", options, samples, tokenCount);
    }

    private static double elapsedSeconds(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000_000.0;
    }

    private static List<Case> readCases(Path corpus) throws Exception {
        List<Case> result = new ArrayList<>();
        for (String line : Files.readAllLines(corpus)) {
            if (!line.isBlank()) {
                JsonNode node = JsonUtils.om.readTree(line);
                result.add(new Case(node.get("id").asText(), node.get("text").asText()));
            }
        }
        if (result.isEmpty()) throw new IllegalArgumentException("Corpus is empty: " + corpus);
        return result;
    }

    private static Map<String, Object> stats(long[] samples, long tokenCount) {
        long[] sorted = samples.clone();
        Arrays.sort(sorted);
        double totalSeconds = Arrays.stream(samples).sum() / 1_000_000_000.0;
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("count", samples.length);
        result.put("mean_ms", totalSeconds * 1000.0 / samples.length);
        result.put("p50_ms", nanos(sorted, 0.50));
        result.put("p95_ms", nanos(sorted, 0.95));
        result.put("p99_ms", nanos(sorted, 0.99));
        result.put("embeddings_s", samples.length / totalSeconds);
        result.put("tokens_s", tokenCount / totalSeconds);
        return result;
    }

    private static double nanos(long[] sorted, double percentile) {
        return sorted[Math.min(sorted.length - 1, (int) Math.ceil(percentile * sorted.length) - 1)] / 1_000_000.0;
    }

    private static void printSummary(String engine, Options options, long[] samples, long tokenCount) {
        Map<String, Object> stats = stats(samples, tokenCount);
        System.out.printf(Locale.ROOT,
                "%s cases=%d iterations=%d mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f embeddings_s=%.1f tokens_s=%.1f output=%s%n",
                engine, samples.length / options.iterations, options.iterations,
                stats.get("mean_ms"), stats.get("p50_ms"), stats.get("p95_ms"), stats.get("p99_ms"),
                stats.get("embeddings_s"), stats.get("tokens_s"), options.output);
    }

    private static String sha256Floats(float[] values) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        for (float value : values) {
            int bits = Float.floatToIntBits(value);
            digest.update((byte) (bits >>> 24));
            digest.update((byte) (bits >>> 16));
            digest.update((byte) (bits >>> 8));
            digest.update((byte) bits);
        }
        StringBuilder result = new StringBuilder();
        for (byte value : digest.digest()) result.append(String.format(Locale.ROOT, "%02x", value));
        return result.toString();
    }

    private record Case(String id, String text) {
    }

    private record Result(String id, int[] tokenIds, float[] embedding, String embeddingBitsSha256) {
    }

    private record Options(Path modelDirectory, Path corpus, Path output, int warmup, int iterations, int poolSize) {
        static Options parse(String[] args) {
            Path model = null;
            Path corpus = Path.of("core/src/test/resources/embedding-showdown.jsonl");
            Path output = Path.of("core/target/embedding-showdown-java.json");
            int warmup = 3;
            int iterations = 10;
            int poolSize = Math.max(1, Runtime.getRuntime().availableProcessors());
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--model-dir" -> model = Path.of(args[++i]);
                    case "--corpus" -> corpus = Path.of(args[++i]);
                    case "--output" -> output = Path.of(args[++i]);
                    case "--warmup" -> warmup = Integer.parseInt(args[++i]);
                    case "--iterations" -> iterations = Integer.parseInt(args[++i]);
                    case "--pool-size" -> poolSize = Integer.parseInt(args[++i]);
                    default -> throw new IllegalArgumentException("Unknown argument " + args[i]);
                }
            }
            if (model == null) throw new IllegalArgumentException("--model-dir is required");
            return new Options(model, corpus, output, warmup, iterations, poolSize);
        }
    }
}
