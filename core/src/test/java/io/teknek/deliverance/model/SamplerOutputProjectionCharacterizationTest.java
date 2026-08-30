package io.teknek.deliverance.model;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static io.teknek.deliverance.tensor.TensorTestSupport.deterministicTensor;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Characterizes the sampler LM-head projection without loading a model checkpoint. */
class SamplerOutputProjectionCharacterizationTest {
    private static final int QWEN_VOCABULARY_SIZE = 151_936;
    private static final int QWEN_06_EMBEDDING_LENGTH = 1024;
    private static final boolean RUN_QWEN_SIZED_CHARACTERIZATION = true;
    private static final int QWEN_SIZED_REPETITIONS = 2;
    private static final List<Integer> QWEN_SIZED_SPLITS = List.of(1, 2, 4, 8, 16, 32, 64);
    private static final int TIMING_WARMUPS = 1;
    private static final int INITIAL_WARMUPS = 2;

    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @Test
    void smallOutputProjectionFormulationsProduceComparableLogits() {
        assumeTrue(MachineSpec.VECTOR_TYPE != MachineSpec.Type.NONE, "Panama vector provider is unavailable");

        int workers = Math.min(4, Math.max(1, Runtime.getRuntime().availableProcessors()));
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(workers))) {
            List<ProjectionResult> results = characterizeScenario("small", 4096, 256, 2,
                    List.of(panama(pool)), List.of(1, 2, 4), pool);

            for (ProjectionResult result : results) {
                System.out.printf("[sampler-output-characterization] %s%n", result);
                assertTrue(result.maxAbsError() < 0.25f,
                        () -> "unexpected output projection error for " + result.name());
            }
        }
    }

    @Test
    void qwenSizedOutputProjectionFormulationCharacterization() {
        if (!RUN_QWEN_SIZED_CHARACTERIZATION) {
            return;
        }
        assumeTrue(MachineSpec.VECTOR_TYPE != MachineSpec.Type.NONE, "Panama vector provider is unavailable");

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            List<OpsCase> providers = new ArrayList<>();
            providers.add(panama(pool));
            try {
                TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
                providers.add(new OpsCase("native_simd", new NativeSimdTensorOperations(panama, 64)));
            } catch (Throwable t) {
                System.out.printf("[sampler-output-characterization] native_simd unavailable: %s%n", t.toString());
            }

            List<ProjectionResult> results = characterizeScenario("qwen", QWEN_VOCABULARY_SIZE,
                    QWEN_06_EMBEDDING_LENGTH, QWEN_SIZED_REPETITIONS, providers, QWEN_SIZED_SPLITS, pool);
            for (ProjectionResult result : results) {
                System.out.printf("[sampler-output-characterization] %s%n", result);
            }
        }
    }

    private List<ProjectionResult> characterizeScenario(String scenarioName, int vocabularySize, int embeddingLength,
            int repetitions, List<OpsCase> providers, List<Integer> splitCounts, WrappedForkJoinPool pool) {
        try (AbstractTensor embedding = deterministicTensor(1, embeddingLength, 409);
             AbstractTensor denseWeights = deterministicTensor(vocabularySize, embeddingLength, 997);
             AbstractTensor q4Weights = AbstractTensorUtils.quantize(denseWeights, DType.Q4, true)) {
            List<ProjectionResult> results = new ArrayList<>();
            ProjectionResult baseline = null;

            for (OpsCase provider : providers) {
                try {
                    List<ProjectionResult> providerResults = characterizeProvider(scenarioName, provider,
                            embedding, q4Weights, vocabularySize, embeddingLength, repetitions, splitCounts, pool,
                            baseline == null ? null : baseline.logits());
                    if (baseline == null) {
                        baseline = providerResults.get(0);
                    }
                    results.addAll(providerResults);
                } catch (Throwable t) {
                    System.out.printf("[sampler-output-characterization] provider=%s failed: %s%n",
                            provider.name(), t.toString());
                }
            }

            return results;
        }
    }

    private List<ProjectionResult> characterizeProvider(String scenarioName, OpsCase provider, AbstractTensor embedding,
            AbstractTensor q4Weights, int vocabularySize, int embeddingLength, int repetitions, List<Integer> splitCounts,
            WrappedForkJoinPool pool, float[] referenceLogits) {
        List<ProjectionResult> results = new ArrayList<>();
        TensorOperations ops = provider.operations();

        try (AbstractTensor logits = allocator.getDirty(DType.F32, TensorShape.of(1, vocabularySize))) {
            prewarmCurrentPath(ops, embedding, q4Weights, logits, vocabularySize, embeddingLength,
                    ops.parallelSplitSize(), pool);

            long currentNanos = timeChunked(ops, embedding, q4Weights, logits, vocabularySize, embeddingLength,
                    repetitions, ops.parallelSplitSize(), pool);
            float[] currentLogits = logits(logits, vocabularySize);
            if (referenceLogits == null) {
                referenceLogits = currentLogits;
            }
            results.add(result(scenarioName, provider.name(), "current_parallelSplitSize", embedding.dType(),
                    ops.parallelSplitSize(), repetitions, currentNanos, referenceLogits, currentLogits));

            long singleCallNanos = timeSingleProviderCall(ops, embedding, q4Weights, logits, vocabularySize,
                    embeddingLength, repetitions);
            results.add(result(scenarioName, provider.name(), "single_provider_call", embedding.dType(), 1,
                    repetitions, singleCallNanos, referenceLogits, logits(logits, vocabularySize)));

            for (int splitCount : splitCounts) {
                long fixedSplitNanos = timeChunked(ops, embedding, q4Weights, logits, vocabularySize, embeddingLength,
                        repetitions, splitCount, pool);
                results.add(result(scenarioName, provider.name(), "fixed_split", embedding.dType(), splitCount,
                        repetitions, fixedSplitNanos, referenceLogits, logits(logits, vocabularySize)));
            }

            long quantizeAndProjectNanos = timeQuantizeEmbeddingEachProjection(ops, embedding, q4Weights, logits,
                    vocabularySize, embeddingLength, repetitions, ops.parallelSplitSize(), pool);
            results.add(result(scenarioName, provider.name(), "i8_quantize_each_current", DType.I8,
                    ops.parallelSplitSize(), repetitions, quantizeAndProjectNanos, referenceLogits,
                    logits(logits, vocabularySize)));

            try (AbstractTensor i8Embedding = AbstractTensorUtils.quantize(embedding, DType.I8, true)) {
                long prequantizedNanos = timeChunked(ops, i8Embedding, q4Weights, logits, vocabularySize,
                        embeddingLength, repetitions, ops.parallelSplitSize(), pool);
                results.add(result(scenarioName, provider.name(), "i8_prequantized_current", DType.I8,
                        ops.parallelSplitSize(), repetitions, prequantizedNanos, referenceLogits,
                        logits(logits, vocabularySize)));
            }
        }

        return results;
    }

    private void prewarmCurrentPath(TensorOperations ops, AbstractTensor embedding, AbstractTensor weights,
            AbstractTensor logits, int vocabularySize, int embeddingLength, int splitCount, WrappedForkJoinPool pool) {
        int splits = Math.max(1, splitCount);
        for (int warmup = 0; warmup < initialWarmupRepetitions(); warmup++) {
            logits.clear();
            VectorMath.pchunk(0, vocabularySize, (chunkStart, chunkSize) -> ops.dotProductChunk(logits, embedding,
                    weights, 0, embeddingLength, chunkStart, chunkSize), splits, pool);
        }
    }

    private long timeSingleProviderCall(TensorOperations ops, AbstractTensor embedding, AbstractTensor weights,
            AbstractTensor logits, int vocabularySize, int embeddingLength, int repetitions) {
        for (int warmup = 0; warmup < warmupRepetitions(); warmup++) {
            logits.clear();
            ops.dotProductChunk(logits, embedding, weights, 0, embeddingLength, 0, vocabularySize);
        }
        long nanos = 0;
        for (int repetition = 0; repetition < repetitions; repetition++) {
            logits.clear();
            long start = System.nanoTime();
            ops.dotProductChunk(logits, embedding, weights, 0, embeddingLength, 0, vocabularySize);
            nanos += System.nanoTime() - start;
        }
        return nanos;
    }

    private long timeChunked(TensorOperations ops, AbstractTensor embedding, AbstractTensor weights,
            AbstractTensor logits, int vocabularySize, int embeddingLength, int repetitions, int splitCount,
            WrappedForkJoinPool pool) {
        long nanos = 0;
        int splits = Math.max(1, splitCount);
        for (int warmup = 0; warmup < warmupRepetitions(); warmup++) {
            logits.clear();
            VectorMath.pchunk(0, vocabularySize, (chunkStart, chunkSize) -> ops.dotProductChunk(logits, embedding,
                    weights, 0, embeddingLength, chunkStart, chunkSize), splits, pool);
        }
        for (int repetition = 0; repetition < repetitions; repetition++) {
            logits.clear();
            long start = System.nanoTime();
            VectorMath.pchunk(0, vocabularySize, (chunkStart, chunkSize) -> ops.dotProductChunk(logits, embedding,
                    weights, 0, embeddingLength, chunkStart, chunkSize), splits, pool);
            nanos += System.nanoTime() - start;
        }
        return nanos;
    }

    private long timeQuantizeEmbeddingEachProjection(TensorOperations ops, AbstractTensor embedding, AbstractTensor weights,
            AbstractTensor logits, int vocabularySize, int embeddingLength, int repetitions, int splitCount,
            WrappedForkJoinPool pool) {
        long nanos = 0;
        int splits = Math.max(1, splitCount);
        for (int warmup = 0; warmup < warmupRepetitions(); warmup++) {
            logits.clear();
            try (AbstractTensor i8Embedding = AbstractTensorUtils.quantize(embedding, DType.I8, true)) {
                VectorMath.pchunk(0, vocabularySize, (chunkStart, chunkSize) -> ops.dotProductChunk(logits,
                        i8Embedding, weights, 0, embeddingLength, chunkStart, chunkSize), splits, pool);
            }
        }
        for (int repetition = 0; repetition < repetitions; repetition++) {
            logits.clear();
            long start = System.nanoTime();
            try (AbstractTensor i8Embedding = AbstractTensorUtils.quantize(embedding, DType.I8, true)) {
                VectorMath.pchunk(0, vocabularySize, (chunkStart, chunkSize) -> ops.dotProductChunk(logits,
                        i8Embedding, weights, 0, embeddingLength, chunkStart, chunkSize), splits, pool);
            }
            nanos += System.nanoTime() - start;
        }
        return nanos;
    }

    private ProjectionResult result(String scenarioName, String providerName, String formulation, DType inputDType,
            int splitCount, int repetitions, long nanos, float[] reference, float[] actual) {
        InferenceProfiler.timer(metricRegistry, "sampler.characterization." + scenarioName + "." + providerName
                + "." + formulation).update(nanos, TimeUnit.NANOSECONDS);
        return new ProjectionResult(scenarioName, providerName, formulation, inputDType, splitCount, repetitions,
                TimeUnit.NANOSECONDS.toMicros(nanos), maxAbsError(reference, actual), actual);
    }

    private OpsCase panama(WrappedForkJoinPool pool) {
        return new OpsCase("panama", new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool));
    }

    private static float[] logits(AbstractTensor tensor, int vocabularySize) {
        float[] values = new float[vocabularySize];
        for (int token = 0; token < vocabularySize; token++) {
            values[token] = tensor.get(0, token);
        }
        return values;
    }

    private static float maxAbsError(float[] expected, float[] actual) {
        float max = 0.0f;
        for (int i = 0; i < expected.length; i++) {
            max = Math.max(max, Math.abs(expected[i] - actual[i]));
        }
        return max;
    }

    private static int warmupRepetitions() {
        return Math.max(1, TIMING_WARMUPS);
    }

    private static int initialWarmupRepetitions() {
        return Math.max(0, INITIAL_WARMUPS);
    }

    private record OpsCase(String name, TensorOperations operations) {
    }

    private record ProjectionResult(String scenario, String provider, String formulation, DType inputDType,
            int splitCount, int repetitions, long totalMicros, float maxAbsError, float[] logits) {
        String name() {
            return scenario + "." + provider + "." + formulation + "." + inputDType + ".splits_" + splitCount;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "scenario=%s provider=%s formulation=%s input=%s splits=%d repetitions=%d total_ms=%.3f mean_us=%.3f max_abs_error=%.6f",
                    scenario, provider, formulation, inputDType, splitCount, repetitions, totalMicros / 1000.0,
                    totalMicros / (double) repetitions, maxAbsError);
        }
    }
}
