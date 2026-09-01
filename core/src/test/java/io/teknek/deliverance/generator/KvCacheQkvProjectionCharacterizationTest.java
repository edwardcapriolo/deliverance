package io.teknek.deliverance.generator;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorRuntime;
import io.teknek.deliverance.tensorlib.TensorRuntimeMode;
import io.teknek.deliverance.tensorlib.TensorRuntimeNative;
import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.ForkJoinTask;

import static io.teknek.deliverance.tensor.TensorTestSupport.deterministicTensor;

class KvCacheQkvProjectionCharacterizationTest {
    private static final int BATCH_SIZE = 1;
    private static final int EMBEDDING_LENGTH = 1024;
    private static final int ATTENTION_LENGTH = 1024;
    private static final int KV_LENGTH = 1024;
    private static final int REPETITIONS = 32;
    private static final int SPLIT_SIZE = 64;
    private static final int POOL_SIZE = 16;

    @Test
    void qwen06GqaQkvProjectionCurrentShape() {
        MetricRegistry metricRegistry = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(8));
             AbstractTensor denseInput = deterministicTensor(BATCH_SIZE, EMBEDDING_LENGTH, 101);
             AbstractTensor input = AbstractTensorUtils.quantize(denseInput, DType.I8, true);
             AbstractTensor queryAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(ATTENTION_LENGTH, EMBEDDING_LENGTH, 201), DType.Q4, true);
             AbstractTensor keyAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(KV_LENGTH, EMBEDDING_LENGTH, 301), DType.Q4, true);
             AbstractTensor valueAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(KV_LENGTH, EMBEDDING_LENGTH, 401), DType.Q4, true);
             AbstractTensor queryBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, ATTENTION_LENGTH));
             AbstractTensor keyBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, KV_LENGTH));
             AbstractTensor valueBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, KV_LENGTH))) {

            TensorOperations ops = new NativeSimdTensorOperations(
                    new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool), SPLIT_SIZE);

            for (int i = 0; i < 4; i++) {
                runCurrentGqaQkv(metricRegistry, pool, ops, input, queryAttnWeights, keyAttnWeights, valueAttnWeights,
                        queryBatch, keyBatch, valueBatch);
            }

            long start = System.nanoTime();
            for (int i = 0; i < REPETITIONS; i++) {
                queryBatch.clear();
                keyBatch.clear();
                valueBatch.clear();
                runCurrentGqaQkv(metricRegistry, pool, ops, input, queryAttnWeights, keyAttnWeights, valueAttnWeights,
                        queryBatch, keyBatch, valueBatch);
            }
            double totalMs = (System.nanoTime() - start) / 1_000_000.0;
            System.out.printf(Locale.ROOT,
                    "[qkv-projection-characterization] batch=%d embedding=%d attention=%d kv=%d split=%d pool=%d repetitions=%d total_ms=%.3f mean_us=%.3f%n",
                    BATCH_SIZE, EMBEDDING_LENGTH, ATTENTION_LENGTH, KV_LENGTH, SPLIT_SIZE, POOL_SIZE, REPETITIONS,
                    totalMs, totalMs * 1000.0 / REPETITIONS);
            InferenceProfiler.printSummary("qwen06 qkv projection characterization", 20);
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void qwen06GqaQkvProjectionTensorRuntimeAnalyzeShape() {
        MetricRegistry metricRegistry = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(POOL_SIZE));
             TensorRuntime runtime = new TensorRuntime(POOL_SIZE, TensorRuntimeMode.ANALYZE,
                     TensorRuntimeNative.unavailable("qkv characterization"), metricRegistry);
             AbstractTensor denseInput = deterministicTensor(BATCH_SIZE, EMBEDDING_LENGTH, 101);
             AbstractTensor input = AbstractTensorUtils.quantize(denseInput, DType.I8, true);
             AbstractTensor queryAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(ATTENTION_LENGTH, EMBEDDING_LENGTH, 201), DType.Q4, true);
             AbstractTensor keyAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(KV_LENGTH, EMBEDDING_LENGTH, 301), DType.Q4, true);
             AbstractTensor valueAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(KV_LENGTH, EMBEDDING_LENGTH, 401), DType.Q4, true);
             AbstractTensor queryBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, ATTENTION_LENGTH));
             AbstractTensor keyBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, KV_LENGTH));
             AbstractTensor valueBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, KV_LENGTH))) {

            TensorOperations ops = new NativeSimdTensorOperations(
                    new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool), SPLIT_SIZE);

            for (int i = 0; i < 4; i++) {
                runTensorRuntimeGqaQkv(metricRegistry, pool, runtime, ops, input, queryAttnWeights, keyAttnWeights,
                        valueAttnWeights, queryBatch, keyBatch, valueBatch);
            }

            long start = System.nanoTime();
            for (int i = 0; i < REPETITIONS; i++) {
                queryBatch.clear();
                keyBatch.clear();
                valueBatch.clear();
                runTensorRuntimeGqaQkv(metricRegistry, pool, runtime, ops, input, queryAttnWeights, keyAttnWeights,
                        valueAttnWeights, queryBatch, keyBatch, valueBatch);
            }
            double totalMs = (System.nanoTime() - start) / 1_000_000.0;
            System.out.printf(Locale.ROOT,
                    "[qkv-projection-characterization] mode=tensorruntime_analyze batch=%d embedding=%d attention=%d kv=%d split=%d pool=%d repetitions=%d total_ms=%.3f mean_us=%.3f%n",
                    BATCH_SIZE, EMBEDDING_LENGTH, ATTENTION_LENGTH, KV_LENGTH, SPLIT_SIZE, POOL_SIZE, REPETITIONS,
                    totalMs, totalMs * 1000.0 / REPETITIONS);
            InferenceProfiler.printSummary("qwen06 qkv projection tensorruntime analyze characterization", 30);
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void qwen06GqaQkvProjectionSinglePchunkGroupedShape() {
        MetricRegistry metricRegistry = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(8));
             AbstractTensor denseInput = deterministicTensor(BATCH_SIZE, EMBEDDING_LENGTH, 101);
             AbstractTensor input = AbstractTensorUtils.quantize(denseInput, DType.I8, true);
             AbstractTensor queryAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(ATTENTION_LENGTH, EMBEDDING_LENGTH, 201), DType.Q4, true);
             AbstractTensor keyAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(KV_LENGTH, EMBEDDING_LENGTH, 301), DType.Q4, true);
             AbstractTensor valueAttnWeights = AbstractTensorUtils.quantize(
                     deterministicTensor(KV_LENGTH, EMBEDDING_LENGTH, 401), DType.Q4, true);
             AbstractTensor queryBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, ATTENTION_LENGTH));
             AbstractTensor keyBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, KV_LENGTH));
             AbstractTensor valueBatch = allocator.getDirty(DType.F32, TensorShape.of(BATCH_SIZE, KV_LENGTH))) {

            TensorOperations ops = new NativeSimdTensorOperations(
                    new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool), 8);

            for (int i = 0; i < 10; i++) {
                runSinglePchunkGroupedQkv(metricRegistry, pool, ops, input, queryAttnWeights, keyAttnWeights,
                        valueAttnWeights, queryBatch, keyBatch, valueBatch);
            }

            long start = System.nanoTime();
            for (int i = 0; i < REPETITIONS; i++) {
                queryBatch.clear();
                keyBatch.clear();
                valueBatch.clear();
                runSinglePchunkGroupedQkv(metricRegistry, pool, ops, input, queryAttnWeights, keyAttnWeights,
                        valueAttnWeights, queryBatch, keyBatch, valueBatch);
            }
            double totalMs = (System.nanoTime() - start) / 1_000_000.0;
            System.out.printf(Locale.ROOT,
                    "[qkv-projection-characterization] mode=single_pchunk_grouped batch=%d embedding=%d attention=%d kv=%d split=%d pool=%d repetitions=%d total_ms=%.3f mean_us=%.3f%n",
                    BATCH_SIZE, EMBEDDING_LENGTH, ATTENTION_LENGTH, KV_LENGTH, 8, 8, REPETITIONS,
                    totalMs, totalMs * 1000.0 / REPETITIONS);
            InferenceProfiler.printSummary("qwen06 qkv projection single pchunk grouped characterization", 30);
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    private static void runCurrentGqaQkv(MetricRegistry metricRegistry, WrappedForkJoinPool pool, TensorOperations ops,
            AbstractTensor input, AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights, AbstractTensor queryBatch, AbstractTensor keyBatch,
            AbstractTensor valueBatch) {
        try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention.qkv_projection").time()) {
            ForkJoinTask<?> queryTask = pool.getUnderlying().submit(() -> project(metricRegistry, pool, ops,
                    queryBatch, input, queryAttnWeights, EMBEDDING_LENGTH, ATTENTION_LENGTH,
                    "kvcacheselfattention.q_projection"));
            ForkJoinTask<?> keyTask = pool.getUnderlying().submit(() -> project(metricRegistry, pool, ops,
                    keyBatch, input, keyAttnWeights, EMBEDDING_LENGTH, KV_LENGTH,
                    "kvcacheselfattention.k_projection"));
            ForkJoinTask<?> valueTask = pool.getUnderlying().submit(() -> project(metricRegistry, pool, ops,
                    valueBatch, input, valueAttnWeights, EMBEDDING_LENGTH, KV_LENGTH,
                    "kvcacheselfattention.v_projection"));
            queryTask.join();
            keyTask.join();
            valueTask.join();
        }
    }

    private static void project(MetricRegistry metricRegistry, WrappedForkJoinPool pool, TensorOperations ops,
            AbstractTensor output, AbstractTensor input, AbstractTensor weight, int inputLength, int outputLength,
            String metricName) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, metricName).time()) {
            VectorMath.pchunk(0, outputLength, (chunkStart, chunkSize) ->
                    ops.dotProductChunk(output, input, weight, 0, inputLength, chunkStart, chunkSize),
                    SPLIT_SIZE, pool);
        }
    }

    private static void runTensorRuntimeGqaQkv(MetricRegistry metricRegistry, WrappedForkJoinPool pool,
            TensorRuntime runtime, TensorOperations ops, AbstractTensor input, AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights, AbstractTensor queryBatch,
            AbstractTensor keyBatch, AbstractTensor valueBatch) {
        try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention.qkv_projection.tensorruntime").time()) {
            ForkJoinTask<?> queryTask = pool.getUnderlying().submit(() -> projectTensorRuntime(metricRegistry, runtime,
                    ops, queryBatch, input, queryAttnWeights, EMBEDDING_LENGTH, ATTENTION_LENGTH,
                    "kvcacheselfattention.q_projection.tensorruntime"));
            ForkJoinTask<?> keyTask = pool.getUnderlying().submit(() -> projectTensorRuntime(metricRegistry, runtime,
                    ops, keyBatch, input, keyAttnWeights, EMBEDDING_LENGTH, KV_LENGTH,
                    "kvcacheselfattention.k_projection.tensorruntime"));
            ForkJoinTask<?> valueTask = pool.getUnderlying().submit(() -> projectTensorRuntime(metricRegistry, runtime,
                    ops, valueBatch, input, valueAttnWeights, EMBEDDING_LENGTH, KV_LENGTH,
                    "kvcacheselfattention.v_projection.tensorruntime"));
            queryTask.join();
            keyTask.join();
            valueTask.join();
        }
    }

    private static void projectTensorRuntime(MetricRegistry metricRegistry, TensorRuntime runtime, TensorOperations ops,
            AbstractTensor output, AbstractTensor input, AbstractTensor weight, int inputLength, int outputLength,
            String metricName) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, metricName).time()) {
            runtime.runChunks(metricName, 0, outputLength, SPLIT_SIZE, Optional.of(input), (chunkStart, chunkSize) ->
                    ops.dotProductChunk(output, input, weight, 0, inputLength, chunkStart, chunkSize));
        }
    }

    private static void runSinglePchunkGroupedQkv(MetricRegistry metricRegistry, WrappedForkJoinPool pool,
            TensorOperations ops, AbstractTensor input, AbstractTensor queryAttnWeights, AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights, AbstractTensor queryBatch, AbstractTensor keyBatch,
            AbstractTensor valueBatch) {
        try (Timer.Context ignoredQkv = InferenceProfiler.timer(metricRegistry,
                "kvcacheselfattention.qkv_projection.single_pchunk_grouped").time()) {
            VectorMath.pchunk(0, ATTENTION_LENGTH, (chunkStart, chunkSize) -> {
                try (Timer.Context ignoredQ = InferenceProfiler.timer(metricRegistry,
                        "kvcacheselfattention.q_projection.single_pchunk_grouped").time()) {
                    ops.dotProductChunk(queryBatch, input, queryAttnWeights, 0, EMBEDDING_LENGTH, chunkStart, chunkSize);
                }
                try (Timer.Context ignoredK = InferenceProfiler.timer(metricRegistry,
                        "kvcacheselfattention.k_projection.single_pchunk_grouped").time()) {
                    ops.dotProductChunk(keyBatch, input, keyAttnWeights, 0, EMBEDDING_LENGTH, chunkStart, chunkSize);
                }
                try (Timer.Context ignoredV = InferenceProfiler.timer(metricRegistry,
                        "kvcacheselfattention.v_projection.single_pchunk_grouped").time()) {
                    ops.dotProductChunk(valueBatch, input, valueAttnWeights, 0, EMBEDDING_LENGTH, chunkStart, chunkSize);
                }
            }, 8, pool);
        }
    }
}
