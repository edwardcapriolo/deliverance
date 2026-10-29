package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.GroupedQkvProjection;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.NativeOps;
import io.teknek.deliverance.tensor2.TensorProviderKind;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Characterizes the Qwen3-0.6B BF16 grouped decode QKV shape and chunking. */
class Qwen3GroupedQkvProjectionCharacterizationTest {
    private static final int EMBEDDING_LENGTH = 1024;
    private static final int ATTENTION_LENGTH = 2048;
    private static final int KV_LENGTH = 1024;
    private static final int SPLIT_SIZE = 32;

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    void qwen06BGroupedDecodeBf16Completes() {
        Assumptions.assumeTrue(NativeOps.isAvailable(), "native Tensor2 unavailable");
        MetricRegistry metrics = new MetricRegistry();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(10));
             TensorRef input = new Lighter(metrics, pool.getUnderlying()).allocate(DType.BF16,
                     TensorShape.of(1, EMBEDDING_LENGTH));
             TensorRef query = new Lighter(metrics, pool.getUnderlying()).allocate(DType.F32,
                     TensorShape.of(1, ATTENTION_LENGTH));
             TensorRef key = new Lighter(metrics, pool.getUnderlying()).allocate(DType.F32,
                     TensorShape.of(1, KV_LENGTH));
             TensorRef value = new Lighter(metrics, pool.getUnderlying()).allocate(DType.F32,
                     TensorShape.of(1, KV_LENGTH))) {
            Lighter lighter = new Lighter(metrics, pool.getUnderlying());
            lighter.putTensorOperations(TensorProviderKind.SIMD, new NativeOps());
            try (TensorRef queryWeights = lighter.allocate(DType.BF16,
                         TensorShape.of(ATTENTION_LENGTH, EMBEDDING_LENGTH));
                 TensorRef keyWeights = lighter.allocate(DType.BF16,
                         TensorShape.of(KV_LENGTH, EMBEDDING_LENGTH));
                 TensorRef valueWeights = lighter.allocate(DType.BF16,
                         TensorShape.of(KV_LENGTH, EMBEDDING_LENGTH))) {
                VectorMath.pchunk(0, ATTENTION_LENGTH, (offset, length) ->
                                new GroupedQkvProjection(input, query, key, value, queryWeights, keyWeights,
                                        valueWeights, EMBEDDING_LENGTH, offset, length, KV_LENGTH).execute(lighter),
                        SPLIT_SIZE, pool);
            }
            assertEquals(ATTENTION_LENGTH, query.shape().last());
            assertEquals(KV_LENGTH, key.shape().last());
            assertEquals(KV_LENGTH, value.shape().last());
        }
    }
}
