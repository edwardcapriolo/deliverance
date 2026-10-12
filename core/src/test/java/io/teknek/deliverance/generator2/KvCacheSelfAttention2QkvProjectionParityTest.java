package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Compares legacy and TensorRef grouped Q/K/V projection semantics at model-shaped chunk boundaries. */
class KvCacheSelfAttention2QkvProjectionParityTest {
    private static final int EMBEDDING_LENGTH = 32;
    private static final int ATTENTION_LENGTH = 13;
    private static final int KV_LENGTH = 5;
    private static final int SPLIT_COUNT = 4;

    /** Verifies grouped projection parity for each supported input dtype. */
    @ParameterizedTest(name = "grouped QKV projection {0}")
    @EnumSource(value = DType.class, names = {"F32", "BF16", "I8"})
    void groupedQkvProjectionMatchesLegacy(DType inputType) {
        MetricRegistry metrics = new MetricRegistry();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(4))) {
            ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
            TensorOperations legacyOps = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            Lighter lighter = new Lighter(metrics);
            try (AbstractTensor denseInput = matrix(1, EMBEDDING_LENGTH, 11);
                 AbstractTensor input = quantize(denseInput, inputType);
                 AbstractTensor denseQueryWeight = matrix(ATTENTION_LENGTH, EMBEDDING_LENGTH, 17);
                 AbstractTensor denseKeyWeight = matrix(KV_LENGTH, EMBEDDING_LENGTH, 23);
                 AbstractTensor denseValueWeight = matrix(KV_LENGTH, EMBEDDING_LENGTH, 29);
                 AbstractTensor queryWeight = AbstractTensorUtils.quantize(denseQueryWeight, DType.Q4, true);
                 AbstractTensor keyWeight = AbstractTensorUtils.quantize(denseKeyWeight, DType.Q4, true);
                 AbstractTensor valueWeight = AbstractTensorUtils.quantize(denseValueWeight, DType.Q4, true);
                 AbstractTensor legacyQuery = new FloatBufferTensor(TensorShape.of(1, ATTENTION_LENGTH));
                 AbstractTensor legacyKey = new FloatBufferTensor(TensorShape.of(1, KV_LENGTH));
                 AbstractTensor legacyValue = new FloatBufferTensor(TensorShape.of(1, KV_LENGTH));
                 TensorRef inputRef = TensorRef.borrowed(input);
                 TensorRef queryWeightRef = TensorRef.borrowed(queryWeight);
                 TensorRef keyWeightRef = TensorRef.borrowed(keyWeight);
                 TensorRef valueWeightRef = TensorRef.borrowed(valueWeight);
                 TensorRef queryRef = TensorRef.borrowed(legacyQuery);
                 TensorRef keyRef = TensorRef.borrowed(legacyKey);
                 TensorRef valueRef = TensorRef.borrowed(legacyValue)) {
                runLegacy(legacyOps, input, queryWeight, keyWeight, valueWeight,
                        legacyQuery, legacyKey, legacyValue, pool);
                runTensorRef(lighter, legacyOps, pool, inputRef, queryWeightRef, keyWeightRef, valueWeightRef,
                        queryRef, keyRef, valueRef);
                assertClose(legacyQuery, queryRef, tolerance(inputType), "query", inputType);
                assertClose(legacyKey, keyRef, tolerance(inputType), "key", inputType);
                assertClose(legacyValue, valueRef, tolerance(inputType), "value", inputType);
            }
        }
    }

    /** Runs the legacy direct pchunk grouped projection. */
    private static void runLegacy(TensorOperations operations, AbstractTensor input, AbstractTensor queryWeight,
            AbstractTensor keyWeight, AbstractTensor valueWeight, AbstractTensor query, AbstractTensor key,
            AbstractTensor value, WrappedForkJoinPool pool) {
        VectorMath.pchunk(0, ATTENTION_LENGTH, (offset, length) -> {
            operations.dotProductChunk(query, input, queryWeight, 0, EMBEDDING_LENGTH, offset, length);
            int kvLength = Math.min(length, KV_LENGTH - offset);
            if (kvLength > 0) {
                operations.dotProductChunk(key, input, keyWeight, 0, EMBEDDING_LENGTH, offset, kvLength);
                operations.dotProductChunk(value, input, valueWeight, 0, EMBEDDING_LENGTH, offset, kvLength);
            }
        }, SPLIT_COUNT, pool);
    }

    /** Runs the TensorRef fused graph using the same grouped chunk callback. */
    private static void runTensorRef(Lighter lighter, TensorOperations operations, WrappedForkJoinPool pool,
            TensorRef input, TensorRef queryWeight, TensorRef keyWeight, TensorRef valueWeight, TensorRef query,
            TensorRef key, TensorRef value) {
        TensorPlan plan = new TensorPlan(operations, pool, new MetricRegistry(), null, null, lighter);
        TensorPlan.Tensor inputNode = plan.input("input", input);
        TensorPlan.ImmutableTensor queryWeightNode = plan.immutable("queryWeight", queryWeight);
        TensorPlan.ImmutableTensor keyWeightNode = plan.immutable("keyWeight", keyWeight);
        TensorPlan.ImmutableTensor valueWeightNode = plan.immutable("valueWeight", valueWeight);
        TensorPlan.Tensor fused = plan.fuse("qkv", TensorShape.of(ATTENTION_LENGTH))
                .read("input", inputNode)
                .read("queryWeight", queryWeightNode)
                .read("keyWeight", keyWeightNode)
                .read("valueWeight", valueWeightNode)
                .write("query", plan.input("query", query))
                .read("key", plan.input("key", key))
                .read("value", plan.input("value", value))
                .splitCount(SPLIT_COUNT)
                .mapRef("grouped qkv", TensorPlan.TensorOp.CUSTOM, (context, offset, length) -> {
                    int chunkStart = Math.toIntExact(offset);
                    int chunkLength = Math.toIntExact(length);
                    lighter.dotProductRows(context.tensor("query"), context.tensor("input"),
                            context.tensor("queryWeight"), 0, EMBEDDING_LENGTH, chunkStart, chunkLength, chunkStart);
                    int kvChunkLength = Math.min(chunkLength, KV_LENGTH - chunkStart);
                    if (kvChunkLength > 0) {
                        lighter.dotProductRows(context.tensor("key"), context.tensor("input"),
                                context.tensor("keyWeight"), 0, EMBEDDING_LENGTH, chunkStart, kvChunkLength, chunkStart);
                        lighter.dotProductRows(context.tensor("value"), context.tensor("input"),
                                context.tensor("valueWeight"), 0, EMBEDDING_LENGTH, chunkStart, kvChunkLength, chunkStart);
                    }
                }).tensor();
        fused.materializeRef();
    }

    /** Converts a dense source tensor through the production dtype conversion path. */
    private static AbstractTensor quantize(AbstractTensor source, DType type) {
        return type == DType.F32 ? new FloatBufferTensor(source) : AbstractTensorUtils.quantize(source, type, true);
    }

    /** Creates deterministic matrix data for repeatable parity comparisons. */
    private static FloatBufferTensor matrix(int rows, int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 13 + column * 7 + seed) % 31 - 15) / 16.0f, row, column);
            }
        }
        return tensor;
    }

    /** Returns the tolerance appropriate to the input quantization. */
    private static float tolerance(DType inputType) {
        return inputType == DType.F32 ? 1.0e-5f : 5.0e-2f;
    }

    /** Compares a TensorRef output against its legacy tensor counterpart. */
    private static void assertClose(AbstractTensor expected, TensorRef actual, float tolerance, String label,
            DType inputType) {
        TensorRefBackedTensor actualTensor = new TensorRefBackedTensor(actual);
        assertEquals(expected.shape(), actualTensor.shape(), label + " shape " + inputType);
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.get(row, column), actualTensor.get(row, column), tolerance,
                        label + " inputType=" + inputType + " row=" + row + " column=" + column);
                assertTrue(Float.isFinite(actualTensor.get(row, column)),
                        label + " non-finite inputType=" + inputType);
            }
        }
    }
}
