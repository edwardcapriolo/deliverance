package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdQwenTpShapeTest {

    @ParameterizedTest(name = "{0} batch={1} in={2} rows={3}")
    @MethodSource("qwenProjectionShapes")
    public void nativeSimdQwenTpProjectionShapesMatchPanama(String name, int batchSize, int inputColumns, int rows,
            DType inputType, DType weightType, float tolerance) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);
            try (FloatBufferTensor denseInput = deterministicInput(batchSize, inputColumns);
                 FloatBufferTensor denseWeight = deterministicWeight(rows, inputColumns);
                 AbstractTensor input = convertInput(denseInput, inputType);
                 AbstractTensor weight = convertWeight(denseWeight, weightType);
                 FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
                 FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {

                panama.registerModelTensor(weight);
                panama.batchDotProduct(expected, input, weight, 0, 0, inputColumns, 0, 0, rows);
                simd.registerModelTensor(weight);
                simd.batchDotProduct(actual, input, weight, 0, 0, inputColumns, 0, 0, rows);

                assertTensorClose(expected, actual, tolerance, name);
            }
        }
    }

    @ParameterizedTest(name = "dotProductChunk {0} batch={1} in={2} rows={3} chunkStart={4} chunk={5}")
    @MethodSource("qwenDotProductChunkShapes")
    public void nativeSimdQwenTpDotProductChunkShapesMatchPanama(String name, int batchSize, int inputColumns,
            int rows, int chunkStart, int chunkSize, DType inputType, DType weightType, float tolerance) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);
            try (FloatBufferTensor denseInput = deterministicInput(batchSize, inputColumns);
                 FloatBufferTensor denseWeight = deterministicWeight(rows, inputColumns);
                 AbstractTensor input = convertInput(denseInput, inputType);
                 AbstractTensor weight = convertWeight(denseWeight, weightType);
                 FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
                 FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {

                panama.registerModelTensor(weight);
                panama.dotProductChunk(expected, input, weight, 0, inputColumns, chunkStart, chunkSize);
                simd.registerModelTensor(weight);
                simd.dotProductChunk(actual, input, weight, 0, inputColumns, chunkStart, chunkSize);

                assertTensorClose(expected, actual, tolerance, name);
            }
        }
    }

    @ParameterizedTest(name = "{0} batch={1} in={2} fullRows={3} shardStart={4} shardRows={5}")
    @MethodSource("qwenRowShardShapes")
    public void nativeSimdQwenTpRowShardMatchesFullProjection(String name, int batchSize, int inputColumns,
            int fullRows, int shardStart, int shardRows, DType inputType, DType weightType, float tolerance) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations simd = new NativeSimdTensorOperations(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    allocator, pool));
            try (FloatBufferTensor denseInput = deterministicInput(batchSize, inputColumns);
                 FloatBufferTensor denseWeight = deterministicWeight(fullRows, inputColumns);
                 AbstractTensor input = convertInput(denseInput, inputType);
                 AbstractTensor fullWeight = convertWeight(denseWeight, weightType);
                 AbstractTensor shardWeight = convertWeight(rowShard(denseWeight, shardStart, shardRows), weightType);
                 FloatBufferTensor fullOutput = new FloatBufferTensor(batchSize, fullRows);
                 FloatBufferTensor shardOutput = new FloatBufferTensor(batchSize, shardRows)) {

                simd.registerModelTensor(fullWeight);
                simd.batchDotProduct(fullOutput, input, fullWeight, 0, 0, inputColumns, 0, 0, fullRows);
                simd.registerModelTensor(shardWeight);
                simd.batchDotProduct(shardOutput, input, shardWeight, 0, 0, inputColumns, 0, 0, shardRows);

                for (int row = 0; row < batchSize; row++) {
                    for (int col = 0; col < shardRows; col++) {
                        assertEquals(fullOutput.get(row, shardStart + col), shardOutput.get(row, col), tolerance,
                                name + " row=" + row + " col=" + col);
                    }
                }
            }
        }
    }

    @ParameterizedTest(name = "{0} batch={1} hidden={2} out={3} shardStart={4} shardLength={5}")
    @MethodSource("qwenColumnShardShapes")
    public void nativeSimdQwenTpColumnShardContributionMatchesPanamaFullContribution(String name, int batchSize,
            int hiddenColumns, int outputRows, int shardStart, int shardLength, DType inputType, DType weightType,
            float tolerance) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);
            try (FloatBufferTensor denseInput = deterministicInput(batchSize, hiddenColumns);
                 FloatBufferTensor denseWeight = deterministicWeight(outputRows, hiddenColumns);
                 AbstractTensor fullInput = convertInput(denseInput, inputType);
                 AbstractTensor shardInput = convertInput(columnShard(denseInput, shardStart, shardLength), inputType);
                 AbstractTensor fullWeight = convertWeight(denseWeight, weightType);
                 AbstractTensor shardWeight = convertWeight(columnShard(denseWeight, shardStart, shardLength), weightType);
                 FloatBufferTensor expected = new FloatBufferTensor(batchSize, outputRows);
                 FloatBufferTensor actual = new FloatBufferTensor(batchSize, outputRows)) {

                panama.registerModelTensor(fullWeight);
                panama.dotProductChunk(expected, fullInput, fullWeight, shardStart, shardLength, 0, outputRows);
                simd.registerModelTensor(shardWeight);
                simd.dotProductChunk(actual, shardInput, shardWeight, 0, shardLength, 0, outputRows);

                assertTensorClose(expected, actual, tolerance, name);
            }
        }
    }

    private static Stream<Arguments> qwenProjectionShapes() {
        return Stream.of(
                Arguments.of("q_proj_i8q4", 10, 1024, 256, DType.I8, DType.Q4, 0.20f),
                Arguments.of("kv_proj_i8q4", 10, 1024, 128, DType.I8, DType.Q4, 0.20f),
                Arguments.of("mlp_gate_i8q4", 10, 1024, 768, DType.I8, DType.Q4, 0.20f),
                Arguments.of("attention_output_i8q4", 10, 256, 1024, DType.I8, DType.Q4, 0.20f),
                Arguments.of("mlp_down_i8q4", 10, 768, 1024, DType.I8, DType.Q4, 0.20f),
                Arguments.of("q_proj_f32q4", 10, 1024, 256, DType.F32, DType.Q4, 0.08f),
                Arguments.of("mlp_gate_f32q4", 10, 1024, 768, DType.F32, DType.Q4, 0.08f)
        );
    }

    private static Stream<Arguments> qwenDotProductChunkShapes() {
        return Stream.of(
                Arguments.of("q_proj_full_chunk_i8q4", 10, 1024, 256, 0, 256, DType.I8, DType.Q4, 0.20f),
                Arguments.of("k_proj_full_chunk_i8q4", 10, 1024, 128, 0, 128, DType.I8, DType.Q4, 0.20f),
                Arguments.of("v_proj_full_chunk_i8q4", 10, 1024, 128, 0, 128, DType.I8, DType.Q4, 0.20f),
                Arguments.of("q_proj_half_chunk_i8q4", 10, 1024, 256, 128, 128, DType.I8, DType.Q4, 0.20f),
                Arguments.of("mlp_gate_chunk_i8q4", 10, 1024, 768, 0, 768, DType.I8, DType.Q4, 0.20f),
                Arguments.of("q_proj_full_chunk_f32q4", 10, 1024, 256, 0, 256, DType.F32, DType.Q4, 0.08f),
                Arguments.of("k_proj_full_chunk_f32q4", 10, 1024, 128, 0, 128, DType.F32, DType.Q4, 0.08f)
        );
    }

    private static Stream<Arguments> qwenRowShardShapes() {
        return Stream.of(
                Arguments.of("q_proj_rank0", 10, 1024, 1024, 0, 256, DType.I8, DType.Q4, 0.20f),
                Arguments.of("q_proj_rank3", 10, 1024, 1024, 768, 256, DType.I8, DType.Q4, 0.20f),
                Arguments.of("mlp_gate_rank2", 10, 1024, 3072, 1536, 768, DType.I8, DType.Q4, 0.20f)
        );
    }

    private static Stream<Arguments> qwenColumnShardShapes() {
        return Stream.of(
                Arguments.of("attention_o_rank0", 10, 1024, 1024, 0, 256, DType.I8, DType.Q4, 0.20f),
                Arguments.of("attention_o_rank3", 10, 1024, 1024, 768, 256, DType.I8, DType.Q4, 0.20f),
                Arguments.of("mlp_down_rank0", 10, 3072, 1024, 0, 768, DType.I8, DType.Q4, 0.20f),
                Arguments.of("mlp_down_rank3", 10, 3072, 1024, 2304, 768, DType.I8, DType.Q4, 0.20f),
                Arguments.of("attention_o_rank0_f32", 10, 1024, 1024, 0, 256, DType.F32, DType.Q4, 0.08f),
                Arguments.of("mlp_down_rank0_f32", 10, 3072, 1024, 0, 768, DType.F32, DType.Q4, 0.08f)
        );
    }

    private static AbstractTensor convertInput(AbstractTensor input, DType inputType) {
        if (inputType == DType.F32) {
            return new FloatBufferTensor(input);
        }
        return AbstractTensorUtils.quantize(input, inputType, true);
    }

    private static AbstractTensor convertWeight(AbstractTensor weight, DType weightType) {
        if (weightType == DType.F32) {
            return new FloatBufferTensor(weight);
        }
        return AbstractTensorUtils.quantize(weight, weightType, true);
    }

    private static FloatBufferTensor deterministicInput(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 17 + col * 31) % 257 - 128) / 64.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor deterministicWeight(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 43 + col * 19) % 251 - 125) / 80.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor rowShard(AbstractTensor source, int startInclusive, int length) {
        FloatBufferTensor shard = new FloatBufferTensor(length, source.shape().last());
        for (int row = 0; row < length; row++) {
            shard.copyFrom(source, source.getOffset(startInclusive + row, 0), shard.getOffset(row, 0),
                    source.shape().last());
        }
        return shard;
    }

    private static FloatBufferTensor columnShard(AbstractTensor source, int startInclusive, int length) {
        FloatBufferTensor shard = new FloatBufferTensor(source.shape().first(), length);
        for (int row = 0; row < source.shape().first(); row++) {
            shard.copyFrom(source, source.getOffset(row, startInclusive), shard.getOffset(row, 0), length);
        }
        return shard;
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance, String label) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        label + " row=" + row + " col=" + col);
            }
        }
    }
}
