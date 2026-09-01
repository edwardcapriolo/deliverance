package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdTensorOpsFuzzParityTest {
    private static final long SEED = 0x5eed0fL;

    @ParameterizedTest(name = "{0}")
    @MethodSource("gemmCases")
    public void nativeSimdGemmFamilyMatchesPanama(Case c) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations naive = new NaiveTensorOperations();
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);
            Optional<TensorOperations> gpu = gpu();

            int inputCols = alignToBlock(c.inputColumnOffset() + c.columnLength);
            int weightCols = alignToBlock(c.weightColumnOffset() + c.columnLength);
            int rows = c.bRowOffset + c.rowChunkSize + 3;
            int resultCols = c.rRowOffset + c.bRowOffset + c.rowChunkSize + 3;

            try (FloatBufferTensor denseInput = deterministicInput(c.batchSize, inputCols, c.seed);
                 FloatBufferTensor denseWeight0 = deterministicWeight(rows, weightCols, c.seed + 17);
                 FloatBufferTensor denseWeight1 = deterministicWeight(rows, weightCols, c.seed + 31);
                 AbstractTensor input = convertInput(denseInput, c.inputType);
                 AbstractTensor weight0 = convertWeight(denseWeight0, c.weightType);
                 AbstractTensor weight1 = convertWeight(denseWeight1, c.weightType)) {
                switch (c.op) {
                    case BATCH_DOT -> assertBatchDotProduct(c, naive, panama, simd, gpu, input, weight0, resultCols);
                    case DOT_CHUNK -> assertDotProductChunk(c, naive, panama, simd, gpu, input, weight0, resultCols);
                    case BATCH_CHUNK -> assertDotProductBatchChunk(c, naive, panama, simd, gpu, input, weight0, weight1, resultCols);
                }
            }
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("elementwiseCases")
    public void nativeSimdElementwiseFamilyMatchesPanama(ElementwiseCase c) {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
            TensorOperations naive = new NaiveTensorOperations();
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);
            Optional<TensorOperations> gpu = gpu();
            try (FloatBufferTensor panamaA = deterministicInput(1, c.size, c.seed);
                 FloatBufferTensor simdA = new FloatBufferTensor(panamaA);
                 FloatBufferTensor panamaB = deterministicInput(1, c.size, c.seed + 23);
                 FloatBufferTensor simdB = new FloatBufferTensor(panamaB)) {
                switch (c.op) {
                    case ACCUMULATE -> {
                        panama.accumulate(panamaA, panamaB, c.offset, c.length);
                        try (FloatBufferTensor naiveA = deterministicInput(1, c.size, c.seed);
                             FloatBufferTensor naiveB = deterministicInput(1, c.size, c.seed + 23)) {
                            naive.accumulate(naiveA, naiveB, c.offset, c.length);
                            assertTensorClose(naiveA, panamaA, 0.0001f, c + " panama");
                        }
                        simd.accumulate(simdA, simdB, c.offset, c.length);
                        assertTensorClose(panamaA, simdA, 0.0001f, c.toString());
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuA = deterministicInput(1, c.size, c.seed);
                                 FloatBufferTensor gpuB = deterministicInput(1, c.size, c.seed + 23)) {
                                ops.accumulate(gpuA, gpuB, c.offset, c.length);
                                assertTensorClose(panamaA, gpuA, 0.0001f, c + " gpu");
                            }
                        });
                    }
                    case MACCUMULATE -> {
                        panama.maccumulate(panamaA, panamaB, c.offset, c.length);
                        try (FloatBufferTensor naiveA = deterministicInput(1, c.size, c.seed);
                             FloatBufferTensor naiveB = deterministicInput(1, c.size, c.seed + 23)) {
                            naive.maccumulate(naiveA, naiveB, c.offset, c.length);
                            assertTensorClose(naiveA, panamaA, 0.0001f, c + " panama");
                        }
                        simd.maccumulate(simdA, simdB, c.offset, c.length);
                        assertTensorClose(panamaA, simdA, 0.0001f, c.toString());
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuA = deterministicInput(1, c.size, c.seed);
                                 FloatBufferTensor gpuB = deterministicInput(1, c.size, c.seed + 23)) {
                                ops.maccumulate(gpuA, gpuB, c.offset, c.length);
                                assertTensorClose(panamaA, gpuA, 0.0001f, c + " gpu");
                            }
                        });
                    }
                    case SAXPY -> {
                        panama.saxpy(1.75f, panamaA, panamaB, c.offset, c.offset, c.length);
                        try (FloatBufferTensor naiveA = deterministicInput(1, c.size, c.seed);
                             FloatBufferTensor naiveB = deterministicInput(1, c.size, c.seed + 23)) {
                            naive.saxpy(1.75f, naiveA, naiveB, c.offset, c.offset, c.length);
                            assertTensorClose(naiveB, panamaB, 0.0001f, c + " panama");
                        }
                        simd.saxpy(1.75f, simdA, simdB, c.offset, c.offset, c.length);
                        assertTensorClose(panamaB, simdB, 0.0001f, c.toString());
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuA = deterministicInput(1, c.size, c.seed);
                                 FloatBufferTensor gpuB = deterministicInput(1, c.size, c.seed + 23)) {
                                ops.saxpy(1.75f, gpuA, gpuB, c.offset, c.offset, c.length);
                                assertTensorClose(panamaB, gpuB, 0.0001f, c + " gpu");
                            }
                        });
                    }
                    case EXP -> {
                        try (FloatBufferTensor naiveOut = filled(1, c.size, -77.0f);
                             FloatBufferTensor panamaOut = filled(1, c.size, -77.0f);
                             FloatBufferTensor simdOut = filled(1, c.size, -77.0f)) {
                            naive.exp(panamaA, naiveOut, c.offset, c.length);
                            panama.exp(panamaA, panamaOut, c.offset, c.length);
                            assertTensorClose(naiveOut, panamaOut, 0.0001f, c + " panama");
                            simd.exp(simdA, simdOut, c.offset, c.length);
                            assertTensorClose(panamaOut, simdOut, 0.0001f, c.toString());
                        }
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuIn = deterministicInput(1, c.size, c.seed);
                                 FloatBufferTensor gpuOut = filled(1, c.size, -77.0f);
                                 FloatBufferTensor panamaOut = filled(1, c.size, -77.0f)) {
                                ops.exp(gpuIn, gpuOut, c.offset, c.length);
                                panama.exp(gpuIn, panamaOut, c.offset, c.length);
                                assertTensorClose(panamaOut, gpuOut, 0.0001f, c + " gpu");
                            }
                        });
                    }
                    case MAX -> {
                        try (FloatBufferTensor naiveInput = deterministicInput(3, c.size, c.seed);
                             FloatBufferTensor panamaInput = new FloatBufferTensor(naiveInput);
                             FloatBufferTensor simdInput = new FloatBufferTensor(naiveInput)) {
                            for (int row = 0; row < naiveInput.shape().first(); row++) {
                                float naiveMax = naive.max(naiveInput, row, c.offset, c.length);
                                float panamaMax = panama.max(panamaInput, row, c.offset, c.length);
                                float simdMax = simd.max(simdInput, row, c.offset, c.length);
                                assertEquals(naiveMax, panamaMax, 0.0f, c + " panama row=" + row);
                                assertEquals(panamaMax, simdMax, 0.0f, c + " simd row=" + row);
                            }
                        }
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuInput = deterministicInput(3, c.size, c.seed);
                                 FloatBufferTensor panamaInput = new FloatBufferTensor(gpuInput)) {
                                for (int row = 0; row < gpuInput.shape().first(); row++) {
                                    assertEquals(panama.max(panamaInput, row, c.offset, c.length),
                                            ops.max(gpuInput, row, c.offset, c.length), 0.0f, c + " gpu row=" + row);
                                }
                            }
                        });
                    }
                    case SUM -> {
                        try (FloatBufferTensor naiveInput = deterministicInput(3, c.size, c.seed);
                             FloatBufferTensor panamaInput = new FloatBufferTensor(naiveInput);
                             FloatBufferTensor simdInput = new FloatBufferTensor(naiveInput)) {
                            for (int row = 0; row < naiveInput.shape().first(); row++) {
                                float naiveSum = naive.sum(naiveInput, row, c.offset, c.length);
                                float panamaSum = panama.sum(panamaInput, row, c.offset, c.length);
                                float simdSum = simd.sum(simdInput, row, c.offset, c.length);
                                assertEquals(naiveSum, panamaSum, 1.0e-5f, c + " panama row=" + row);
                                assertEquals(panamaSum, simdSum, 1.0e-5f, c + " simd row=" + row);
                            }
                        }
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuInput = deterministicInput(3, c.size, c.seed);
                                 FloatBufferTensor panamaInput = new FloatBufferTensor(gpuInput)) {
                                for (int row = 0; row < gpuInput.shape().first(); row++) {
                                    assertEquals(panama.sum(panamaInput, row, c.offset, c.length),
                                            ops.sum(gpuInput, row, c.offset, c.length), 1.0e-5f,
                                            c + " gpu row=" + row);
                                }
                            }
                        });
                    }
                    case ARGMAX -> {
                        try (FloatBufferTensor naiveInput = deterministicInput(1, c.size, c.seed);
                             FloatBufferTensor panamaInput = new FloatBufferTensor(naiveInput);
                             FloatBufferTensor simdInput = new FloatBufferTensor(naiveInput);
                             FloatBufferTensor naiveOut = new FloatBufferTensor(1, 2);
                             FloatBufferTensor panamaOut = new FloatBufferTensor(1, 2);
                             FloatBufferTensor simdOut = new FloatBufferTensor(1, 2)) {
                            int tieIndex = c.offset + Math.max(0, c.length / 3);
                            int duplicateTieIndex = c.offset + Math.max(0, (c.length * 2) / 3);
                            naiveInput.set(99.0f, 0, duplicateTieIndex);
                            naiveInput.set(99.0f, 0, tieIndex);
                            panamaInput.set(99.0f, 0, duplicateTieIndex);
                            panamaInput.set(99.0f, 0, tieIndex);
                            simdInput.set(99.0f, 0, duplicateTieIndex);
                            simdInput.set(99.0f, 0, tieIndex);

                            naive.argMax(naiveInput, naiveOut, c.offset, c.length);
                            panama.argMax(panamaInput, panamaOut, c.offset, c.length);
                            simd.argMax(simdInput, simdOut, c.offset, c.length);
                            assertTensorClose(naiveOut, panamaOut, 0.0f, c + " panama");
                            assertTensorClose(panamaOut, simdOut, 0.0f, c.toString());
                        }
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuInput = deterministicInput(1, c.size, c.seed);
                                 FloatBufferTensor panamaInput = new FloatBufferTensor(gpuInput);
                                 FloatBufferTensor gpuOut = new FloatBufferTensor(1, 2);
                                 FloatBufferTensor panamaOut = new FloatBufferTensor(1, 2)) {
                                panama.argMax(panamaInput, panamaOut, c.offset, c.length);
                                ops.argMax(gpuInput, gpuOut, c.offset, c.length);
                                assertTensorClose(panamaOut, gpuOut, 0.0f, c + " gpu");
                            }
                        });
                    }
                    case SOFTMAX -> {
                        try (FloatBufferTensor naiveInput = deterministicInput(1, c.size, c.seed);
                             FloatBufferTensor panamaInput = new FloatBufferTensor(naiveInput);
                             FloatBufferTensor simdInput = new FloatBufferTensor(naiveInput)) {
                            naive.softMax(naiveInput, c.offset, c.length);
                            panama.softMax(panamaInput, c.offset, c.length);
                            simd.softMax(simdInput, c.offset, c.length);
                            assertTensorClose(naiveInput, panamaInput, 0.0001f, c + " panama");
                            assertTensorClose(panamaInput, simdInput, 0.0001f, c.toString());
                        }
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuInput = deterministicInput(1, c.size, c.seed);
                                 FloatBufferTensor panamaInput = new FloatBufferTensor(gpuInput)) {
                                panama.softMax(panamaInput, c.offset, c.length);
                                ops.softMax(gpuInput, c.offset, c.length);
                                assertTensorClose(panamaInput, gpuInput, 0.0001f, c + " gpu");
                            }
                        });
                    }
                    case ACTIVATION_MULTIPLY_QUANTIZE -> {
                        try (FloatBufferTensor naiveGate = deterministicInput(3, c.size, c.seed);
                             FloatBufferTensor naiveUp = deterministicInput(3, c.size, c.seed + 23);
                             FloatBufferTensor panamaGate = new FloatBufferTensor(naiveGate);
                             FloatBufferTensor panamaUp = new FloatBufferTensor(naiveUp);
                             FloatBufferTensor simdGate = new FloatBufferTensor(naiveGate);
                             FloatBufferTensor simdUp = new FloatBufferTensor(naiveUp);
                             AbstractTensor naiveOut = naive.activationMultiplyQuantize(naiveGate, naiveUp,
                                     ActivationFunction.Type.SILU, DType.I8, c.offset, c.length);
                             AbstractTensor panamaOut = panama.activationMultiplyQuantize(panamaGate, panamaUp,
                                     ActivationFunction.Type.SILU, DType.I8, c.offset, c.length);
                             AbstractTensor simdOut = simd.activationMultiplyQuantize(simdGate, simdUp,
                                     ActivationFunction.Type.SILU, DType.I8, c.offset, c.length)) {
                            assertTensorClose(naiveOut, panamaOut, 0.05f, c + " panama");
                            assertTensorClose(panamaOut, simdOut, 0.05f, c.toString());
                        }
                        gpu.ifPresent(ops -> {
                            try (FloatBufferTensor gpuGate = deterministicInput(3, c.size, c.seed);
                                 FloatBufferTensor gpuUp = deterministicInput(3, c.size, c.seed + 23);
                                 AbstractTensor gpuOut = ops.activationMultiplyQuantize(gpuGate, gpuUp,
                                         ActivationFunction.Type.SILU, DType.I8, c.offset, c.length);
                                 FloatBufferTensor panamaGate = deterministicInput(3, c.size, c.seed);
                                 FloatBufferTensor panamaUp = deterministicInput(3, c.size, c.seed + 23);
                                 AbstractTensor panamaOut = panama.activationMultiplyQuantize(panamaGate, panamaUp,
                                         ActivationFunction.Type.SILU, DType.I8, c.offset, c.length)) {
                                assertTensorClose(panamaOut, gpuOut, 0.05f, c + " gpu");
                            }
                        });
                    }
                }
            }
        }
    }

    private static void assertBatchDotProduct(Case c, TensorOperations naive, TensorOperations panama,
            TensorOperations simd, Optional<TensorOperations> gpu,
            AbstractTensor input, AbstractTensor weight, int resultCols) {
        try (FloatBufferTensor reference = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor expected = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor actual = new FloatBufferTensor(c.batchSize, resultCols)) {
            naive.batchDotProduct(reference, input, weight, c.aColumnOffset, c.bColumnOffset, c.columnLength,
                    c.rRowOffset, c.bRowOffset, c.rowChunkSize);
            panama.registerModelTensor(weight);
            panama.batchDotProduct(expected, input, weight, c.aColumnOffset, c.bColumnOffset, c.columnLength,
                    c.rRowOffset, c.bRowOffset, c.rowChunkSize);
            assertTensorClose(reference, expected, c.tolerance(), c + " panama");
            simd.registerModelTensor(weight);
            simd.batchDotProduct(actual, input, weight, c.aColumnOffset, c.bColumnOffset, c.columnLength,
                    c.rRowOffset, c.bRowOffset, c.rowChunkSize);
            assertTensorClose(expected, actual, c.tolerance(), c.toString());
            gpu.ifPresent(ops -> {
                try (FloatBufferTensor gpuActual = new FloatBufferTensor(c.batchSize, resultCols)) {
                    ops.registerModelTensor(weight);
                    ops.batchDotProduct(gpuActual, input, weight, c.aColumnOffset, c.bColumnOffset, c.columnLength,
                            c.rRowOffset, c.bRowOffset, c.rowChunkSize);
                    assertTensorClose(expected, gpuActual, c.tolerance(), c + " gpu");
                }
            });
        }
    }

    private static void assertDotProductChunk(Case c, TensorOperations naive, TensorOperations panama,
            TensorOperations simd, Optional<TensorOperations> gpu,
            AbstractTensor input, AbstractTensor weight, int resultCols) {
        try (FloatBufferTensor reference = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor expected = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor actual = new FloatBufferTensor(c.batchSize, resultCols)) {
            naive.dotProductChunk(reference, input, weight, c.aColumnOffset, c.columnLength,
                    c.bRowOffset, c.rowChunkSize);
            panama.registerModelTensor(weight);
            panama.dotProductChunk(expected, input, weight, c.aColumnOffset, c.columnLength,
                    c.bRowOffset, c.rowChunkSize);
            assertTensorClose(reference, expected, c.tolerance(), c + " panama");
            simd.registerModelTensor(weight);
            simd.dotProductChunk(actual, input, weight, c.aColumnOffset, c.columnLength,
                    c.bRowOffset, c.rowChunkSize);
            assertTensorClose(expected, actual, c.tolerance(), c.toString());
            gpu.ifPresent(ops -> {
                try (FloatBufferTensor gpuActual = new FloatBufferTensor(c.batchSize, resultCols)) {
                    ops.registerModelTensor(weight);
                    ops.dotProductChunk(gpuActual, input, weight, c.aColumnOffset, c.columnLength,
                            c.bRowOffset, c.rowChunkSize);
                    assertTensorClose(expected, gpuActual, c.tolerance(), c + " gpu");
                }
            });
        }
    }

    private static void assertDotProductBatchChunk(Case c, TensorOperations naive, TensorOperations panama,
            TensorOperations simd, Optional<TensorOperations> gpu,
            AbstractTensor input, AbstractTensor weight0, AbstractTensor weight1, int resultCols) {
        try (FloatBufferTensor reference0 = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor reference1 = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor expected0 = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor expected1 = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor actual0 = new FloatBufferTensor(c.batchSize, resultCols);
             FloatBufferTensor actual1 = new FloatBufferTensor(c.batchSize, resultCols)) {
            naive.dotProductBatchChunk(new AbstractTensor[]{reference0, reference1}, input,
                    new AbstractTensor[]{weight0, weight1}, c.aColumnOffset, c.columnLength,
                    c.bRowOffset, c.rowChunkSize);
            panama.registerModelTensor(weight0);
            panama.registerModelTensor(weight1);
            panama.dotProductBatchChunk(new AbstractTensor[]{expected0, expected1}, input,
                    new AbstractTensor[]{weight0, weight1}, c.aColumnOffset, c.columnLength,
                    c.bRowOffset, c.rowChunkSize);
            assertTensorClose(reference0, expected0, c.tolerance(), c + " panama weight0");
            assertTensorClose(reference1, expected1, c.tolerance(), c + " panama weight1");
            simd.registerModelTensor(weight0);
            simd.registerModelTensor(weight1);
            simd.dotProductBatchChunk(new AbstractTensor[]{actual0, actual1}, input,
                    new AbstractTensor[]{weight0, weight1}, c.aColumnOffset, c.columnLength,
                    c.bRowOffset, c.rowChunkSize);
            assertTensorClose(expected0, actual0, c.tolerance(), c + " weight0");
            assertTensorClose(expected1, actual1, c.tolerance(), c + " weight1");
            gpu.ifPresent(ops -> {
                try (FloatBufferTensor gpuActual0 = new FloatBufferTensor(c.batchSize, resultCols);
                     FloatBufferTensor gpuActual1 = new FloatBufferTensor(c.batchSize, resultCols)) {
                    ops.registerModelTensor(weight0);
                    ops.registerModelTensor(weight1);
                    ops.dotProductBatchChunk(new AbstractTensor[]{gpuActual0, gpuActual1}, input,
                            new AbstractTensor[]{weight0, weight1}, c.aColumnOffset, c.columnLength,
                            c.bRowOffset, c.rowChunkSize);
                    assertTensorClose(expected0, gpuActual0, c.tolerance(), c + " gpu weight0");
                    assertTensorClose(expected1, gpuActual1, c.tolerance(), c + " gpu weight1");
                }
            });
        }
    }

    private static Optional<TensorOperations> gpu() {
        try {
            return Optional.of(new NativeGPUTensorOperations());
        } catch (Throwable t) {
            return Optional.empty();
        }
    }

    private static Stream<Arguments> gemmCases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        for (Op op : Op.values()) {
            cases.add(new Case("qwen_q_" + op, op, 10, 1024, 0, 0, 0, 0, 256, DType.I8, DType.Q4, id++));
            cases.add(new Case("qwen_kv_" + op, op, 10, 1024, 0, 0, 0, 0, 128, DType.I8, DType.Q4, id++));
            cases.add(new Case("qwen_mlp_gate_" + op, op, 10, 1024, 0, 0, 0, 0, 768, DType.I8, DType.Q4, id++));
            cases.add(new Case("qwen_down_" + op, op, 10, 768, 0, 0, 0, 0, 1024, DType.I8, DType.Q4, id++));
        }
        int[] f32Q4Batches = {1, 13};
        int[] f32Q4ColumnOffsets = {0, 32, 64};
        int[] f32Q4Lengths = {32, 64, 96};
        int[] f32Q4RowOffsets = {0, 4};
        int[] f32Q4RowChunks = {1, 2, 4, 5, 13};
        for (int batch : f32Q4Batches) {
            for (int columnOffset : f32Q4ColumnOffsets) {
                for (int columnLength : f32Q4Lengths) {
                    for (int rowOffset : f32Q4RowOffsets) {
                        for (int rowChunk : f32Q4RowChunks) {
                            cases.add(new Case("focused_f32_q4_dot_chunk_batch" + batch
                                    + "_offset" + columnOffset
                                    + "_k" + columnLength
                                    + "_row" + rowOffset
                                    + "_chunk" + rowChunk,
                                    Op.DOT_CHUNK, batch, columnLength, columnOffset, columnOffset,
                                    0, rowOffset, rowChunk, DType.F32, DType.Q4, id++));
                        }
                    }
                }
            }
        }
        cases.add(new Case("sampler_lm_head_first_chunk_f32q4", Op.DOT_CHUNK,
                1, 1024, 0, 0, 0, 0, 64, DType.F32, DType.Q4, id++));
        cases.add(new Case("sampler_lm_head_mid_chunk_f32q4", Op.DOT_CHUNK,
                1, 1024, 0, 0, 0, 2048, 64, DType.F32, DType.Q4, id++));
        cases.add(new Case("sampler_lm_head_tail_chunk_f32q4", Op.DOT_CHUNK,
                1, 1024, 0, 0, 0, 4032, 64, DType.F32, DType.Q4, id++));
        cases.add(new Case("regression_i8_q4_batch_chunk_batch13_k96_row128", Op.BATCH_CHUNK,
                13, 96, 0, 32, 2, 0, 128, DType.I8, DType.Q4, -931254078));
        int[] batchRows = {1, 2, 3, 5, 10, 13};
        int[] rowChunks = {1, 2, 3, 5, 7, 13, 16, 21, 128, 256};
        int[] offsets = {0, 1, 7, 31, 32, 64};
        int[] lengths = {1, 2, 3, 7, 16, 31, 32, 33, 64, 95, 96, 127, 128, 129, 256, 768, 1024};
        DType[][] dtypePairs = {
                {DType.F32, DType.F32},
                {DType.F32, DType.BF16},
                {DType.BF16, DType.BF16},
                {DType.F32, DType.Q4},
                {DType.F32, DType.I8},
                {DType.BF16, DType.Q4},
                {DType.I8, DType.Q4}
        };
        Random random = new Random(SEED);
        for (int i = 0; i < 72; i++) {
            Op op = Op.values()[random.nextInt(Op.values().length)];
            DType[] pair = dtypePairs[random.nextInt(dtypePairs.length)];
            int aOffset = offsets[random.nextInt(offsets.length)];
            int bOffset = offsets[random.nextInt(offsets.length)];
            int columnLength = lengths[random.nextInt(lengths.length)];
            if (pair[0] == DType.I8 || pair[1] == DType.Q4 || pair[1] == DType.I8) {
                columnLength = lengths[8 + random.nextInt(lengths.length - 8)];
            }
            int bRowOffset = random.nextInt(5);
            int rRowOffset = op == Op.BATCH_DOT ? bRowOffset + random.nextInt(3) : random.nextInt(3);
            cases.add(new Case("fuzz_" + i + "_" + op, op,
                    batchRows[random.nextInt(batchRows.length)], columnLength,
                    aOffset, bOffset, rRowOffset, bRowOffset,
                    rowChunks[random.nextInt(rowChunks.length)], pair[0], pair[1], random.nextInt()));
        }
        return cases.stream().map(Arguments::of);
    }

    private static Stream<Arguments> elementwiseCases() {
        List<ElementwiseCase> cases = new ArrayList<>();
        int id = 0;
        int[] sizes = {1, 2, 3, 7, 16, 31, 32, 33, 64, 127, 128, 257, 1024};
        for (ElementwiseOp op : ElementwiseOp.values()) {
            for (int size : sizes) {
                if (op == ElementwiseOp.ACTIVATION_MULTIPLY_QUANTIZE && size < 32) {
                    continue;
                }
                if (op == ElementwiseOp.ACTIVATION_MULTIPLY_QUANTIZE && (3 * size) % 32 != 0) {
                    continue;
                }
                int offset = size == 1 ? 0 : Math.min(size - 1, id % Math.max(1, size / 2));
                int length = Math.max(1, size - offset);
                if (op == ElementwiseOp.ACTIVATION_MULTIPLY_QUANTIZE) {
                    offset = (offset / 32) * 32;
                    length = ((size - offset) / 32) * 32;
                    if (length < 32) {
                        continue;
                    }
                }
                cases.add(new ElementwiseCase(op.name().toLowerCase() + "_" + size, op, size, offset, length, id++));
            }
        }
        return cases.stream().map(Arguments::of);
    }

    private static AbstractTensor convertInput(AbstractTensor input, DType inputType) {
        if (inputType == DType.F32) {
            return new FloatBufferTensor(input);
        }
        if (inputType == DType.BF16) {
            return new BFloat16BufferTensor(input);
        }
        return AbstractTensorUtils.quantize(input, inputType, true);
    }

    private static AbstractTensor convertWeight(AbstractTensor weight, DType weightType) {
        if (weightType == DType.F32) {
            return new FloatBufferTensor(weight);
        }
        if (weightType == DType.BF16) {
            return new BFloat16BufferTensor(weight);
        }
        return AbstractTensorUtils.quantize(weight, weightType, true);
    }

    private static int alignToBlock(int value) {
        int block = 32;
        return ((Math.max(1, value) + block - 1) / block) * block;
    }

    private static FloatBufferTensor deterministicInput(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 17 + col * 31 + seed) % 257 - 128) / 64.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor deterministicWeight(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 43 + col * 19 + seed) % 251 - 125) / 80.0f, row, col);
            }
        }
        return tensor;
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance, String label) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        label + " row=" + row + " col=" + col + " expected=" + expected.get(row, col)
                                + " actual=" + actual.get(row, col));
            }
        }
    }

    private static FloatBufferTensor filled(int rows, int cols, float value) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(value, row, col);
            }
        }
        return tensor;
    }

    private enum Op {
        BATCH_DOT,
        DOT_CHUNK,
        BATCH_CHUNK
    }

    private enum ElementwiseOp {
        ACCUMULATE,
        MACCUMULATE,
        SAXPY,
        EXP,
        MAX,
        SUM,
        ARGMAX,
        SOFTMAX,
        ACTIVATION_MULTIPLY_QUANTIZE
    }

    private record Case(String name, Op op, int batchSize, int columnLength, int aColumnOffset, int bColumnOffset,
            int rRowOffset, int bRowOffset, int rowChunkSize, DType inputType, DType weightType, int seed) {
        private float tolerance() {
            if (inputType == DType.F32 && weightType == DType.F32) {
                return 0.01f;
            }
            if (weightType == DType.BF16 || inputType == DType.BF16) {
                return 0.08f;
            }
            if (inputType == DType.I8 || weightType == DType.I8 || weightType == DType.Q4) {
                return 0.30f;
            }
            return 0.10f;
        }

        @Override
        public String toString() {
            return name + " op=" + op + " batch=" + batchSize + " k=" + columnLength
                    + " aOffset=" + aColumnOffset + " bOffset=" + bColumnOffset
                    + " rOffset=" + rRowOffset + " bRowOffset=" + bRowOffset
                    + " rowChunk=" + rowChunkSize + " input=" + inputType + " weight=" + weightType
                    + " seed=" + seed;
        }

        private int inputColumnOffset() {
            return aColumnOffset;
        }

        private int weightColumnOffset() {
            return op == Op.BATCH_DOT ? bColumnOffset : aColumnOffset;
        }
    }

    private record ElementwiseCase(String name, ElementwiseOp op, int size, int offset, int length, int seed) {
        @Override
        public String toString() {
            return name + " op=" + op + " size=" + size + " offset=" + offset + " length=" + length
                    + " seed=" + seed;
        }
    }
}
