package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Stream;
import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterDotProductRowsFuzzTest {
    @org.junit.jupiter.api.Test
    void qwen06GateUpProjectionCharacterization() {
        Assumptions.assumeTrue(NativeOps.isAvailable(), "Native Tensor2 operations are unavailable");
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        int inputColumns = 1024;
        int outputColumns = 3072;
        int chunkSize = 64;
        int repetitions = 32;
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(8));
             FloatBufferTensor denseInput = denseInput(1, inputColumns, 17);
             Q8ByteBufferTensor oldInput = new Q8ByteBufferTensor(denseInput);
             Q4ByteBufferTensor oldGateWeights = q4Weights(outputColumns, inputColumns, 41);
             Q4ByteBufferTensor oldUpWeights = q4Weights(outputColumns, inputColumns, 53);
             FloatBufferTensor oldGateOutput = new FloatBufferTensor(TensorShape.of(1, outputColumns));
             FloatBufferTensor oldUpOutput = new FloatBufferTensor(TensorShape.of(1, outputColumns));
             TensorRef input = TensorRef.borrowed(oldInput);
             TensorRef gateWeights = TensorRef.borrowed(oldGateWeights);
             TensorRef upWeights = TensorRef.borrowed(oldUpWeights);
             TensorRef newGateOutput = new Lighter(metrics, Map.of(TensorProviderKind.SIMD, new NativeOps()))
                     .allocate(DType.F32, TensorShape.of(1, outputColumns));
             TensorRef newUpOutput = new Lighter(metrics, Map.of(TensorProviderKind.SIMD, new NativeOps()))
                     .allocate(DType.F32, TensorShape.of(1, outputColumns))) {
            NativeSimdTensorOperations oldOps = new NativeSimdTensorOperations(
                    new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool), chunkSize);
            Lighter newOps = new Lighter(metrics, Map.of(TensorProviderKind.SIMD, new NativeOps()));
            CompositeOps newComposite = new CompositeOps(newOps);
            AbstractTensor[] oldOutputs = {oldGateOutput, oldUpOutput};
            AbstractTensor[] oldWeights = {oldGateWeights, oldUpWeights};

            for (int i = 0; i < 8; i++) {
                runLegacyGateUp(oldOps, oldOutputs, oldInput, oldWeights, outputColumns, inputColumns, chunkSize);
                runTensorRefGateUp(newComposite, newGateOutput, newUpOutput, input, gateWeights, upWeights,
                        outputColumns, inputColumns, chunkSize);
            }
            long oldStart = System.nanoTime();
            for (int i = 0; i < repetitions; i++) {
                runLegacyGateUp(oldOps, oldOutputs, oldInput, oldWeights, outputColumns, inputColumns, chunkSize);
            }
            double oldMs = (System.nanoTime() - oldStart) / 1_000_000.0 / repetitions;
            long newStart = System.nanoTime();
            for (int i = 0; i < repetitions; i++) {
                runTensorRefGateUp(newComposite, newGateOutput, newUpOutput, input, gateWeights, upWeights,
                        outputColumns, inputColumns, chunkSize);
            }
            double newMs = (System.nanoTime() - newStart) / 1_000_000.0 / repetitions;
            System.out.printf(java.util.Locale.ROOT,
                    "[tensor2-gate-up-characterization] shape=1x%d->%d chunks=%d repetitions=%d "
                            + "old_mean_ms=%.3f new_mean_ms=%.3f delta_ms=%.3f speedup=%.3fx%n",
                    inputColumns, outputColumns, outputColumns / chunkSize, repetitions, oldMs, newMs,
                    oldMs - newMs, oldMs / newMs);
        }
    }

    private static void runLegacyGateUp(NativeSimdTensorOperations ops, AbstractTensor[] outputs,
            AbstractTensor input, AbstractTensor[] weights, int outputColumns, int inputColumns, int chunkSize) {
        for (int chunkStart = 0; chunkStart < outputColumns; chunkStart += chunkSize) {
            ops.dotProductBatchChunk(outputs, input, weights, 0, inputColumns, chunkStart, chunkSize);
        }
    }

    private static void runTensorRefGateUp(CompositeOps ops, TensorRef gateOutput, TensorRef upOutput, TensorRef input,
            TensorRef gateWeights, TensorRef upWeights, int outputColumns, int inputColumns, int chunkSize) {
        for (int chunkStart = 0; chunkStart < outputColumns; chunkStart += chunkSize) {
            ops.dotProductBatchChunk(new DotProductBatchChunk()
                    .results(gateOutput, upOutput)
                    .input(input)
                    .weights(gateWeights, upWeights)
                    .inputColumnStart(0)
                    .weightColumnStart(0)
                    .columnLength(inputColumns)
                    .weightRowStart(chunkStart)
                    .weightRowCount(chunkSize)
                    .outputColumnStart(chunkStart));
        }
    }

    @ParameterizedTest(name = "{0} {1}")
    @MethodSource("f32Cases")
    void f32DotProductRowsMatchesNaive(Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef expectedWeights = expected.allocate(DType.F32, TensorShape.of(c.weightRows(), c.weightColumns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef actualWeights = actual.allocate(DType.F32, TensorShape.of(c.weightRows(), c.weightColumns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()))) {
            fill(expectedInput, c.seed);
            fill(expectedWeights, c.seed + 17);
            fill(actualInput, c.seed);
            fill(actualWeights, c.seed + 17);

            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights, c.inputStart, c.inputLength,
                    c.weightRowStart, c.weightRowCount, c.outputColumnStart);
            actual.dotProductRows(actualOutput, actualInput, actualWeights, c.inputStart, c.inputLength,
                    c.weightRowStart, c.weightRowCount, c.outputColumnStart);
            assertEqual(c, expectedOutput, actualOutput, 0.01f);
        }
    }

    @ParameterizedTest(name = "f32 GQA offsets {0}")
    @MethodSource("candidates")
    void f32DotProductRowsSupportsDifferentOperandOffsets(Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(3, 48));
             TensorRef expectedWeights = expected.allocate(DType.F32, TensorShape.of(6, 48));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(3, 7));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(3, 48));
             TensorRef actualWeights = actual.allocate(DType.F32, TensorShape.of(6, 48));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(3, 7))) {
            fill(expectedInput, 71);
            fill(expectedWeights, 89);
            fill(actualInput, 71);
            fill(actualWeights, 89);
            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights,
                    16, 0, 16, 0, 5, 0);
            actual.dotProductRows(actualOutput, actualInput, actualWeights,
                    16, 0, 16, 0, 5, 0);
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 5; column++) {
                    assertEquals(expectedOutput.get(row, column), actualOutput.get(row, column), 0.01f,
                            candidate.name() + " row=" + row + " column=" + column);
                }
            }
        }
    }

    @ParameterizedTest(name = "strict f32 GQA offsets {0}")
    @MethodSource("strictProductionCandidates")
    void f32DotProductRowsSupportsDifferentOperandOffsetsWithoutFallback(ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        MetricRegistry metrics = new MetricRegistry();
        Lighter actual = new Lighter(metrics, Map.of(candidate.kind(), candidate.operations().get()));
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(3, 48));
             TensorRef expectedWeights = expected.allocate(DType.F32, TensorShape.of(6, 48));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(3, 7));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(3, 48));
             TensorRef actualWeights = actual.allocate(DType.F32, TensorShape.of(6, 48));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(3, 7))) {
            fill(expectedInput, 71);
            fill(expectedWeights, 89);
            fill(actualInput, 71);
            fill(actualWeights, 89);
            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights,
                    16, 0, 16, 0, 5, 0);
            actual.dotProductRows(actualOutput, actualInput, actualWeights,
                    16, 0, 16, 0, 5, 0);
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 5; column++) {
                    assertEquals(expectedOutput.get(row, column), actualOutput.get(row, column), 0.01f,
                            candidate.name() + " row=" + row + " column=" + column);
                }
            }
            assertEquals(1, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, candidate.kind().name()))).getCount());
            assertEquals(0, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.NAIVE.name()))).getCount());
        }
    }

    @ParameterizedTest(name = "strict f32-q4 slice {0}")
    @MethodSource("strictProductionCandidates")
    void f32Q4DotProductRowsUsesLogicalSliceBaseWithoutFallback(ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        MetricRegistry metrics = new MetricRegistry();
        Lighter actual = new Lighter(metrics, Map.of(candidate.kind(), candidate.operations().get()));
        try (TensorRef expectedInputParent = expected.allocate(DType.F32, TensorShape.of(4, 64));
             TensorRef expectedOutputParent = expected.allocate(DType.F32, TensorShape.of(3, 5));
             TensorRef actualInputParent = actual.allocate(DType.F32, TensorShape.of(4, 64));
             TensorRef actualOutputParent = actual.allocate(DType.F32, TensorShape.of(3, 5));
             Q4ByteBufferTensor expectedWeightTensor = q4Weights(4, 64, 123);
             Q4ByteBufferTensor actualWeightTensor = q4Weights(4, 64, 123)) {
            fill(expectedInputParent, 417);
            fill(actualInputParent, 417);
            TensorRef expectedWeightParent = TensorRef.borrowed(expectedWeightTensor);
            TensorRef actualWeightParent = TensorRef.borrowed(actualWeightTensor);
            try (TensorRef expectedInput = expectedInputParent.slice(2);
                 TensorRef expectedWeights = expectedWeightParent.slice(1);
                 TensorRef expectedOutput = expectedOutputParent.slice(1);
                 TensorRef actualInput = actualInputParent.slice(2);
                 TensorRef actualWeights = actualWeightParent.slice(1);
                 TensorRef actualOutput = actualOutputParent.slice(1)) {
                expected.dotProductRows(expectedOutput, expectedInput, expectedWeights, 0, 64, 0, 1, 2);
                actual.dotProductRows(actualOutput, actualInput, actualWeights, 0, 64, 0, 1, 2);
                assertEquals(expectedOutputParent.get(1, 2), actualOutputParent.get(1, 2), 0.001f,
                        candidate.name());
                assertEquals(0.0f, actualOutputParent.get(0, 2), 0.0f, "slice output must not write row zero");
            }
            assertEquals(1, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, candidate.kind().name()))).getCount());
            assertEquals(0, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.NAIVE.name()))).getCount());
        }
    }

    @ParameterizedTest(name = "strict reshape slice {0}")
    @MethodSource("strictPanamaCandidates")
    void sameDTypeReshapeUsesLogicalSliceBaseWithoutFallback(ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter actual = new Lighter(new MetricRegistry(), Map.of(candidate.kind(), candidate.operations().get()));
        try (TensorRef parent = actual.allocate(DType.F32, TensorShape.of(3, 64));
             TensorRef slice = parent.slice(2)) {
            fill(parent, 911);
            try (TensorRef reshaped = actual.reshape(slice, DType.F32)) {
                for (int column = 0; column < 64; column++) {
                    assertEquals(parent.get(2, column), reshaped.get(0, column), 0.0f,
                            candidate.name() + " column=" + column);
                }
            }
        }
    }

    @ParameterizedTest(name = "q8 {0} {1}")
    @MethodSource("q8Cases")
    void q8DotProductRowsMatchesNaive(Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             Q8ByteBufferTensor expectedDense = q8Weights(c.weightRows(), c.weightColumns(), c.seed() + 17);
             Q8ByteBufferTensor actualDense = q8Weights(c.weightRows(), c.weightColumns(), c.seed() + 17)) {
            TensorRef expectedWeights = TensorRef.borrowed(expectedDense);
            TensorRef actualWeights = TensorRef.borrowed(actualDense);
            fill(expectedInput, c.seed);
            fill(actualInput, c.seed);

            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights, c.inputStart, c.inputLength,
                    c.weightRowStart, c.weightRowCount, c.outputColumnStart);
            actual.dotProductRows(actualOutput, actualInput, actualWeights, c.inputStart, c.inputLength,
                    c.weightRowStart, c.weightRowCount, c.outputColumnStart);
            assertEqual(c, expectedOutput, actualOutput, 0.03f);
        }
    }

    @ParameterizedTest(name = "q4 {0} {1}")
    @MethodSource("q4Cases")
    void q4DotProductRowsMatchesNaive(Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             Q4ByteBufferTensor expectedDense = q4Weights(c.weightRows(), c.weightColumns(), c.seed() + 17);
             Q4ByteBufferTensor actualDense = q4Weights(c.weightRows(), c.weightColumns(), c.seed() + 17)) {
            TensorRef expectedWeights = TensorRef.borrowed(expectedDense);
            TensorRef actualWeights = TensorRef.borrowed(actualDense);
            fill(expectedInput, c.seed());
            fill(actualInput, c.seed());

            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights, c.inputStart(), c.inputLength(),
                    c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());
            actual.dotProductRows(actualOutput, actualInput, actualWeights, c.inputStart(), c.inputLength(),
                    c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());
             assertEqual(c, expectedOutput, actualOutput, 0.001f);
        }
    }

    @ParameterizedTest(name = "tensor2 q4 projection {0}")
    @MethodSource("candidates")
    void tensor2Q4ProjectionShapeMatchesNaive(Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef input = actual.allocate(DType.F32, TensorShape.of(4, 2048));
             TensorRef denseWeights = actual.allocate(DType.F32, TensorShape.of(256, 2048));
             TensorRef weights = actual.reshape(denseWeights, DType.Q4);
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(4, 256));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(4, 256))) {
            fill(input, 417);
            fill(denseWeights, 431);
            expected.dotProductRows(expectedOutput, input, weights, 0, 2048, 0, 256, 0);
            actual.dotProductRows(actualOutput, input, weights, 0, 2048, 0, 256, 0);
            for (int row = 0; row < 4; row++) {
                for (int column = 0; column < 256; column++) {
                    assertEquals(expectedOutput.get(row, column), actualOutput.get(row, column), 0.001f,
                            candidate.name() + " row=" + row + " column=" + column);
                }
            }
        }
    }

    @ParameterizedTest(name = "bf16-q4 {0} {1}")
    @MethodSource("bf16Q4Cases")
    void bf16Q4DotProductRowsMatchesNaive(Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.BF16, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             TensorRef actualInput = actual.allocate(DType.BF16, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             Q4ByteBufferTensor expectedDense = q4Weights(c.weightRows(), c.weightColumns(), c.seed() + 17);
             Q4ByteBufferTensor actualDense = q4Weights(c.weightRows(), c.weightColumns(), c.seed() + 17)) {
            TensorRef expectedWeights = TensorRef.borrowed(expectedDense);
            TensorRef actualWeights = TensorRef.borrowed(actualDense);
            fill(expectedInput, c.seed());
            fill(actualInput, c.seed());
            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights, c.inputStart(), c.inputLength(),
                    c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());
            actual.dotProductRows(actualOutput, actualInput, actualWeights, c.inputStart(), c.inputLength(),
                    c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());
            assertEqual(c, expectedOutput, actualOutput, 0.08f);
        }
    }

    @ParameterizedTest(name = "i8-q4 {0} {1}")
    @MethodSource("i8Q4Cases")
    void i8Q4DotProductRowsUsesProductionProvider(I8Q4Case c, ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        MetricRegistry metrics = new MetricRegistry();
        Lighter actual = new Lighter(metrics, Map.of(candidate.kind(), candidate.operations().get()));
        try (TensorRef denseInput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef input = quantizedInput(denseInput, c.seed());
             Q4ByteBufferTensor denseWeights = q4Weights(c.weightRows(), c.weightColumns(), c.seed() + 17);
             TensorRef weights = TensorRef.borrowed(denseWeights);
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()))) {
            expected.dotProductRows(expectedOutput, input, weights, c.inputColumnStart(), c.weightColumnStart(),
                    c.columnLength(), c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());
            actual.dotProductRows(actualOutput, input, weights, c.inputColumnStart(), c.weightColumnStart(),
                    c.columnLength(), c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());

            assertEqual(c, expectedOutput, actualOutput, 0.03f);
            assertEquals(1, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, candidate.kind().name()))).getCount());
            assertEquals(0, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.NAIVE.name()))).getCount());
        }
    }

    @ParameterizedTest(name = "Qwen3-0.6B query projection {0}")
    @MethodSource("strictProductionCandidates")
    void qwen06QueryProjectionShapeMatchesNaive(ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        MetricRegistry metrics = new MetricRegistry();
        Lighter actual = new Lighter(metrics, Map.of(candidate.kind(), candidate.operations().get()));
        int embeddingLength = 1024;
        int attentionLength = 2048;
        int chunkSize = 64;
        Lighter quantizer = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef denseInput = actual.allocate(DType.F32, TensorShape.of(1, embeddingLength));
             TensorRef input = quantizer.reshape(denseInput, DType.I8);
             Q4ByteBufferTensor denseWeights = q4Weights(attentionLength, embeddingLength, 41);
             TensorRef weights = TensorRef.borrowed(denseWeights);
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(1, attentionLength));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(1, attentionLength))) {
            fill(denseInput, 17);
            for (int chunkStart = 0; chunkStart < attentionLength; chunkStart += chunkSize) {
                expected.dotProductRows(expectedOutput, input, weights, 0, embeddingLength, chunkStart, chunkSize,
                        chunkStart);
                actual.dotProductRows(actualOutput, input, weights, 0, embeddingLength, chunkStart, chunkSize,
                        chunkStart);
            }
            for (int column = 0; column < attentionLength; column++) {
                assertEquals(expectedOutput.get(0, column), actualOutput.get(0, column), 0.03f,
                        candidate.name() + " column=" + column);
            }
            assertEquals(attentionLength / chunkSize,
                    metrics.meter(new MetricName("tensor2.dot_product_rows",
                            Map.of(Lighter.TENSOR_OP_KEY, candidate.kind().name()))).getCount());
            assertEquals(0, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.NAIVE.name()))).getCount());
        }
    }

    @ParameterizedTest(name = "Qwen3-0.6B MLP projection shapes {0}")
    @MethodSource("strictProductionCandidates")
    void qwen06MlpProjectionShapesMatchNaive(ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = new Lighter(new MetricRegistry(), Map.of(candidate.kind(), candidate.operations().get()));
        Lighter quantizer = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        assertProjectionShape(expected, actual, quantizer, 1, 1024, 3072, 53, candidate.name() + " gate/up");
        assertProjectionShape(expected, actual, quantizer, 1, 3072, 1024, 71, candidate.name() + " down");
    }

    private static void assertProjectionShape(Lighter expected, Lighter actual, Lighter quantizer, int rows, int inputColumns,
            int outputColumns, int seed, String label) {
        try (TensorRef denseInput = actual.allocate(DType.F32, TensorShape.of(rows, inputColumns));
             TensorRef input = quantizer.reshape(denseInput, DType.I8);
             Q4ByteBufferTensor denseWeights = q4Weights(outputColumns, inputColumns, seed + 11);
             TensorRef weights = TensorRef.borrowed(denseWeights);
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(rows, outputColumns));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(rows, outputColumns))) {
            fill(denseInput, seed);
            expected.dotProductRows(expectedOutput, input, weights, 0, inputColumns, 0,
                    outputColumns, 0);
            actual.dotProductRows(actualOutput, input, weights, 0, inputColumns, 0, outputColumns, 0);
            for (int column = 0; column < outputColumns; column++) {
                assertEquals(expectedOutput.get(0, column), actualOutput.get(0, column), 0.01f,
                        label + " column=" + column);
            }
        }
    }

    @ParameterizedTest(name = "strict i8-q4 batch-shape equivalence {0}")
    @MethodSource("strictProductionCandidates")
    void i8Q4DotProductRowsMatchesAcrossBatchShapes(ProviderCandidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        MetricRegistry metrics = new MetricRegistry();
        Lighter actual = new Lighter(metrics, Map.of(candidate.kind(), candidate.operations().get()));
        Lighter quantizer = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        int rows = 5;
        int columns = 32;
        try (TensorRef denseBatch = quantizer.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef denseSingle = quantizer.allocate(DType.F32, TensorShape.of(1, columns));
             TensorRef batchInput = quantizer.reshape(denseBatch, DType.I8);
             TensorRef singleInput = quantizer.reshape(denseSingle, DType.I8);
             Q4ByteBufferTensor denseWeights = q4Weights(columns, columns, 12_341);
             TensorRef weights = TensorRef.borrowed(denseWeights);
             TensorRef expectedBatch = expected.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef expectedSingle = expected.allocate(DType.F32, TensorShape.of(1, columns));
             TensorRef actualBatch = actual.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef actualSingle = actual.allocate(DType.F32, TensorShape.of(1, columns))) {
            fill(denseBatch, 12_347);
            for (int column = 0; column < columns; column++) {
                denseSingle.set(denseBatch.get(rows - 1, column), 0, column);
            }
            for (int chunkStart = 0; chunkStart < columns; chunkStart += 8) {
                expected.dotProductRows(expectedBatch, batchInput, weights, 0, columns, chunkStart, 8, chunkStart);
                expected.dotProductRows(expectedSingle, singleInput, weights, 0, columns, chunkStart, 8,
                        chunkStart);
                actual.dotProductRows(actualBatch, batchInput, weights, 0, columns, chunkStart, 8, chunkStart);
                actual.dotProductRows(actualSingle, singleInput, weights, 0, columns, chunkStart, 8,
                        chunkStart);
            }

            for (int column = 0; column < columns; column++) {
                assertEquals(expectedBatch.get(rows - 1, column), actualBatch.get(rows - 1, column), 0.03f,
                        candidate.name() + " batch oracle column=" + column);
                assertEquals(expectedSingle.get(0, column), actualSingle.get(0, column), 0.03f,
                        candidate.name() + " single oracle column=" + column);
                assertEquals(actualBatch.get(rows - 1, column), actualSingle.get(0, column), 0.03f,
                        candidate.name() + " batch versus single column=" + column);
            }
            assertEquals(8, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, candidate.kind().name()))).getCount());
            assertEquals(0, metrics.meter(new MetricName("tensor2.dot_product_rows",
                    Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.NAIVE.name()))).getCount());
        }
    }

    @ParameterizedTest(name = "legacy i8-q4 projection {0}")
    @MethodSource("legacyI8Q4ProjectionCases")
    void i8Q4ProjectionMatchesLegacyTensorOperations(I8Q4Case c) {
        Lighter tensorRefOps = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (FloatBufferTensor denseInput = denseInput(c.rows(), c.inputColumns(), c.seed());
             FloatBufferTensor denseWeights = denseWeights(c.weightRows(), c.weightColumns(), c.seed() + 17);
             Q8ByteBufferTensor legacyInput = new Q8ByteBufferTensor(denseInput);
             Q4ByteBufferTensor legacyWeights = new Q4ByteBufferTensor(denseWeights);
             FloatBufferTensor legacyOutput = new FloatBufferTensor(TensorShape.of(c.rows(), c.outputColumns()));
             TensorRef inputF32 = tensorRefOps.allocate(DType.F32, TensorShape.of(c.rows(), c.inputColumns()));
             TensorRef weightF32 = tensorRefOps.allocate(DType.F32, TensorShape.of(c.weightRows(), c.weightColumns()));
             TensorRef tensorRefOutput = tensorRefOps.allocate(DType.F32, TensorShape.of(c.rows(), c.outputColumns()))) {
            copy(denseInput, inputF32);
            copy(denseWeights, weightF32);
            try (TensorRef input = tensorRefOps.reshape(inputF32, DType.I8);
                 TensorRef weights = tensorRefOps.reshape(weightF32, DType.Q4)) {
                new NaiveTensorOperations().dotProductChunk(legacyOutput, legacyInput, legacyWeights,
                        c.inputColumnStart(), c.columnLength(), c.weightRowStart(), c.weightRowCount());
                tensorRefOps.dotProductRows(tensorRefOutput, input, weights, c.inputColumnStart(), c.weightColumnStart(),
                        c.columnLength(), c.weightRowStart(), c.weightRowCount(), c.outputColumnStart());
                for (int row = 0; row < c.rows(); row++) {
                    for (int column = 0; column < c.outputColumns(); column++) {
                        assertEquals(legacyOutput.get(row, column), tensorRefOutput.get(row, column), 0.03f,
                                c + " row=" + row + " column=" + column);
                    }
                }
            }
        }
    }

    static Stream<Arguments> f32Cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] rows = {1, 2, 3, 5, 8};
        int[] inputLengths = {1, 3, 7, 16, 31, 32, 33, 63, 64, 95, 128, 129};
        for (int rowCount : rows) {
            for (int inputLength : inputLengths) {
                cases.add(new Case("f32_" + id, rowCount, 4, 4 + inputLength, 8 + inputLength,
                        5 + (id % 7), 2 + (id % 5), 1 + (id % 3), inputLength, id++));
            }
        }
        Random random = new Random(0xbadc0ffeeL);
        for (int i = 0; i < 128; i++) {
            int inputStart = random.nextInt(8);
            int inputLength = 1 + random.nextInt(160);
            int weightRowStart = random.nextInt(8);
            int weightRowCount = 1 + random.nextInt(96);
            cases.add(new Case("random_" + i, 1 + random.nextInt(8), inputStart, inputStart + inputLength,
                    inputStart + inputLength, weightRowStart, weightRowCount, random.nextInt(8), inputLength,
                    random.nextInt()));
        }
        return candidates().flatMap(candidate -> cases.stream()
                .map(c -> Arguments.of(c, candidate)));
    }

    static Stream<Arguments> q8Cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] inputLengths = {32, 64, 96, 128, 160, 256};
        for (int inputLength : inputLengths) {
            cases.add(new Case("q8_" + id, 1 + id % 5, 32, 32 + inputLength, 32 + inputLength,
                    id % 5, 1 + id % 31, id % 7, inputLength, id++));
        }
        return candidates().flatMap(candidate -> cases.stream()
                .map(c -> Arguments.of(c, candidate)));
    }

    static Stream<Arguments> q4Cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] inputLengths = {32, 64, 96, 128, 160, 256};
        for (int inputLength : inputLengths) {
            cases.add(new Case("q4_" + id, 1 + id % 5, 32, 32 + inputLength, 32 + inputLength,
                    id % 5, 1 + id % 31, id % 7, inputLength, id++));
        }
        return candidates().flatMap(candidate -> cases.stream()
                .map(c -> Arguments.of(c, candidate)));
    }

    static Stream<Arguments> bf16Q4Cases() {
        List<Case> cases = new ArrayList<>();
        int id = 0;
        int[] inputLengths = {32, 64, 128, 256};
        int[] offsets = {0, 32};
        for (int offset : offsets) {
            for (int inputLength : inputLengths) {
                cases.add(new Case("bf16_q4_" + id, 1 + id % 5, offset, offset + inputLength,
                        offset + inputLength, id % 5, 1 + id % 17, id % 5, inputLength, id++));
            }
        }
        return candidates().flatMap(candidate -> cases.stream()
                .map(c -> Arguments.of(c, candidate)));
    }

    static Stream<Arguments> i8Q4Cases() {
        List<I8Q4Case> cases = new ArrayList<>();
        int id = 0;
        int[] lengths = {32, 64, 96, 128, 160, 256, 768};
        int[] rowCounts = {1, 3, 17, 33};
        for (int length : lengths) {
            int inputStart = (id % 2) * 32;
            int weightStart = ((id + 1) % 2) * 32;
            int weightRowCount = rowCounts[id % rowCounts.length];
            cases.add(new I8Q4Case("i8_q4_" + id, 1 + id % 5, inputStart, weightStart, length,
                    inputStart + length, weightStart + length, id % 5, weightRowCount, id % 7, id++));
        }
        return i8Q4Candidates().flatMap(candidate -> cases.stream().map(c -> Arguments.of(c, candidate)));
    }

    static Stream<Arguments> legacyI8Q4ProjectionCases() {
        List<I8Q4Case> cases = new ArrayList<>();
        int id = 0;
        int[] lengths = {32, 64, 96, 128, 256, 768, 2048};
        int[] rowCounts = {1, 3, 17, 33, 128};
        int[] rows = {1, 4, 16};
        for (int rowCount : rows) {
            for (int length : lengths) {
                int inputStart = (id % 3) * 32;
                int weightStart = inputStart;
                int weightRowCount = rowCounts[id % rowCounts.length];
                cases.add(new I8Q4Case("legacy_i8_q4_" + id, rowCount, inputStart, weightStart, length,
                        inputStart + length, weightStart + length, 0, weightRowCount, 0, id++));
            }
        }
        return cases.stream().map(Arguments::of);
    }

    private static Stream<Candidate> candidates() {
        return Stream.of(
                new Candidate("PANAMA", true, () -> new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.PANAMA, new PanamaOps(), TensorProviderKind.NAIVE, new NaiveOps()))),
                new Candidate("SIMD", NativeOps.isAvailable(), () -> new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.SIMD, new NativeOps(), TensorProviderKind.PANAMA, new PanamaOps(),
                                TensorProviderKind.NAIVE, new NaiveOps()))));
    }

    private static Stream<ProviderCandidate> i8Q4Candidates() {
        return Stream.of(
                new ProviderCandidate("PANAMA", true, TensorProviderKind.PANAMA, PanamaOps::new),
                new ProviderCandidate("SIMD", NativeOps.isAvailable(), TensorProviderKind.SIMD, NativeOps::new));
    }

    private static Stream<ProviderCandidate> strictProductionCandidates() {
        return Stream.of(
                new ProviderCandidate("PANAMA", true, TensorProviderKind.PANAMA, PanamaOps::new),
                new ProviderCandidate("SIMD", NativeOps.isAvailable(), TensorProviderKind.SIMD, NativeOps::new));
    }

    private static Stream<ProviderCandidate> strictPanamaCandidates() {
        return Stream.of(new ProviderCandidate("PANAMA", true, TensorProviderKind.PANAMA, PanamaOps::new));
    }

    private static Lighter naiveOnly() {
        return new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
    }

    private static void assertEqual(Case c, TensorRef expected, TensorRef actual, float tolerance) {
        for (int row = 0; row < c.rows(); row++) {
            for (int column = 0; column < c.outputColumns(); column++) {
                assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column), tolerance,
                        c + " row=" + row + " column=" + column);
            }
        }
    }

    private static void assertEqual(I8Q4Case c, TensorRef expected, TensorRef actual, float tolerance) {
        for (int row = 0; row < c.rows(); row++) {
            for (int column = 0; column < c.outputColumns(); column++) {
                assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column), tolerance,
                        c + " row=" + row + " column=" + column);
            }
        }
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(((row * 17 + column * 31 + seed) % 257 - 128) / 64.0f, row, column);
            }
        }
    }

    private static FloatBufferTensor denseInput(int rows, int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 17 + column * 31 + seed) % 257 - 128) / 64.0f, row, column);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor denseWeights(int rows, int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 13 + column * 19 + seed) % 251 - 125) / 96.0f, row, column);
            }
        }
        return tensor;
    }

    private static void copy(FloatBufferTensor source, TensorRef target) {
        for (int row = 0; row < source.shape().first(); row++) {
            for (int column = 0; column < source.shape().last(); column++) {
                target.set(source.get(row, column), row, column);
            }
        }
    }

    private static TensorRef quantizedInput(TensorRef dense, int seed) {
        fill(dense, seed);
        Lighter quantizer = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        return quantizer.reshape(dense, DType.I8);
    }

    private static Q8ByteBufferTensor q8Weights(int rows, int columns, int seed) {
        FloatBufferTensor dense = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                dense.set(((row * 13 + column * 19 + seed) % 251 - 125) / 96.0f, row, column);
            }
        }
        return new Q8ByteBufferTensor(dense);
    }

    private static Q4ByteBufferTensor q4Weights(int rows, int columns, int seed) {
        FloatBufferTensor dense = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                dense.set(((row * 13 + column * 19 + seed) % 251 - 125) / 96.0f, row, column);
            }
        }
        return new Q4ByteBufferTensor(dense);
    }

    private record Case(String name, int rows, int inputStart, int inputColumns, int weightColumns,
            int weightRowStart, int weightRowCount, int outputColumnStart, int inputLength, int seed) {
        int weightRows() {
            return weightRowStart + weightRowCount;
        }

        int outputColumns() {
            return outputColumnStart + weightRowCount + 2;
        }

        @Override
        public String toString() {
            return name + "[rows=" + rows + ", inputStart=" + inputStart() + ", inputLength=" + inputLength
                    + ", weightRowStart=" + weightRowStart + ", weightRowCount=" + weightRowCount
                    + ", outputColumnStart=" + outputColumnStart + "]";
        }
    }

    private record Candidate(String name, boolean enabled, Supplier<Lighter> lighterFactory) {
        Lighter lighter() {
            return lighterFactory.get();
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private record ProviderCandidate(String name, boolean enabled, TensorProviderKind kind,
            Supplier<TensorOps> operations) {
        @Override
        public String toString() {
            return name;
        }
    }

    private record I8Q4Case(String name, int rows, int inputColumnStart, int weightColumnStart, int columnLength,
            int inputColumns, int weightColumns, int weightRowStart, int weightRowCount, int outputColumnStart,
            int seed) {
        int weightRows() {
            return weightRowStart + weightRowCount;
        }

        int outputColumns() {
            return outputColumnStart + weightRowCount + 2;
        }

        @Override
        public String toString() {
            return name + "[rows=" + rows + ", inputStart=" + inputColumnStart
                    + ", weightStart=" + weightColumnStart + ", length=" + columnLength
                    + ", weightRows=" + weightRowCount + "]";
        }
    }
}
