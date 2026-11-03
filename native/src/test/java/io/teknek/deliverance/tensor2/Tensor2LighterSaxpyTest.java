package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.operations.SaxpyCases;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterSaxpyTest {
    @Test
    void nativeBatchSaxpyUsesSlicedAlphaAndOutputRows() {
        Assumptions.assumeTrue(NativeOps.isAvailable(), "native tensor2 unavailable");
        Lighter actual = new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                Map.of(TensorProviderKind.SIMD, new NativeOps()));
        try (TensorRef alpha = actual.allocate(DType.F32, TensorShape.of(3, 4));
             TensorRef x = actual.allocate(DType.F32, TensorShape.of(4, 32));
             TensorRef output = actual.allocate(DType.F32, TensorShape.of(3, 32));
             TensorRef alphaRow = alpha.slice(1);
             TensorRef outputRow = output.slice(2)) {
            output.memorySegment().fill((byte) 0);
            alpha.set(1.0f, 1, 0);
            for (int column = 0; column < 16; column++) {
                x.set(column + 1.0f, 0, column);
            }

            actual.saxpy(alphaRow, x, outputRow, 0, 0, 16, 0, 0, 1);

            for (int column = 0; column < 16; column++) {
                assertEquals(column + 1.0f, output.get(2, column), 0.0f, "column=" + column);
                assertEquals(0.0f, output.get(0, column), 0.0f, "row zero column=" + column);
            }
        }
    }

    @ParameterizedTest(name = "scalar {0}")
    @MethodSource("scalarCasesAndCandidates")
    void scalarSaxpyMatchesLegacyOracle(SaxpyCases.ScalarCase c, Candidate candidate) {
        Lighter actual = candidate.lighter();
        try (FloatBufferTensor legacyX = vector(1, c.xOffset() + c.length() + 3);
             FloatBufferTensor legacyExpected = vector(1, c.yOffset() + c.length() + 3);
             TensorRef x = actual.allocate(DType.F32, TensorShape.of(1, c.xOffset() + c.length() + 3));
             TensorRef expected = actual.allocate(DType.F32, TensorShape.of(1, c.yOffset() + c.length() + 3))) {
            fill(x, legacyX);
            fill(expected, legacyExpected);
            referenceSaxpy(1.75f, legacyX, legacyExpected, c.xOffset(), c.yOffset(), c.length());
            actual.saxpy(1.75f, x, expected, c.xOffset(), c.yOffset(), c.length());
            assertClose(expected, legacyExpected);
        }
    }

    @ParameterizedTest(name = "batch {0}")
    @MethodSource("batchCasesAndCandidates")
    void batchSaxpyMatchesLegacyOracle(SaxpyCases.BatchCase c, Candidate candidate) {
        Lighter actual = candidate.lighter();
        try (FloatBufferTensor legacyAlpha = vector(1, c.alphaOffset() + c.batchSize() + 3);
             FloatBufferTensor legacyX = vector(c.xRowOffset() + c.batchSize() + 2,
                     c.xOffset() + c.length() + 3);
             FloatBufferTensor legacyExpected = vector(1, c.yOffset() + c.length() + 3);
             TensorRef alpha = actual.allocate(DType.F32, TensorShape.of(1, c.alphaOffset() + c.batchSize() + 3));
             TensorRef x = actual.allocate(DType.F32, TensorShape.of(c.xRowOffset() + c.batchSize() + 2,
                     c.xOffset() + c.length() + 3));
             TensorRef expected = actual.allocate(DType.F32, TensorShape.of(1, c.yOffset() + c.length() + 3))) {
            fill(alpha, legacyAlpha);
            fill(x, legacyX);
            fill(expected, legacyExpected);
            referenceSaxpy(legacyAlpha, legacyX, legacyExpected, c.xOffset(), c.yOffset(), c.length(),
                    c.alphaOffset(), c.xRowOffset(), c.batchSize());
            actual.saxpy(alpha, x, expected, c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(),
                    c.xRowOffset(), c.batchSize());
            assertClose(expected, legacyExpected);
        }
    }

    @ParameterizedTest(name = "BF16 scalar {0}")
    @MethodSource("scalarCases")
    void bf16ScalarSaxpyMatchesLegacyOracle(SaxpyCases.ScalarCase c) {
        Lighter actual = panamaOnly();
        try (FloatBufferTensor legacyX = vector(1, c.xOffset() + c.length() + 3);
             BFloat16BufferTensor bf16X = new BFloat16BufferTensor(legacyX);
             FloatBufferTensor legacyExpected = vector(1, c.yOffset() + c.length() + 3);
             TensorRef dense = actual.allocate(DType.F32, TensorShape.of(1, legacyX.shape().last()));
             TensorRef expected = actual.allocate(DType.F32, TensorShape.of(1, legacyExpected.shape().last()))) {
            fill(dense, legacyX);
            fill(expected, legacyExpected);
            assertEquals(legacyX.get(0, 0), dense.get(0, 0), 0.0f);
            try (TensorRef x = actual.reshape(dense, DType.BF16)) {
                assertEquals(DType.BF16, x.dType());
                for (int column = 0; column < legacyX.shape().last(); column++) {
                    assertEquals(bf16X.get(0, column), x.get(0, column), 1.0e-3f,
                            "BF16 reshape column=" + column);
                }
                for (int column = 0; column < c.length(); column++) {
                    int xColumn = c.xOffset() + column;
                    int yColumn = c.yOffset() + column;
                    legacyExpected.set(legacyExpected.get(0, yColumn)
                                    + 1.75f * bf16X.get(0, xColumn), 0, yColumn);
                }
                actual.saxpy(1.75f, x, expected, c.xOffset(), c.yOffset(), c.length());
            }
            assertClose(expected, legacyExpected);
        }
    }

    @ParameterizedTest(name = "BF16 batch {0}")
    @MethodSource("batchCases")
    void bf16BatchSaxpyMatchesLegacyOracle(SaxpyCases.BatchCase c) {
        Lighter actual = panamaOnly();
        try (FloatBufferTensor legacyAlpha = vector(1, c.alphaOffset() + c.batchSize() + 3);
             FloatBufferTensor legacyX = vector(c.xRowOffset() + c.batchSize() + 2,
                     c.xOffset() + c.length() + 3);
             BFloat16BufferTensor bf16X = new BFloat16BufferTensor(legacyX);
             FloatBufferTensor legacyExpected = vector(1, c.yOffset() + c.length() + 3);
             TensorRef alpha = actual.allocate(DType.F32, TensorShape.of(1, legacyAlpha.shape().last()));
             TensorRef dense = actual.allocate(DType.F32, TensorShape.of(legacyX.shape().first(), legacyX.shape().last()));
             TensorRef expected = actual.allocate(DType.F32, TensorShape.of(1, legacyExpected.shape().last()))) {
            fill(alpha, legacyAlpha);
            fill(dense, legacyX);
            fill(expected, legacyExpected);
            assertEquals(legacyX.get(0, 0), dense.get(0, 0), 0.0f);
            try (TensorRef x = actual.reshape(dense, DType.BF16)) {
                assertEquals(DType.BF16, x.dType());
                for (int row = 0; row < legacyX.shape().first(); row++) {
                    for (int column = 0; column < legacyX.shape().last(); column++) {
                        assertEquals(bf16X.get(row, column), x.get(row, column), 1.0e-3f,
                                "BF16 reshape row=" + row + " column=" + column);
                    }
                }
                for (int row = 0; row < c.batchSize(); row++) {
                    float factor = legacyAlpha.get(0, c.alphaOffset() + row);
                    for (int column = 0; column < c.length(); column++) {
                        int xColumn = c.xOffset() + column;
                        int yColumn = c.yOffset() + column;
                        legacyExpected.set(legacyExpected.get(0, yColumn)
                                        + factor * bf16X.get(c.xRowOffset() + row, xColumn),
                                0, yColumn);
                    }
                }
                actual.saxpy(alpha, x, expected, c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(),
                        c.xRowOffset(), c.batchSize());
            }
            assertClose(expected, legacyExpected);
        }
    }

    @ParameterizedTest(name = "BF16 target batch {0} {1}")
    @MethodSource("batchCasesAndCandidates")
    void bf16TargetBatchSaxpyMatchesReference(SaxpyCases.BatchCase c, Candidate candidate) {
        Lighter actual = candidate.lighter();
        try (FloatBufferTensor alphaDense = vector(1, c.alphaOffset() + c.batchSize() + 3);
             FloatBufferTensor xDense = vector(c.xRowOffset() + c.batchSize() + 2,
                     c.xOffset() + c.length() + 3);
             FloatBufferTensor yDense = vector(1, c.yOffset() + c.length() + 3);
             BFloat16BufferTensor alphaReference = new BFloat16BufferTensor(alphaDense);
             BFloat16BufferTensor xReference = new BFloat16BufferTensor(xDense);
             BFloat16BufferTensor expected = new BFloat16BufferTensor(yDense);
             TensorRef alphaF32 = actual.allocate(DType.F32, TensorShape.of(1, alphaDense.shape().last()));
             TensorRef xF32 = actual.allocate(DType.F32, TensorShape.of(xDense.shape().first(), xDense.shape().last()));
             TensorRef yF32 = actual.allocate(DType.F32, TensorShape.of(1, yDense.shape().last()))) {
            fill(alphaF32, alphaDense);
            fill(xF32, xDense);
            fill(yF32, yDense);
            for (int row = 0; row < c.batchSize(); row++) {
                float factor = alphaReference.get(0, c.alphaOffset() + row);
                for (int column = 0; column < c.length(); column++) {
                    int xColumn = c.xOffset() + column;
                    int yColumn = c.yOffset() + column;
                    expected.set(expected.get(0, yColumn)
                                    + factor * xReference.get(c.xRowOffset() + row, xColumn), 0, yColumn);
                }
            }
            try (TensorRef alpha = actual.reshape(alphaF32, DType.BF16);
                 TensorRef x = actual.reshape(xF32, DType.BF16);
                 TensorRef y = actual.reshape(yF32, DType.BF16)) {
                actual.saxpy(alpha, x, y, c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(),
                        c.xRowOffset(), c.batchSize());
                for (int column = 0; column < expected.shape().last(); column++) {
                    assertEquals(expected.get(0, column), y.get(0, column), 0.01f, "column=" + column);
                }
            }
        }
    }

    static Stream<Arguments> scalarCases() {
        return SaxpyCases.scalarCases();
    }

    static Stream<Arguments> batchCases() {
        return SaxpyCases.batchCases();
    }

    static Stream<Arguments> scalarCasesAndCandidates() {
        return SaxpyCases.scalarCases().flatMap(args -> candidates()
                .map(candidate -> Arguments.of(args.get()[0], candidate)));
    }

    static Stream<Arguments> batchCasesAndCandidates() {
        return SaxpyCases.batchCases().flatMap(args -> candidates()
                .map(candidate -> Arguments.of(args.get()[0], candidate)));
    }

    private static Stream<Candidate> candidates() {
        return Stream.of(
                new Candidate("NAIVE", true, () -> new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.NAIVE, new NaiveOps()))),
                new Candidate("PANAMA", true, () -> new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.PANAMA, new PanamaOps(), TensorProviderKind.NAIVE, new NaiveOps()))),
                new Candidate("SIMD", NativeOps.isAvailable(), () -> new Lighter(
                        new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.SIMD, new NativeOps(), TensorProviderKind.PANAMA, new PanamaOps(),
                                TensorProviderKind.NAIVE, new NaiveOps()))));
    }

    private static Lighter panamaOnly() {
        return new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
    }

    private record Candidate(String name, boolean enabled, Supplier<Lighter> factory) {
        Lighter lighter() {
            org.junit.jupiter.api.Assumptions.assumeTrue(enabled, name + " unavailable");
            return factory.get();
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private static FloatBufferTensor vector(int rows, int columns) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                tensor.set(((row * 17 + column * 31) % 257 - 128) / 64.0f, row, column);
            }
        }
        return tensor;
    }

    private static void fill(TensorRef target, FloatBufferTensor source) {
        for (int row = 0; row < source.shape().first(); row++) {
            for (int column = 0; column < source.shape().last(); column++) {
                target.set(source.get(row, column), row, column);
            }
        }
    }

    private static void referenceSaxpy(float alpha, FloatBufferTensor x, FloatBufferTensor y, int xOffset,
            int yOffset, int length) {
        for (int column = 0; column < length; column++) {
            y.set(y.get(0, yOffset + column) + alpha * x.get(0, xOffset + column), 0, yOffset + column);
        }
    }

    private static void referenceSaxpy(FloatBufferTensor alpha, FloatBufferTensor x, FloatBufferTensor y,
            int xOffset, int yOffset, int length, int alphaOffset, int xRowOffset, int batchSize) {
        for (int row = 0; row < batchSize; row++) {
            referenceSaxpy(alpha.get(0, alphaOffset + row), xRow(x, xRowOffset + row), y, xOffset, yOffset, length);
        }
    }

    private static FloatBufferTensor xRow(FloatBufferTensor source, int row) {
        FloatBufferTensor result = new FloatBufferTensor(1, (int) source.shape().last());
        for (int column = 0; column < source.shape().last(); column++) {
            result.set(source.get(row, column), 0, column);
        }
        return result;
    }

    private static void assertClose(TensorRef actual, FloatBufferTensor expected) {
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.get(row, column), actual.underlying().get(row, column), 0.001f,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
