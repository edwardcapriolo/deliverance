package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.SaxpyCases;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterSaxpyTest {
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
            new io.teknek.deliverance.tensor.operations.NaiveTensorOperations().saxpy(1.75f, legacyX,
                    legacyExpected, c.xOffset(), c.yOffset(), c.length());
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
            new io.teknek.deliverance.tensor.operations.NaiveTensorOperations().saxpy(legacyAlpha, legacyX,
                    legacyExpected, c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(), c.xRowOffset(),
                    c.batchSize());
            actual.saxpy(alpha, x, expected, c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(),
                    c.xRowOffset(), c.batchSize());
            assertClose(expected, legacyExpected);
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
                target.underlying().set(source.get(row, column), row, column);
            }
        }
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
