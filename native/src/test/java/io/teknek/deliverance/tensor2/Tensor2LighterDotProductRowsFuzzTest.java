package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
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

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterDotProductRowsFuzzTest {
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
            assertEqual(c, expectedOutput, actualOutput, 0.08f);
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

    private static Stream<Candidate> candidates() {
        return Stream.of(
                new Candidate("PANAMA", true, () -> new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.PANAMA, new PanamaOps(), TensorProviderKind.NAIVE, new NaiveOps()))),
                new Candidate("SIMD", NativeOps.isAvailable(), () -> new Lighter(new io.dropwizard.metrics5.MetricRegistry(),
                        Map.of(TensorProviderKind.SIMD, new NativeOps(), TensorProviderKind.PANAMA, new PanamaOps(),
                                TensorProviderKind.NAIVE, new NaiveOps()))));
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

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(((row * 17 + column * 31 + seed) % 257 - 128) / 64.0f, row, column);
            }
        }
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
}
