package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Tensor2LighterReshapeFuzzTest {

    @Test
    void shouldQuantizeForEfficiencyRequiresStrictlySmallerAllocatedStorage() {
        Lighter lighter = naiveOnly();
        try (TensorRef f32 = lighter.allocate(DType.F32, TensorShape.of(1, 64));
             TensorRef bf16 = lighter.allocate(DType.BF16, TensorShape.of(1, 64));
             TensorRef i8 = lighter.allocate(DType.I8, TensorShape.of(1, 64));
             TensorRef q4 = lighter.allocate(DType.Q4, TensorShape.of(1, 64))) {

            assertFalse(lighter.shouldQuantizeForEfficiency(f32, DType.F32));
            assertTrue(lighter.shouldQuantizeForEfficiency(f32, DType.BF16));
            assertTrue(lighter.shouldQuantizeForEfficiency(f32, DType.I8));
            assertTrue(lighter.shouldQuantizeForEfficiency(f32, DType.Q4));

            assertFalse(lighter.shouldQuantizeForEfficiency(bf16, DType.F32));
            assertFalse(lighter.shouldQuantizeForEfficiency(bf16, DType.BF16));
            assertTrue(lighter.shouldQuantizeForEfficiency(bf16, DType.I8));
            assertTrue(lighter.shouldQuantizeForEfficiency(bf16, DType.Q4));

            assertFalse(lighter.shouldQuantizeForEfficiency(i8, DType.F32));
            assertFalse(lighter.shouldQuantizeForEfficiency(i8, DType.BF16));
            assertFalse(lighter.shouldQuantizeForEfficiency(i8, DType.I8));
            assertTrue(lighter.shouldQuantizeForEfficiency(i8, DType.Q4));

            assertFalse(lighter.shouldQuantizeForEfficiency(q4, DType.F32));
            assertFalse(lighter.shouldQuantizeForEfficiency(q4, DType.BF16));
            assertFalse(lighter.shouldQuantizeForEfficiency(q4, DType.I8));
            assertFalse(lighter.shouldQuantizeForEfficiency(q4, DType.Q4));
        }
    }

    @Test
    void shouldQuantizeForEfficiencyReturnsFalseForUnsupportedTargetShape() {
        Lighter lighter = naiveOnly();
        try (TensorRef f32 = lighter.allocate(DType.F32, TensorShape.of(1, 33))) {
            assertFalse(lighter.shouldQuantizeForEfficiency(f32, DType.I8));
            assertFalse(lighter.shouldQuantizeForEfficiency(f32, DType.Q4));
        }
    }

    @ParameterizedTest(name = "{4} {0}->{1} rows={2} cols={3}")
    @MethodSource("reshapeCasesAndCandidates")
    void reshapeAlwaysReturnsNewTensorAndPreservesValuesWithinOutputPrecision(DType inputDType, DType outputDType,
            int rows, int columns, Candidate candidate, int seed) {
        Lighter oracle = naiveOnly();
        Lighter actualLighter = candidate.lighter();
        try (TensorRef oracleDense = oracle.allocate(DType.F32, TensorShape.of(rows, columns));
             TensorRef actualDense = actualLighter.allocate(DType.F32, TensorShape.of(rows, columns))) {
            fill(oracleDense, seed);
            fill(actualDense, seed);
            try (TensorRef oracleInput = inputDType == DType.F32 ? copy(oracle, oracleDense)
                    : oracle.reshape(oracleDense, inputDType);
                 TensorRef expected = oracle.reshape(oracleInput, outputDType);
                 TensorRef actualInput = inputDType == DType.F32 ? copy(actualLighter, actualDense)
                         : actualLighter.reshape(actualDense, inputDType);
                 TensorRef actual = actualLighter.reshape(actualInput, outputDType)) {

                assertNotSame(actualInput, actual);
                assertEquals(outputDType, actual.dType());
                assertEquals(actualInput.shape(), actual.shape());

                for (int row = 0; row < rows; row++) {
                    for (int column = 0; column < columns; column++) {
                        assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column),
                                tolerance(inputDType, outputDType, expected.underlying().get(row, column)),
                                candidate + " " + inputDType + "->" + outputDType + " row=" + row
                                        + " column=" + column);
                    }
                }
            }
        }
    }

    static Stream<Arguments> reshapeCasesAndCandidates() {
        List<Candidate> candidates = List.of(
                new Candidate("NAIVE", Tensor2LighterReshapeFuzzTest::naiveOnly),
                new Candidate("PANAMA", () -> new Lighter(nullMetricRegistry(), Map.of(
                        TensorProviderKind.PANAMA, new PanamaOps()))));
        return List.of(DType.F32, DType.BF16, DType.I8, DType.Q4).stream()
                .flatMap(input -> List.of(DType.F32, DType.BF16, DType.I8, DType.Q4).stream()
                        .flatMap(output -> Stream.of(
                                Arguments.of(input, output, 1, 32,
                                        input.ordinal() * 31 + output.ordinal()),
                                Arguments.of(input, output, 3, 64,
                                        input.ordinal() * 37 + output.ordinal() + 11))
                                .flatMap(args -> candidates.stream().map(candidate -> Arguments.of(args.get()[0],
                                        args.get()[1], args.get()[2], args.get()[3], candidate, args.get()[4])))));
    }

    private static TensorRef copy(Lighter lighter, TensorRef source) {
        TensorRef copy = lighter.allocate(source.dType(), source.shape());
        for (int row = 0; row < source.shape().first(); row++) {
            for (int column = 0; column < source.shape().last(); column++) {
                copy.underlying().set(source.underlying().get(row, column), row, column);
            }
        }
        return copy;
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                float value = ((row * 17 + column * 31 + seed) % 257 - 128) / 96.0f;
                tensor.underlying().set(value, row, column);
            }
        }
    }

    private static float tolerance(DType inputDType, DType outputDType, float expected) {
        if (inputDType == DType.Q4 || outputDType == DType.Q4) {
            return Math.max(0.35f, Math.abs(expected) * 0.30f);
        }
        if (inputDType == DType.I8 || outputDType == DType.I8) {
            return Math.max(0.03f, Math.abs(expected) * 0.03f);
        }
        return switch (outputDType) {
            case BF16 -> Math.max(0.01f, Math.abs(expected) * 0.01f);
            default -> 1.0e-6f;
        };
    }

    private static Lighter naiveOnly() {
        return new Lighter(nullMetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
    }

    private static io.dropwizard.metrics5.MetricRegistry nullMetricRegistry() {
        return new io.dropwizard.metrics5.MetricRegistry();
    }

    private record Candidate(String name, Supplier<Lighter> lighterFactory) {
        Lighter lighter() {
            return lighterFactory.get();
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
