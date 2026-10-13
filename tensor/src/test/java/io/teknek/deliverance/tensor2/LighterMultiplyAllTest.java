package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Shared deterministic fuzz cases for TensorRef scalar multiplication providers. */
class LighterMultiplyAllTest {
    private record Provider(String name, TensorProviderKind kind, TensorOps operations) {
    }

    private record Case(int rows, int columns, float factor, long seed) {
    }

    @ParameterizedTest(name = "multiplyAll {0}")
    @MethodSource("providers")
    void multipliesEveryElementAcrossSharedFuzzCases(Provider provider) {
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(provider.kind(), provider.operations()));
        for (Case testCase : cases()) {
            try (TensorRef tensor = lighter.allocate(DType.F32,
                    TensorShape.of(testCase.rows(), testCase.columns()))) {
                Random random = new Random(testCase.seed());
                float[][] expected = new float[testCase.rows()][testCase.columns()];
                for (int row = 0; row < testCase.rows(); row++) {
                    for (int column = 0; column < testCase.columns(); column++) {
                        float value = -3.0f + random.nextFloat() * 6.0f;
                        tensor.set(value, row, column);
                        expected[row][column] = value * testCase.factor();
                    }
                }

                lighter.multiplyAll(testCase.factor(), tensor);

                for (int row = 0; row < testCase.rows(); row++) {
                    for (int column = 0; column < testCase.columns(); column++) {
                        assertEquals(expected[row][column], tensor.get(row, column), 1.0e-5f,
                                provider.name() + " case=" + testCase + " row=" + row + " column=" + column);
                    }
                }
            }
        }
    }

    @Test
    void multiplyAllRejectsNonFiniteFactor() {
        Lighter lighter = new Lighter();
        try (TensorRef tensor = lighter.allocate(DType.F32, TensorShape.of(2, 3))) {
            assertThrows(IllegalArgumentException.class, () -> lighter.multiplyAll(Float.NaN, tensor));
            assertThrows(IllegalArgumentException.class, () -> lighter.multiplyAll(Float.POSITIVE_INFINITY, tensor));
        }
    }

    @Test
    void multiplyAllRejectsNonMatrixTensor() {
        Lighter lighter = new Lighter();
        try (TensorRef tensor = lighter.allocate(DType.F32, TensorShape.of(2, 3, 4))) {
            assertThrows(IllegalArgumentException.class, () -> lighter.multiplyAll(2.0f, tensor));
        }
    }

    static Stream<Arguments> providers() {
        return Stream.of(
                Arguments.of(new Provider("naive", TensorProviderKind.NAIVE, new NaiveOps())),
                Arguments.of(new Provider("panama", TensorProviderKind.PANAMA, new PanamaOps())));
    }

    private static List<Case> cases() {
        List<Case> cases = new java.util.ArrayList<>(List.of(
                new Case(1, 1, 0.0f, 1L),
                new Case(1, 3, 1.0f, 2L),
                new Case(2, 7, -1.0f, 3L),
                new Case(3, 16, 0.5f, 4L),
                new Case(4, 17, -2.25f, 5L),
                new Case(7, 37, 1.0e-4f, 6L),
                new Case(5, 31, 3.0f, 7L)));
        Random random = new Random(0xE6BADD2L);
        for (int i = 0; i < 40; i++) {
            cases.add(new Case(1 + random.nextInt(8), 1 + random.nextInt(41),
                    -2.0f + random.nextFloat() * 4.0f, random.nextLong()));
        }
        return List.copyOf(cases);
    }
}
