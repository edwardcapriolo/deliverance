package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterAccumulateFuzzTest {
    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void accumulateMatchesNaive(Case c) {
        Lighter expected = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
        Lighter actual = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef expectedA = expected.allocate(c.dtype(), TensorShape.of(c.rows(), c.columns()));
             TensorRef expectedB = expected.allocate(c.dtype(), TensorShape.of(c.sourceRows(), c.columns()));
             TensorRef actualA = actual.allocate(c.dtype(), TensorShape.of(c.rows(), c.columns()));
             TensorRef actualB = actual.allocate(c.dtype(), TensorShape.of(c.sourceRows(), c.columns()))) {
            fill(expectedA, c.seed());
            fill(expectedB, c.seed() + 17);
            fill(actualA, c.seed());
            fill(actualB, c.seed() + 17);
            expected.accumulate(new Accumulate(expectedB).into(expectedA).offsetAndLength(c.offset(), c.length()));
            actual.accumulate(new Accumulate(actualB).into(actualA).offsetAndLength(c.offset(), c.length()));
            for (int row = 0; row < c.rows(); row++) {
                for (int column = 0; column < c.columns(); column++) {
                    assertEquals(expectedA.get(row, column), actualA.get(row, column), 0.0f,
                            c + " row=" + row + " column=" + column);
                }
            }
        }
    }

    static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of(new Case("f32-broadcast", DType.F32, 3, 41, 1, 33, 1, 7)),
                Arguments.of(new Case("bf16-matching", DType.BF16, 3, 41, 4, 29, 3, 19)),
                Arguments.of(new Case("f32-matching", DType.F32, 2, 64, 0, 64, 2, 31)));
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(((row * 13 + column * 7 + seed) % 97 - 48) / 32.0f, row, column);
            }
        }
    }

    private record Case(String name, DType dtype, int rows, int columns, int offset, int length,
            int sourceRows, int seed) {
        @Override
        public String toString() {
            return name;
        }
    }
}
