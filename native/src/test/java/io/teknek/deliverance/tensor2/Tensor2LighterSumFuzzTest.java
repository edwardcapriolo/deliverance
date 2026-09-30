package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterSumFuzzTest {
    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void sumMatchesNaiveOracle(Case c) {
        Lighter expected = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
        Lighter actual = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(1, 1));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(1, 1))) {
            fill(expectedInput, c.seed());
            fill(actualInput, c.seed());
            expected.sum(new Sum(expectedInput).into(expectedOutput).row(c.row())
                    .offsetAndLength(c.offset(), c.length()));
            actual.sum(new Sum(actualInput).into(actualOutput).row(c.row())
                    .offsetAndLength(c.offset(), c.length()));
            assertEquals(expectedOutput.get(0, 0), actualOutput.get(0, 0), 1.0e-5f, c.toString());
        }
    }

    static Stream<Arguments> cases() {
        List<Case> cases = new ArrayList<>();
        int index = 0;
        for (int rows : new int[]{1, 2, 5}) {
            for (int columns : new int[]{1, 7, 16, 17, 33, 65, 127}) {
                int offset = index % columns;
                int length = columns - offset;
                cases.add(new Case("case_" + index++, rows, columns, index % rows, offset, length, 700 + index));
            }
        }
        return cases.stream().map(Arguments::of);
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(((row * 19 + column * 11 + seed) % 41 - 20) / 8.0f, row, column);
            }
        }
    }

    private record Case(String name, int rows, int columns, int row, int offset, int length, int seed) {
        @Override
        public String toString() { return name; }
    }
}
