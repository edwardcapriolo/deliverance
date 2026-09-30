package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterExpFuzzTest {
    @ParameterizedTest(name = "{0} {1}")
    @MethodSource("casesAndCandidates")
    void expMatchesNaiveOracle(Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is unavailable");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.rows(), c.columns()))) {
            fill(expectedInput, c.seed());
            fill(actualInput, c.seed());
            fill(expectedOutput, -17);
            fill(actualOutput, -17);
            expected.exp(new Exp(expectedInput).into(expectedOutput).offsetAndLength(c.offset(), c.length()));
            actual.exp(new Exp(actualInput).into(actualOutput).offsetAndLength(c.offset(), c.length()));
            for (int row = 0; row < c.rows(); row++) {
                for (int column = 0; column < c.columns(); column++) {
                    assertEquals(expectedOutput.get(row, column), actualOutput.get(row, column), 1.0e-5f,
                            c + " row=" + row + " column=" + column);
                }
            }
        }
    }

    static Stream<Arguments> casesAndCandidates() {
        Candidate panama = new Candidate("PANAMA", true,
                new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps())));
        Candidate nativeOps;
        try {
            nativeOps = new Candidate("SIMD", true,
                    new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.SIMD, new NativeOps())));
        } catch (RuntimeException e) {
            nativeOps = new Candidate("SIMD", false, null);
        }
        Candidate[] candidates = {panama, nativeOps};
        return cases().flatMap(c -> Stream.of(candidates).map(candidate -> Arguments.of(c, candidate)));
    }

    static Stream<Case> cases() {
        List<Case> cases = new ArrayList<>();
        int index = 0;
        int[] rows = {1, 2, 3, 5};
        int[] columns = {1, 7, 16, 17, 31, 32, 33, 65, 127};
        for (int row : rows) {
            for (int column : columns) {
                int offset = index % (column + 1);
                int length = column - offset == 0 ? 0 : 1 + (index * 7) % (column - offset);
                if (length > 0) {
                    cases.add(new Case("edge_" + index++, row, column, offset, length, 1000 + index));
                }
            }
        }
        return cases.stream();
    }

    private static Lighter naiveOnly() {
        return new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(((row * 19 + column * 11 + seed) % 41 - 20) / 8.0f, row, column);
            }
        }
    }

    private record Case(String name, int rows, int columns, int offset, int length, int seed) {
        @Override
        public String toString() {
            return name + "[" + rows + "x" + columns + ", offset=" + offset + ", length=" + length + "]";
        }
    }

    private record Candidate(String name, boolean enabled, Lighter lighter) {
        @Override
        public String toString() {
            return name;
        }
    }
}
