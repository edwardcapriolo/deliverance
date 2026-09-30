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

class Tensor2LighterDotProductF32Bf16FuzzTest {
    @ParameterizedTest(name = "{0}")
    @MethodSource("casesAndCandidates")
    void f32InputBf16WeightsMatchNaive(Case c, Candidate candidate) {
        org.junit.jupiter.api.Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " unavailable");
        Lighter expected = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
        Lighter actual = candidate.lighter();
        try (TensorRef expectedInput = expected.allocate(DType.F32, TensorShape.of(c.inputRows(), c.columns()));
             TensorRef expectedWeights = expected.allocate(DType.BF16, TensorShape.of(c.weightRows(), c.columns()));
             TensorRef expectedOutput = expected.allocate(DType.F32, TensorShape.of(c.inputRows(), c.weightRows()));
             TensorRef actualInput = actual.allocate(DType.F32, TensorShape.of(c.inputRows(), c.columns()));
             TensorRef actualWeights = actual.allocate(DType.BF16, TensorShape.of(c.weightRows(), c.columns()));
             TensorRef actualOutput = actual.allocate(DType.F32, TensorShape.of(c.inputRows(), c.weightRows()))) {
            fill(expectedInput, c.seed());
            fill(expectedWeights, c.seed() + 13);
            fill(actualInput, c.seed());
            fill(actualWeights, c.seed() + 13);
            expected.dotProductRows(expectedOutput, expectedInput, expectedWeights, c.inputOffset(), c.weightOffset(),
                    c.length(), 0, c.weightRows(), 0);
            actual.dotProductRows(actualOutput, actualInput, actualWeights, c.inputOffset(), c.weightOffset(),
                    c.length(), 0, c.weightRows(), 0);
            for (int row = 0; row < c.inputRows(); row++) {
                for (int column = 0; column < c.weightRows(); column++) {
                    assertEquals(expectedOutput.get(row, column), actualOutput.get(row, column), 0.05f,
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
        for (int columns : new int[]{32, 64, 96, 128}) {
            cases.add(new Case("case_" + index++, 1 + index % 3, 3, columns, 0, 0, columns, 900 + index));
            int offset = columns == 32 ? 0 : 32;
            cases.add(new Case("offset_" + index++, 2, 4, columns, offset, offset, columns - offset, 1200 + index));
        }
        return cases.stream();
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(((row * 17 + column * 5 + seed) % 61 - 30) / 16.0f, row, column);
            }
        }
    }

    private record Case(String name, int inputRows, int weightRows, int columns, int inputOffset,
            int weightOffset, int length, int seed) {
        @Override
        public String toString() { return name; }
    }

    private record Candidate(String name, boolean enabled, Lighter lighter) {
        @Override
        public String toString() { return name; }
    }
}
