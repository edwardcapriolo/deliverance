package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterBatchDotProductFuzzTest {
    @ParameterizedTest(name = "{0} {1}")
    @MethodSource("casesAndCandidates")
    void batchDotProductMatchesNaiveLighter(BatchDotProductFuzzCases.Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is not implemented yet");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        TensorRef expectedA = expected.allocate(DType.F32, TensorShape.of(c.aRows(), c.aColumns()));
        TensorRef expectedB = expected.allocate(DType.F32, TensorShape.of(c.bRows(), c.bColumns()));
        TensorRef expectedResult = expected.allocate(DType.F32, TensorShape.of(c.resultRows(), c.resultColumns()));
        TensorRef actualA = actual.allocate(DType.F32, TensorShape.of(c.aRows(), c.aColumns()));
        TensorRef actualB = actual.allocate(DType.F32, TensorShape.of(c.bRows(), c.bColumns()));
        TensorRef actualResult = actual.allocate(DType.F32, TensorShape.of(c.resultRows(), c.resultColumns()));

        fill(expectedA, c.seed());
        fill(expectedB, c.seed() + 23);
        fill(actualA, c.seed());
        fill(actualB, c.seed() + 23);

        expected.batchDotProduct(operation(c, expectedResult, expectedA, expectedB));
        actual.batchDotProduct(operation(c, actualResult, actualA, actualB));

        for (int row = 0; row < c.resultRows(); row++) {
            for (int column = 0; column < c.resultColumns(); column++) {
                assertEquals(expectedResult.underlying().get(row, column), actualResult.underlying().get(row, column),
                        0.01f, c + " row=" + row + " column=" + column);
            }
        }
    }

    static Stream<Arguments> casesAndCandidates() {
        Candidate panama = new Candidate("PANAMA", true, new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        Candidate simd = new Candidate("SIMD", NativeOps.isAvailable(), new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.SIMD, new NativeOps(),
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        return BatchDotProductFuzzCases.cases()
                .flatMap(args -> Stream.of(panama, simd).map(candidate -> Arguments.of(args.get()[0], candidate)));
    }

    private static BatchDotProduct operation(BatchDotProductFuzzCases.Case c, TensorRef result, TensorRef a,
            TensorRef b) {
        return new BatchDotProduct()
                .result(result)
                .a(a)
                .b(b)
                .aRowOffset(c.aRowOffset())
                .aColumnOffset(c.aColumnOffset())
                .bColumnOffset(c.bColumnOffset())
                .columnLength(c.columnLength())
                .resultRowOffset(c.resultRowOffset())
                .bRowOffset(c.bRowOffset())
                .rowChunkSize(c.rowChunkSize());
    }

    private static Lighter naiveOnly() {
        return new Lighter(nullMetricRegistry(), Map.of(TensorProviderKind.NAIVE, new NaiveOps()));
    }

    private static io.dropwizard.metrics5.MetricRegistry nullMetricRegistry() {
        return new io.dropwizard.metrics5.MetricRegistry();
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                float value = ((row * 17 + column * 31 + seed) % 257 - 128) / 64.0f;
                tensor.underlying().set(value, row, column);
            }
        }
    }

    private record Candidate(String name, boolean enabled, Lighter lighter) {
        @Override
        public String toString() {
            return name;
        }
    }
}
