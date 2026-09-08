package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterMultiplyFuzzTest {

    @ParameterizedTest(name = "{0} {1}")
    @MethodSource("casesAndCandidates")
    void multiplyAccumulateMatchesNaiveLighter(MultiplyAccumulateFuzzCases.Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is not implemented yet");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        TensorRef expectedA = expected.allocate(c.dType(), TensorShape.of(c.aRows(), c.columns()));
        TensorRef expectedB = expected.allocate(c.dType(), TensorShape.of(c.bRows(), c.columns()));
        TensorRef actualA = actual.allocate(c.dType(), TensorShape.of(c.aRows(), c.columns()));
        TensorRef actualB = actual.allocate(c.dType(), TensorShape.of(c.bRows(), c.columns()));

        fill(expectedA, c.seed());
        fill(expectedB, c.seed() + 23);
        fill(actualA, c.seed());
        fill(actualB, c.seed() + 23);

        expected.multiplyAccumulate(new MultiplyAccumulate(expectedB).into(expectedA).offsetAndLength(c.offset(), c.length()));
        actual.multiplyAccumulate(new MultiplyAccumulate(actualB).into(actualA).offsetAndLength(c.offset(), c.length()));

        for (int row = 0; row < c.aRows(); row++) {
            for (int column = 0; column < c.columns(); column++) {
                assertEquals(expectedA.underlying().get(row, column), actualA.underlying().get(row, column), 0.0f,
                        c + " row=" + row + " column=" + column);
            }
        }
    }

    static Stream<Arguments> casesAndCandidates() {
        Candidate panama = new Candidate("PANAMA", true, new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        Candidate simd = new Candidate("SIMD", false, new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.SIMD, (a, b, offset, length) -> Either.Left(OpSupport.Unsupported),
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        return MultiplyAccumulateFuzzCases.cases()
                .flatMap(args -> Stream.of(panama, simd).map(candidate -> Arguments.of(args.get()[0], candidate)));
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
