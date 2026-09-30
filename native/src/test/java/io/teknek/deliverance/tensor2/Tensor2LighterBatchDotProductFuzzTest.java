package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
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

    @ParameterizedTest(name = "q8 {0} {1}")
    @MethodSource("q8CasesAndCandidates")
    void batchDotProductF32Q8MatchesNaiveLighter(BatchDotProductFuzzCases.Case c, Candidate candidate) {
        Assumptions.assumeTrue(candidate.enabled(), candidate.name() + " is not implemented yet");
        Lighter expected = naiveOnly();
        Lighter actual = candidate.lighter();
        TensorRef expectedA = expected.allocate(DType.F32, TensorShape.of(c.aRows(), c.aColumns()));
        TensorRef expectedResult = expected.allocate(DType.F32, TensorShape.of(c.resultRows(), c.resultColumns()));
        TensorRef actualA = actual.allocate(DType.F32, TensorShape.of(c.aRows(), c.aColumns()));
        TensorRef actualResult = actual.allocate(DType.F32, TensorShape.of(c.resultRows(), c.resultColumns()));
        Q8ByteBufferTensor expectedQ8 = q8Tensor(c.bRows(), c.bColumns(), c.seed() + 23);
        Q8ByteBufferTensor actualQ8 = q8Tensor(c.bRows(), c.bColumns(), c.seed() + 23);
        TensorRef expectedB = TensorRef.borrowed(expectedQ8);
        TensorRef actualB = TensorRef.borrowed(actualQ8);

        fill(expectedA, c.seed());
        fill(actualA, c.seed());

        expected.batchDotProduct(operation(c, expectedResult, expectedA, expectedB));
        actual.batchDotProduct(operation(c, actualResult, actualA, actualB));

        for (int row = 0; row < c.resultRows(); row++) {
            for (int column = 0; column < c.resultColumns(); column++) {
                assertEquals(expectedResult.underlying().get(row, column), actualResult.underlying().get(row, column),
                        0.03f, c + " row=" + row + " column=" + column);
            }
        }
    }

    @ParameterizedTest(name = "panama strict {0} {1}")
    @MethodSource("panamaStrictDTypeCases")
    void panamaBatchDotProductSupportsProductionDTypesWithoutFallback(DTypeCase dtypeCase,
            BatchDotProductFuzzCases.Case c) {
        Lighter expected = naiveOnly();
        io.dropwizard.metrics5.MetricRegistry metrics = new io.dropwizard.metrics5.MetricRegistry();
        Lighter actual = new Lighter(metrics, Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        TensorRef expectedA = operand(expected, dtypeCase.aType(), c.aRows(), c.aColumns(), c.seed());
        TensorRef expectedB = operand(expected, dtypeCase.bType(), c.bRows(), c.bColumns(), c.seed() + 23);
        TensorRef expectedResult = expected.allocate(DType.F32, TensorShape.of(c.resultRows(), c.resultColumns()));
        TensorRef actualA = operand(actual, dtypeCase.aType(), c.aRows(), c.aColumns(), c.seed());
        TensorRef actualB = operand(actual, dtypeCase.bType(), c.bRows(), c.bColumns(), c.seed() + 23);
        TensorRef actualResult = actual.allocate(DType.F32, TensorShape.of(c.resultRows(), c.resultColumns()));

        expected.batchDotProduct(operation(c, expectedResult, expectedA, expectedB));
        actual.batchDotProduct(operation(c, actualResult, actualA, actualB));

        assertClose(dtypeCase, c, expectedResult, actualResult);
        assertEquals(1, metrics.meter(new io.dropwizard.metrics5.MetricName("tensor2.batch_dot_product",
                Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.PANAMA.name()))).getCount(), dtypeCase + " " + c);
        assertEquals(0, metrics.meter(new io.dropwizard.metrics5.MetricName("tensor2.batch_dot_product",
                Map.of(Lighter.TENSOR_OP_KEY, TensorProviderKind.NAIVE.name()))).getCount(), dtypeCase + " " + c);
    }

    static Stream<Arguments> casesAndCandidates() {
        Candidate panama = new Candidate("PANAMA", true, () -> new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        Candidate simd = new Candidate("SIMD", NativeOps.isAvailable(), () -> new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.SIMD, new NativeOps(),
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        return BatchDotProductFuzzCases.cases()
                .flatMap(args -> Stream.of(panama, simd).map(candidate -> Arguments.of(args.get()[0], candidate)));
    }

    static Stream<Arguments> q8CasesAndCandidates() {
        Candidate panama = new Candidate("PANAMA", true, () -> new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        Candidate simd = new Candidate("SIMD", NativeOps.isAvailable(), () -> new Lighter(nullMetricRegistry(), Map.of(
                TensorProviderKind.SIMD, new NativeOps(),
                TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())));
        return BatchDotProductFuzzCases.q8Cases()
                .flatMap(args -> Stream.of(panama, simd).map(candidate -> Arguments.of(args.get()[0], candidate)));
    }

    static Stream<Arguments> panamaStrictDTypeCases() {
        List<DTypeCase> denseCases = List.of(
                new DTypeCase(DType.F32, DType.BF16, 0.03f),
                new DTypeCase(DType.BF16, DType.BF16, 0.06f));
        List<DTypeCase> q4Cases = List.of(
                new DTypeCase(DType.F32, DType.Q4, 0.08f),
                new DTypeCase(DType.I8, DType.Q4, 0.08f),
                new DTypeCase(DType.BF16, DType.Q4, 0.10f));
        return Stream.concat(
                BatchDotProductFuzzCases.cases().flatMap(args -> denseCases.stream()
                        .map(dtypeCase -> Arguments.of(dtypeCase, args.get()[0]))),
                BatchDotProductFuzzCases.q8Cases().flatMap(args -> q4Cases.stream()
                        .map(dtypeCase -> Arguments.of(dtypeCase, args.get()[0]))));
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

    private static TensorRef operand(Lighter lighter, DType dType, int rows, int columns, int seed) {
        if (dType == DType.Q4) {
            return TensorRef.owned(q4Tensor(rows, columns, seed));
        }
        TensorRef dense = lighter.allocate(DType.F32, TensorShape.of(rows, columns));
        fill(dense, seed);
        if (dType == DType.F32) {
            return dense;
        }
        TensorRef converted = lighter.reshape(dense, dType);
        dense.close();
        return converted;
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

    private static Q8ByteBufferTensor q8Tensor(int rows, int columns, int seed) {
        FloatBufferTensor dense = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                float value = ((row * 13 + column * 19 + seed) % 251 - 125) / 96.0f;
                dense.set(value, row, column);
            }
        }
        return new Q8ByteBufferTensor(dense);
    }

    private static Q4ByteBufferTensor q4Tensor(int rows, int columns, int seed) {
        FloatBufferTensor dense = new FloatBufferTensor(TensorShape.of(rows, columns));
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                float value = ((row * 13 + column * 19 + seed) % 251 - 125) / 96.0f;
                dense.set(value, row, column);
            }
        }
        return new Q4ByteBufferTensor(dense);
    }

    private static void assertClose(DTypeCase dtypeCase, BatchDotProductFuzzCases.Case c, TensorRef expected,
            TensorRef actual) {
        for (int row = 0; row < c.resultRows(); row++) {
            for (int column = 0; column < c.resultColumns(); column++) {
                assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column),
                        dtypeCase.tolerance(), dtypeCase + " " + c + " row=" + row + " column=" + column);
            }
        }
    }

    private record DTypeCase(DType aType, DType bType, float tolerance) {
        @Override
        public String toString() {
            return aType + "x" + bType;
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
