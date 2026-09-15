package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BatchDotProductTest {
    @ParameterizedTest(name = "{0}")
    @MethodSource("cases")
    void panamaMatchesNaiveWithInputRowOffsets(String name, int aRows, int bRows, int columns, int resultRows,
            int aRowOffset, int aColumnOffset, int bColumnOffset, int columnLength, int resultRowOffset,
            int bRowOffset, int rowChunkSize) {
        Allocator allocator = new Allocator();
        TensorRef a = allocator.allocate(DType.F32, TensorShape.of(aRows, columns));
        TensorRef b = allocator.allocate(DType.F32, TensorShape.of(bRows, columns));
        TensorRef naiveResult = allocator.allocate(DType.F32,
                TensorShape.of(resultRows, resultRowOffset + bRowOffset + rowChunkSize));
        TensorRef panamaResult = allocator.allocate(DType.F32,
                TensorShape.of(resultRows, resultRowOffset + bRowOffset + rowChunkSize));
        TensorRef lighterResult = allocator.allocate(DType.F32,
                TensorShape.of(resultRows, resultRowOffset + bRowOffset + rowChunkSize));
        fill(a, 1.0f);
        fill(b, -3.0f);

        BatchDotProduct naiveOperation = operation(naiveResult, a, b, aRowOffset, aColumnOffset, bColumnOffset,
                columnLength, resultRowOffset, bRowOffset, rowChunkSize);
        BatchDotProduct panamaOperation = operation(panamaResult, a, b, aRowOffset, aColumnOffset, bColumnOffset,
                columnLength, resultRowOffset, bRowOffset, rowChunkSize);
        BatchDotProduct lighterOperation = operation(lighterResult, a, b, aRowOffset, aColumnOffset, bColumnOffset,
                columnLength, resultRowOffset, bRowOffset, rowChunkSize);

        Either<OpSupport, Void> naive = new NaiveOps().batchDotProduct(naiveOperation);
        Either<OpSupport, Void> panama = new PanamaOps().batchDotProduct(panamaOperation);
        new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps(),
                TensorProviderKind.NAIVE, new NaiveOps())).batchDotProduct(lighterOperation);

        assertTrue(naive.isRight(), name);
        assertTrue(panama.isRight(), name);
        assertTensorEquals(naiveResult, panamaResult, name + " panama");
        assertTensorEquals(naiveResult, lighterResult, name + " lighter");
    }

    private static Stream<Arguments> cases() {
        return Stream.of(
                Arguments.of("full_rows", 4, 5, 16, 4, 0, 0, 0, 16, 0, 0, 5),
                Arguments.of("a_row_offset", 8, 7, 16, 3, 4, 0, 0, 16, 0, 1, 4),
                Arguments.of("column_offsets", 9, 8, 33, 4, 3, 5, 7, 19, 2, 2, 3),
                Arguments.of("tail_length", 7, 6, 65, 2, 5, 3, 1, 61, 1, 0, 6),
                Arguments.of("attention_tile_shape", 16, 20, 128, 8, 6, 0, 0, 128, 0, 0, 20)
        );
    }

    private static BatchDotProduct operation(TensorRef result, TensorRef a, TensorRef b, int aRowOffset,
            int aColumnOffset, int bColumnOffset, int columnLength, int resultRowOffset, int bRowOffset,
            int rowChunkSize) {
        return new BatchDotProduct()
                .result(result)
                .a(a)
                .b(b)
                .aRowOffset(aRowOffset)
                .aColumnOffset(aColumnOffset)
                .bColumnOffset(bColumnOffset)
                .columnLength(columnLength)
                .resultRowOffset(resultRowOffset)
                .bRowOffset(bRowOffset)
                .rowChunkSize(rowChunkSize);
    }

    private static void fill(TensorRef tensor, float seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(((row * 31 + column * 17 + (int) seed) % 127 - 63) / 16.0f,
                        row, column);
            }
        }
    }

    private static void assertTensorEquals(TensorRef expected, TensorRef actual, String label) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column),
                        0.0001f, label + " row=" + row + " column=" + column);
            }
        }
    }
}
