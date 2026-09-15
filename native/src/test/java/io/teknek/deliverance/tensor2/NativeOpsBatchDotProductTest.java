package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeOpsBatchDotProductTest {
    @Test
    void nativeBatchDotProductMatchesNaiveWithInputRowOffset() {
        Allocator allocator = new Allocator();
        TensorRef a = allocator.allocate(DType.F32, TensorShape.of(8, 16));
        TensorRef b = allocator.allocate(DType.F32, TensorShape.of(6, 16));
        TensorRef expected = allocator.allocate(DType.F32, TensorShape.of(3, 7));
        TensorRef actual = allocator.allocate(DType.F32, TensorShape.of(3, 7));
        fill(a, 3);
        fill(b, 11);

        BatchDotProduct expectedOp = operation(expected, a, b);
        BatchDotProduct actualOp = operation(actual, a, b);

        assertTrue(new NaiveOps().batchDotProduct(expectedOp).isRight());
        assertTrue(new NativeOps().batchDotProduct(actualOp).isRight());
        assertClose(expected, actual);
    }

    private static BatchDotProduct operation(TensorRef result, TensorRef a, TensorRef b) {
        return new BatchDotProduct()
                .result(result)
                .a(a)
                .b(b)
                .aRowOffset(4)
                .aColumnOffset(2)
                .bColumnOffset(1)
                .columnLength(11)
                .resultRowOffset(1)
                .bRowOffset(1)
                .rowChunkSize(5);
    }

    private static void fill(TensorRef tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(((row * 13 + column * 7 + seed) % 41 - 20) / 8.0f, row, column);
            }
        }
    }

    private static void assertClose(TensorRef expected, TensorRef actual) {
        assertEquals(expected.shape(), actual.shape());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int column = 0; column < expected.shape().last(); column++) {
                assertEquals(expected.underlying().get(row, column), actual.underlying().get(row, column), 0.0001f,
                        "row=" + row + " column=" + column);
            }
        }
    }
}
