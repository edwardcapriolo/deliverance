package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PanamaOpsTest {

    @Test
    void multiplyAccumulateF32ByF32() {
        Allocator allocator = new Allocator();
        TensorRef a = allocator.allocate(DType.F32, TensorShape.of(2, 5));
        TensorRef b = allocator.allocate(DType.F32, TensorShape.of(2, 5));
        fill(a, 1.0f);
        fill(b, 10.0f);

        Either<OpSupport, Void> result = new PanamaOps().multiplyAccumulate(a, b, 1, 3);

        assertTrue(result.isRight());
        assertEquals(1.0f, a.underlying().get(0, 0));
        assertEquals(22.0f, a.underlying().get(0, 1));
        assertEquals(36.0f, a.underlying().get(0, 2));
        assertEquals(52.0f, a.underlying().get(0, 3));
        assertEquals(5.0f, a.underlying().get(0, 4));
        assertEquals(162.0f, a.underlying().get(1, 3));
    }

    @Test
    void multiplyAccumulateBroadcastsSingleSourceRow() {
        Allocator allocator = new Allocator();
        TensorRef a = allocator.allocate(DType.F32, TensorShape.of(2, 3));
        TensorRef b = allocator.allocate(DType.F32, TensorShape.of(1, 3));
        fill(a, 1.0f);
        b.underlying().set(10.0f, 0, 0);
        b.underlying().set(20.0f, 0, 1);
        b.underlying().set(30.0f, 0, 2);

        Either<OpSupport, Void> result = new PanamaOps().multiplyAccumulate(a, b, 0, 3);

        assertTrue(result.isRight());
        assertEquals(10.0f, a.underlying().get(0, 0));
        assertEquals(100.0f, a.underlying().get(1, 1));
        assertEquals(180.0f, a.underlying().get(1, 2));
    }

    @Test
    void unsupportedDTypeReturnsUnsupported() {
        Allocator allocator = new Allocator();
        TensorRef a = allocator.allocate(DType.F32, TensorShape.of(1, 3));
        TensorRef b = new TensorRef(new TensorRefState(LeaseState.USED, allocator, a.underlying(),
                TensorShape.of(1, 3), DType.BF16, 0, "cpu", null));

        Either<OpSupport, Void> result = new PanamaOps().multiplyAccumulate(a, b, 0, 3);

        assertTrue(result.isLeft());
    }

    private static void fill(TensorRef tensor, float start) {
        float value = start;
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(value++, row, column);
            }
        }
    }
}
