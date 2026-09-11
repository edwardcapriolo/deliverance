package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.dysfx.Either;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PanamaOpsTest {

    private record ScaleCase(int rows, int columns, int offset, int length, float factor) { }

    private static List<ScaleCase> scaleCases() {
        return List.of(
                new ScaleCase(1, 3, 0, 3, 2.0f),
                new ScaleCase(2, 5, 1, 3, -1.5f),
                new ScaleCase(3, 16, 0, 16, 0.25f),
                new ScaleCase(2, 17, 1, 15, 3.0f),
                new ScaleCase(4, 32, 0, 32, -0.5f),
                new ScaleCase(2, 33, 1, 31, 1.25f),
                new ScaleCase(3, 64, 0, 64, 0.75f),
                new ScaleCase(2, 65, 2, 61, -2.0f)
        );
    }

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

    @Test
    void scaleF32MatchesPanamaTensorOperations() {
        Allocator allocator = new Allocator();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations oldOps = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    Mockito.mock(TensorAllocator.class), pool);
            for (ScaleCase scaleCase : scaleCases()) {
                TensorRef actual = allocator.allocate(DType.F32, TensorShape.of(scaleCase.rows(), scaleCase.columns()));
                FloatBufferTensor expected = new FloatBufferTensor(scaleCase.rows(), scaleCase.columns());
                fill(actual, 1.0f);
                fill(expected, 1.0f);

                oldOps.scale(scaleCase.factor(), expected, scaleCase.offset(), scaleCase.length());
                Either<OpSupport, Void> result = new PanamaOps().scale(scaleCase.factor(), actual,
                        scaleCase.offset(), scaleCase.length());

                assertTrue(result.isRight(), scaleCase::toString);
                assertEquals(TensorDisplayUtil.pretty2dDisplayAll(expected).trim(), pretty(actual).trim(),
                        scaleCase::toString);
            }
        }
    }

    @Test
    void scaleBF16MatchesPanamaTensorOperations() {
        Allocator allocator = new Allocator();
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            PanamaTensorOperations oldOps = new PanamaTensorOperations(MachineSpec.Type.AVX_256,
                    Mockito.mock(TensorAllocator.class), pool);
            for (ScaleCase scaleCase : scaleCases()) {
                TensorRef actual = allocator.allocate(DType.BF16, TensorShape.of(scaleCase.rows(), scaleCase.columns()));
                BFloat16BufferTensor expected = new BFloat16BufferTensor(scaleCase.rows(), scaleCase.columns());
                fill(actual, 1.0f);
                fill(expected, 1.0f);

                oldOps.scale(scaleCase.factor(), expected, scaleCase.offset(), scaleCase.length());
                Either<OpSupport, Void> result = new PanamaOps().scale(scaleCase.factor(), actual,
                        scaleCase.offset(), scaleCase.length());

                assertTrue(result.isRight(), scaleCase::toString);
                assertEquals(TensorDisplayUtil.pretty2dDisplayAll(expected).trim(), pretty(actual).trim(),
                        scaleCase::toString);
            }
        }
    }

    private static void fill(TensorRef tensor, float start) {
        float value = start;
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.underlying().set(value++, row, column);
            }
        }
    }

    private static void fill(FloatBufferTensor tensor, float start) {
        float value = start;
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(value++, row, column);
            }
        }
    }

    private static void fill(BFloat16BufferTensor tensor, float start) {
        float value = start;
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(value++, row, column);
            }
        }
    }

    private static String pretty(TensorRef tensor) {
        StringBuilder builder = new StringBuilder();
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                builder.append(String.format("[%d][%d]=%8.4f ", row, column, tensor.underlying().get(row, column)));
            }
            builder.append('\n');
        }
        return builder.toString();
    }
}
