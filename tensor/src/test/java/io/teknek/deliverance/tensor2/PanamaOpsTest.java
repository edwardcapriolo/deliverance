package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.dysfx.Either;
import org.junit.jupiter.api.Test;

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
                TensorShape.of(1, 3), DType.BF16, 0, "cpu", java.util.Map.of(), null));

        Either<OpSupport, Void> result = new PanamaOps().multiplyAccumulate(a, b, 0, 3);

        assertTrue(result.isLeft());
    }

    @Test
    void scaleF32MatchesReference() {
        Allocator allocator = new Allocator();
        for (ScaleCase scaleCase : scaleCases()) {
            TensorRef actual = allocator.allocate(DType.F32, TensorShape.of(scaleCase.rows(), scaleCase.columns()));
            float[][] expected = filledReference(scaleCase.rows(), scaleCase.columns(), 1.0f);
            fill(actual, 1.0f);
            multiplyReference(expected, scaleCase.factor(), scaleCase.offset(), scaleCase.length());

            Either<OpSupport, Void> result = new PanamaOps().scale(scaleCase.factor(), actual,
                    scaleCase.offset(), scaleCase.length());

            assertTrue(result.isRight(), scaleCase::toString);
            assertReference(expected, actual, 1.0e-6f, scaleCase.toString());
        }
    }

    @Test
    void scaleBF16MatchesReference() {
        Allocator allocator = new Allocator();
        for (ScaleCase scaleCase : scaleCases()) {
            TensorRef actual = allocator.allocate(DType.BF16, TensorShape.of(scaleCase.rows(), scaleCase.columns()));
            float[][] expected = filledReference(scaleCase.rows(), scaleCase.columns(), 1.0f);
            fill(actual, 1.0f);
            multiplyReference(expected, scaleCase.factor(), scaleCase.offset(), scaleCase.length());

            Either<OpSupport, Void> result = new PanamaOps().scale(scaleCase.factor(), actual,
                    scaleCase.offset(), scaleCase.length());

            assertTrue(result.isRight(), scaleCase::toString);
            assertReference(expected, actual, scaleCase.toString());
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

    private static float[][] filledReference(int rows, int columns, float start) {
        float[][] values = new float[rows][columns];
        float value = start;
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                values[row][column] = value++;
            }
        }
        return values;
    }

    private static void multiplyReference(float[][] values, float factor, int offset, int length) {
        for (float[] row : values) {
            for (int column = offset; column < offset + length; column++) {
                row[column] *= factor;
            }
        }
    }

    private static void assertReference(float[][] expected, TensorRef actual, float tolerance, String label) {
        for (int row = 0; row < expected.length; row++) {
            for (int column = 0; column < expected[row].length; column++) {
                assertEquals(expected[row][column], actual.underlying().get(row, column), tolerance,
                        label + " row=" + row + " column=" + column);
            }
        }
    }

    private static void assertReference(float[][] expected, TensorRef actual, String label) {
        for (int row = 0; row < expected.length; row++) {
            for (int column = 0; column < expected[row].length; column++) {
                float tolerance = Math.max(0.08f, Math.abs(expected[row][column]) * 0.01f);
                assertEquals(expected[row][column], actual.underlying().get(row, column), tolerance,
                        label + " row=" + row + " column=" + column);
            }
        }
    }
}
