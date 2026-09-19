package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class AllocatorQuantizedTest {

    @Test
    void allocatesI8WithScaleSidecar() {
        Lighter lighter = new Lighter();
        try (TensorRef i8 = lighter.allocate(DType.I8, TensorShape.of(2, 64))) {
            TensorRef scale = Q8Layout.scale(i8);
            assertNotNull(scale);
            assertEquals(DType.I8, i8.dType());
            assertEquals(DType.F32, scale.dType());
            assertEquals(TensorShape.of(2, 2), scale.shape());

            scale.underlying().set(0.5f, 1, 1);
            ((I8Tensor) i8.underlying()).setRawByte((byte) -6, 1, 40);

            assertEquals(-3.0f, i8.underlying().get(1, 40), 0.0f);
        }
    }

    @Test
    void reattachesScaleSidecarWhenI8TensorIsReused() {
        Lighter lighter = new Lighter();
        try (TensorRef first = lighter.allocate(DType.I8, TensorShape.of(1, 32))) {
            Q8Layout.scale(first).underlying().set(2.0f, 0, 0);
            ((I8Tensor) first.underlying()).setRawByte((byte) 3, 0, 0);
            assertEquals(6.0f, first.underlying().get(0, 0), 0.0f);
        }

        try (TensorRef second = lighter.allocate(DType.I8, TensorShape.of(1, 32))) {
            Q8Layout.scale(second).underlying().set(4.0f, 0, 0);
            ((I8Tensor) second.underlying()).setRawByte((byte) 3, 0, 0);
            assertEquals(12.0f, second.underlying().get(0, 0), 0.0f);
        }
    }

    @Test
    void rejectsI8ShapesThatCannotHaveCompleteBlocks() {
        Lighter lighter = new Lighter();
        assertThrows(IllegalArgumentException.class, () -> lighter.allocate(DType.I8, TensorShape.of(2, 33)));
    }

    @Test
    void allocatesQ4WithScaleSidecar() {
        Lighter lighter = new Lighter();
        try (TensorRef q4 = lighter.allocate(DType.Q4, TensorShape.of(2, 64))) {
            TensorRef scale = Q4Layout.scale(q4);
            assertNotNull(scale);
            assertEquals(DType.Q4, q4.dType());
            assertEquals(DType.F32, scale.dType());
            assertEquals(TensorShape.of(2, 2), scale.shape());

            scale.underlying().set(0.25f, 1, 1);
            ((Q4Tensor) q4.underlying()).setPackedByte((byte) 0xF5, 1, 1, 8);

            assertEquals(-0.75f, q4.underlying().get(1, 40), 0.0f);
            assertEquals(1.75f, q4.underlying().get(1, 56), 0.0f);
            assertEquals((byte) 0xF5, ((Q4Tensor) q4.underlying()).getPackedByte(1, 1, 8));
        }
    }

    @Test
    void reattachesScaleSidecarWhenQ4TensorIsReused() {
        Lighter lighter = new Lighter();
        try (TensorRef first = lighter.allocate(DType.Q4, TensorShape.of(1, 32))) {
            Q4Layout.scale(first).underlying().set(2.0f, 0, 0);
            ((Q4Tensor) first.underlying()).setPackedByte((byte) 0xB9, 0, 0, 0);
            assertEquals(2.0f, first.underlying().get(0, 0), 0.0f);
            assertEquals(6.0f, first.underlying().get(0, 16), 0.0f);
        }

        try (TensorRef second = lighter.allocate(DType.Q4, TensorShape.of(1, 32))) {
            Q4Layout.scale(second).underlying().set(4.0f, 0, 0);
            ((Q4Tensor) second.underlying()).setPackedByte((byte) 0xB9, 0, 0, 0);
            assertEquals(4.0f, second.underlying().get(0, 0), 0.0f);
            assertEquals(12.0f, second.underlying().get(0, 16), 0.0f);
        }
    }

    @Test
    void rejectsQ4ShapesThatCannotHaveCompleteBlocks() {
        Lighter lighter = new Lighter();
        assertThrows(IllegalArgumentException.class, () -> lighter.allocate(DType.Q4, TensorShape.of(2, 33)));
    }
}
