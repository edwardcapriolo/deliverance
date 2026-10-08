package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoraTensorMathTest {

    @Test
    void sameDtypeReturnsOriginalTensor() {
        try (FloatBufferTensor source = new FloatBufferTensor(2, 2)) {
            assertSame(source, LoraTensorMath.toDType(source, DType.F32));
        }
    }

    @Test
    void convertsAndScalesValuesToBfloat16() {
        try (FloatBufferTensor source = values()) {
            try (AbstractTensor converted = LoraTensorMath.scaledCopy(source, DType.BF16, 2.5f)) {
                assertEquals(DType.BF16, converted.dType());
                assertEquals(2.5f, converted.get(0, 0), 0.02f);
                assertEquals(-5.0f, converted.get(0, 1), 0.02f);
                assertEquals(7.5f, converted.get(1, 0), 0.02f);
                assertEquals(-10.0f, converted.get(1, 1), 0.02f);
            }
        }
    }

    @Test
    void convertsValuesToFloat16() {
        try (FloatBufferTensor source = values();
             AbstractTensor converted = LoraTensorMath.toDType(source, DType.F16)) {
            assertEquals(DType.F16, converted.dType());
            assertEquals(1.0f, converted.get(0, 0), 0.01f);
            assertEquals(-2.0f, converted.get(0, 1), 0.01f);
            assertEquals(3.0f, converted.get(1, 0), 0.01f);
            assertEquals(-4.0f, converted.get(1, 1), 0.01f);
        }
    }

    @Test
    void rejectsNonDenseTargetDtypes() {
        try (FloatBufferTensor source = values()) {
            assertThrows(UnsupportedOperationException.class,
                    () -> LoraTensorMath.toDType(source, DType.Q4));
            assertThrows(UnsupportedOperationException.class,
                    () -> LoraTensorMath.allocateLike(DType.I8, 2, 2));
        }
    }

    private static FloatBufferTensor values() {
        FloatBufferTensor source = new FloatBufferTensor(2, 2);
        source.set(1.0f, 0, 0);
        source.set(-2.0f, 0, 1);
        source.set(3.0f, 1, 0);
        source.set(-4.0f, 1, 1);
        return source;
    }
}
