package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.DType;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoraTensorMathTest {

    private final Lighter lighter = new Lighter();

    @Test
    void sameDtypeReturnsOriginalTensor() {
        try (TensorRef source = values()) {
            assertSame(source, LoraTensorMath.toDType(source, DType.F32, lighter));
        }
    }

    @Test
    void convertsAndScalesValuesToBfloat16() {
        try (TensorRef source = values()) {
            try (TensorRef converted = LoraTensorMath.scaledCopy(source, DType.BF16, 2.5f, lighter)) {
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
        try (TensorRef source = values();
             TensorRef converted = LoraTensorMath.toDType(source, DType.F16, lighter)) {
            assertEquals(DType.F16, converted.dType());
            assertEquals(1.0f, converted.get(0, 0), 0.01f);
            assertEquals(-2.0f, converted.get(0, 1), 0.01f);
            assertEquals(3.0f, converted.get(1, 0), 0.01f);
            assertEquals(-4.0f, converted.get(1, 1), 0.01f);
        }
    }

    @Test
    void rejectsNonDenseTargetDtypes() {
        try (TensorRef source = values()) {
            assertThrows(UnsupportedOperationException.class,
                    () -> LoraTensorMath.toDType(source, DType.Q4, lighter));
        }
    }

    private TensorRef values() {
        TensorRef source = lighter.allocate(DType.F32, TensorShape.of(2, 2));
        source.set(1.0f, 0, 0);
        source.set(-2.0f, 0, 1);
        source.set(3.0f, 1, 0);
        source.set(-4.0f, 1, 1);
        return source;
    }
}
