package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class F32TensorTest {

    @Test
    void setAndGetTwoDimensionalValues() {
        F32Tensor tensor = new F32Tensor(3, 4);

        tensor.set(1.25f, 1, 2);

        assertEquals(1.25f, tensor.get(1, 2));
        assertEquals(0.0f, tensor.get(0, 0));
    }

    @Test
    void setAndGetVarargValues() {
        F32Tensor tensor = new F32Tensor(2, 3, 4);

        tensor.set(7.5f, 1, 2, 3);

        assertEquals(7.5f, tensor.get(1, 2, 3));
    }

    @Test
    void memorySegmentCoversFloatStorage() {
        F32Tensor tensor = new F32Tensor(2, 3);

        assertEquals(2 * 3 * Float.BYTES, tensor.getMemorySegment().byteSize());
    }
}
