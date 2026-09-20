package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class F16TensorTest {

    @Test
    void setAndGetTwoDimensionalValues() {
        F16Tensor tensor = new F16Tensor(3, 4);

        tensor.set(1.25f, 1, 2);

        assertEquals(1.25f, tensor.get(1, 2));
        assertEquals(0.0f, tensor.get(0, 0));
    }

    @Test
    void memorySegmentCoversF16Storage() {
        F16Tensor tensor = new F16Tensor(2, 3);

        assertEquals(2 * 3 * Short.BYTES, tensor.getMemorySegment().byteSize());
    }
}
