package io.teknek.deliverance.tensor2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BF16TensorTest {

    @Test
    void setAndGetTwoDimensionalValues() {
        BF16Tensor tensor = new BF16Tensor(3, 4);

        tensor.set(1.25f, 1, 2);

        assertEquals(1.25f, tensor.get(1, 2));
        assertEquals(0.0f, tensor.get(0, 0));
    }

    @Test
    void memorySegmentCoversBf16Storage() {
        BF16Tensor tensor = new BF16Tensor(2, 3);

        assertEquals(2 * 3 * Short.BYTES, tensor.getMemorySegment().byteSize());
    }
}
