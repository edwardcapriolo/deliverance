package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

public final class TensorTestSupport {
    private TensorTestSupport() {
    }

    /** Creates an F32 tensor from row-major values. */
    public static FloatBufferTensor tensorOf(int rows, int cols, float... values) {
        assertEquals(rows * cols, values.length, "tensor value count must match shape");
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        int index = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(values[index++], row, col);
            }
        }
        return tensor;
    }

    /** Creates a deterministic F32 tensor from a small integer hash, useful for repeatable test fixtures. */
    public static FloatBufferTensor deterministicTensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 13 + col * 7 + seed) % 17 - 8) / 8.0f, row, col);
            }
        }
        return tensor;
    }
}
