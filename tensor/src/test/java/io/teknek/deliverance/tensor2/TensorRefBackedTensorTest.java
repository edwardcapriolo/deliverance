package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorRefBackedTensorTest {

    @Test
    void exposesReshapedBf16AsOldAbstractTensorForCopyFrom() {
        Lighter lighter = new Lighter();
        try (TensorRef f32 = lighter.allocate(DType.F32, TensorShape.of(2, 32))) {
            for (int row = 0; row < f32.shape().first(); row++) {
                for (int column = 0; column < f32.shape().last(); column++) {
                    f32.underlying().set(((row * 17 + column * 31) % 127 - 63) / 32.0f, row, column);
                }
            }

            try (TensorRefBackedTensor converted = new TensorRefBackedTensor(lighter.reshape(f32, DType.BF16));
                 BFloat16BufferTensor destination = new BFloat16BufferTensor(3, 32)) {
                destination.copyFrom(converted, 0, destination.getOffset(1, 0), (int) converted.size());

                for (int row = 0; row < converted.shape().first(); row++) {
                    for (int column = 0; column < converted.shape().last(); column++) {
                        assertEquals(converted.get(row, column), destination.get(row + 1, column), 0.0f,
                                "row=" + row + " column=" + column);
                    }
                }
            }
        }
    }

    @Test
    void canBorrowOldTensorAndBridgeReshapedResultBack() {
        Lighter lighter = new Lighter();
        try (FloatBufferTensor source = new FloatBufferTensor(1, 32)) {
            for (int column = 0; column < source.shape().last(); column++) {
                source.set((column - 16) / 8.0f, 0, column);
            }
            try (TensorRef sourceRef = TensorRef.borrowed(source);
                 TensorRefBackedTensor converted = new TensorRefBackedTensor(lighter.reshape(sourceRef, DType.BF16))) {
                assertEquals(DType.BF16, converted.dType());
                assertEquals(source.shape(), converted.shape());
                assertEquals(source.get(0, 5), converted.get(0, 5), 0.01f);
            }
        }
    }
}
