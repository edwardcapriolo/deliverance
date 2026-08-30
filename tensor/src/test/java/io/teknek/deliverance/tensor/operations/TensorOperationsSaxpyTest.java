package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TensorOperationsSaxpyTest {

    @Test
    public void saxpyUsesAlphaAndInputRowOffsetsForBatchWindow() {
        TensorOperations ops = new NaiveTensorOperations();
        try (AbstractTensor alpha = new FloatBufferTensor(1, 5);
             AbstractTensor x = new FloatBufferTensor(4, 3);
             AbstractTensor y = new FloatBufferTensor(1, 3)) {
            for (int i = 0; i < 5; i++) {
                alpha.set(i + 1, 0, i);
            }
            int value = 1;
            for (int row = 0; row < 4; row++) {
                for (int col = 0; col < 3; col++) {
                    x.set(value++, row, col);
                }
            }

            ops.saxpy(alpha, x, y, 0, 0, 3, 2, 1, 2);

            assertEquals("[0][0]= 40.0000 [0][1]= 47.0000 [0][2]= 54.0000".trim(),
                    TensorDisplayUtil.pretty2dDisplayAll(y).trim());
        }
    }

    @Test
    public void saxpySupportsI8InputAndF32Output() {
        TensorOperations ops = panamaOps();
        try (AbstractTensor dense = new FloatBufferTensor(2, 64);
             AbstractTensor alpha = new FloatBufferTensor(1, 2);
             AbstractTensor expected = new FloatBufferTensor(1, 64);
             AbstractTensor actual = new FloatBufferTensor(1, 64)) {
            for (int row = 0; row < 2; row++) {
                for (int col = 0; col < 64; col++) {
                    dense.set((row + 1) * ((col % 13) - 6) / 7.0f, row, col);
                }
            }
            alpha.set(0.25f, 0, 0);
            alpha.set(-0.5f, 0, 1);
            try (AbstractTensor i8 = AbstractTensorUtils.quantize(dense, DType.I8, true)) {
                new NaiveTensorOperations().saxpy(alpha, i8, expected, 0, 0, 64, 0, 0, 2);
                ops.saxpy(alpha, i8, actual, 0, 0, 64, 0, 0, 2);
            }
            for (int col = 0; col < 64; col++) {
                assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-6f, "col=" + col);
            }
        }
    }

    private TensorOperations panamaOps() {
        return new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                new io.teknek.deliverance.tensor.ArrayQueueTensorAllocator(new io.dropwizard.metrics5.MetricRegistry()),
                new io.teknek.deliverance.math.WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2)));
    }
}
