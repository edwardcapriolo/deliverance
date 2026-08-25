package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorOperationsActivationMultiplyQuantizeTest {

    @Test
    void fusedActivationMultiplyQuantizeMatchesSeparatePath() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1))) {
            MetricRegistry metrics = new MetricRegistry();
            TensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(metrics), pool);
            try (AbstractTensor gate = tensor(2, 64, 3);
                 AbstractTensor up = tensor(2, 64, 11);
                 AbstractTensor separateHidden = new FloatBufferTensor(TensorShape.of(2, 64))) {

                for (int row = 0; row < gate.shape().first(); row++) {
                    for (int col = 0; col < gate.shape().last(); col++) {
                        separateHidden.set(ActivationFunction.eval(ActivationFunction.Type.SILU, gate.get(row, col))
                                * up.get(row, col), row, col);
                    }
                }
                try (AbstractTensor separate = ops.quantize(separateHidden, DType.I8, 0, 64);
                     AbstractTensor fused = ops.activationMultiplyQuantize(gate, up, ActivationFunction.Type.SILU,
                             DType.I8, 0, 64)) {
                    for (int row = 0; row < gate.shape().first(); row++) {
                        for (int col = 0; col < gate.shape().last(); col++) {
                            assertEquals(separate.get(row, col), fused.get(row, col), 1.0e-6f,
                                    "row=" + row + " col=" + col);
                        }
                    }
                }
            }
        }
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(rows, cols));
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 17 + col * 31 + seed) % 41) - 20) / 20.0f, row, col);
            }
        }
        return tensor;
    }
}
