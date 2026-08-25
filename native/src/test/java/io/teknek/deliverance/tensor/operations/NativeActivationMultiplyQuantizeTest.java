package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NativeActivationMultiplyQuantizeTest {

    static Stream<Arguments> providers() {
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(1));
        TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
        return Stream.of(
                Arguments.of("panama", panama),
                Arguments.of("native-simd", new NativeSimdTensorOperations(panama))
        );
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("providers")
    void fusedActivationMultiplyQuantizeMatchesSeparatePath(String name, TensorOperations ops) {
        try (AbstractTensor gate = tensor(3, 64, 3);
             AbstractTensor up = tensor(3, 64, 11);
             AbstractTensor separateHidden = new FloatBufferTensor(TensorShape.of(3, 64))) {

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
                        assertEquals(separate.get(row, col), fused.get(row, col), 0.008f,
                                name + " row=" + row + " col=" + col);
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
