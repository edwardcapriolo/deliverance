package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterExpTest {
    @Test
    void expWritesOnlyTheRequestedWindow() {
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(2, 9));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 9))) {
            for (int row = 0; row < 2; row++) {
                for (int column = 0; column < 9; column++) {
                    input.set(column - 4.0f, row, column);
                    output.set(-99.0f, row, column);
                }
            }
            lighter.exp(new Exp(input).into(output).offsetAndLength(2, 5));
            assertEquals(-99.0f, output.get(0, 1));
            assertEquals(Math.exp(-2.0), output.get(0, 2), 1.0e-6);
            assertEquals(Math.exp(2.0), output.get(1, 6), 1.0e-6);
            assertEquals(-99.0f, output.get(1, 7));
        }
    }

    @Test
    void nativeExpUsesExistingSimdKernel() {
        Lighter lighter = new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.SIMD, new NativeOps()));
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(2, 17));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 17))) {
            for (int row = 0; row < 2; row++) {
                for (int column = 0; column < 17; column++) {
                    input.set((row * 17 + column - 12) / 8.0f, row, column);
                }
            }
            lighter.exp(new Exp(input).into(output).offsetAndLength(2, 11));
            for (int row = 0; row < 2; row++) {
                for (int column = 2; column < 13; column++) {
                    assertEquals(Math.exp(input.get(row, column)), output.get(row, column), 1.0e-5);
                }
            }
        }
    }
}
