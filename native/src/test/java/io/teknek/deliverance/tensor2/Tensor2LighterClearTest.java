package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterClearTest {
    private final Lighter lighter = new Lighter(new MetricRegistry());

    @Test
    void clearZerosDensePayload() {
        try (TensorRef tensor = lighter.allocate(DType.F32, TensorShape.of(2, 8))) {
            tensor.set(3.0f, 1, 4);
            lighter.clear(tensor);
            assertEquals(0.0f, tensor.get(1, 4));
        }
    }

    @Test
    void clearZerosI8PayloadAndScale() {
        try (TensorRef tensor = lighter.allocate(DType.I8, TensorShape.of(1, 32))) {
            tensor.sidecar(Q8Layout.SCALE_SIDECAR).set(2.0f, 0, 0);
            tensor.set(7.0f, 0, 0);
            lighter.clear(tensor);
            assertEquals(0.0f, Math.abs(tensor.get(0, 0)));
            assertEquals(0.0f, tensor.sidecar(Q8Layout.SCALE_SIDECAR).get(0, 0));
        }
    }

    @Test
    void clearZerosQ4PayloadAndScale() {
        try (TensorRef tensor = lighter.allocate(DType.Q4, TensorShape.of(1, 32))) {
            tensor.sidecar(Q4Layout.SCALE_SIDECAR).set(2.0f, 0, 0);
            tensor.memorySegment().set(ValueLayout.JAVA_BYTE, 0, (byte) 0xFF);
            lighter.clear(tensor);
            assertEquals(0.0f, Math.abs(tensor.get(0, 0)));
            assertEquals(0.0f, tensor.sidecar(Q4Layout.SCALE_SIDECAR).get(0, 0));
        }
    }
}
