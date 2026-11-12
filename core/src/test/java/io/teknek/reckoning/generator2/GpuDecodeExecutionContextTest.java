package io.teknek.reckoning.generator2;

import io.teknek.reckoning.tensor.impl.FloatBufferTensor;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;

class GpuDecodeExecutionContextTest {
    @Test
    void cpuOnlyLighterDoesNotAttemptGpuProjection() {
        Lighter lighter = new Lighter();
        GpuDecodeExecutionContext context = new GpuDecodeExecutionContext(lighter);
        try (FloatBufferTensor inputTensor = new FloatBufferTensor(1, 4);
             FloatBufferTensor outputTensor = new FloatBufferTensor(1, 2);
             FloatBufferTensor weightTensor = new FloatBufferTensor(2, 4);
             TensorRef input = TensorRef.borrowed(inputTensor);
             TensorRef output = TensorRef.borrowed(outputTensor);
             TensorRef weight = TensorRef.borrowed(weightTensor)) {
            assertFalse(context.project(input, output, weight, 4, 0, 2));
        } finally {
            context.close();
        }
    }
}
