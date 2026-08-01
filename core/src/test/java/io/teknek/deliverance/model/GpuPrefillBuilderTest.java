package io.teknek.deliverance.model;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorTestSupport;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class GpuPrefillBuilderTest {

    @Test
    void nativeGpuOperationCanRunTargetedDotProductChunkWhenAvailable() throws Exception {
        TensorOperations gpu;
        try {
            gpu = (TensorOperations) Class.forName("io.teknek.deliverance.tensor.operations.NativeGPUTensorOperations")
                    .getConstructor().newInstance();
        } catch (Throwable t) {
            Assumptions.abort("Native GPU operations are not available: " + t.getMessage());
            return;
        }

        try (AbstractTensor a = TensorTestSupport.tensorOf(2, 3, 1, 2, 3, 4, 5, 6);
             AbstractTensor b = TensorTestSupport.tensorOf(4, 3,
                     1, 0, 0,
                     0, 1, 0,
                     0, 0, 1,
                     1, 1, 1);
             AbstractTensor result = TensorTestSupport.tensorOf(2, 4, 0, 0, 0, 0, 0, 0, 0, 0)) {

            gpu.registerModelTensor(b);
            gpu.dotProductChunk(result, a, b, 0, 3, 0, 4);

            assertEquals(1.0f, result.get(0, 0));
            assertEquals(2.0f, result.get(0, 1));
            assertEquals(3.0f, result.get(0, 2));
            assertEquals(6.0f, result.get(0, 3));
            assertEquals(4.0f, result.get(1, 0));
            assertEquals(5.0f, result.get(1, 1));
            assertEquals(6.0f, result.get(1, 2));
            assertEquals(15.0f, result.get(1, 3));
        }
    }
}
