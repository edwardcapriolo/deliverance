package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NativeGPUDecodePagedAttentionIT {

    @Test
    void gpuDecodePagedAttentionMatchesReferenceForSmallF32Pages() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2))) {
            TensorOperations reference = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations gpu = loadGpuOperations();
            assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

            int numberOfHeads = 4;
            int numberOfKeyValueHeads = 2;
            int headSize = 8;
            int kvLength = numberOfKeyValueHeads * headSize;
            int visibleRows = 5;
            try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 3);
                 AbstractTensor keyPage0 = tensor(3, kvLength, 7);
                 AbstractTensor keyPage1 = tensor(3, kvLength, 11);
                 AbstractTensor valuePage0 = tensor(3, kvLength, 13);
                 AbstractTensor valuePage1 = tensor(3, kvLength, 17);
                 AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * headSize);
                 AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * headSize)) {
                AbstractTensor[] keyPages = { keyPage0, keyPage1 };
                AbstractTensor[] valuePages = { valuePage0, valuePage1 };

                assertTrue(gpu.supportsDecodePagedAttention(actual, query, keyPages, valuePages, visibleRows,
                        numberOfHeads, numberOfKeyValueHeads, headSize, 0.25f, 2.0f));
                reference.decodePagedAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, 0.25f, 2.0f);
                gpu.decodePagedAttention(actual, query, keyPages, valuePages, visibleRows, numberOfHeads,
                        numberOfKeyValueHeads, headSize, 0.25f, 2.0f);

                for (int col = 0; col < expected.shape().last(); col++) {
                    assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-4f, "col=" + col);
                }
            }
        }
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
        return tensor;
    }

    private static TensorOperations loadGpuOperations() {
        try {
            return new NativeGPUTensorOperations();
        } catch (Throwable t) {
            return null;
        }
    }
}
