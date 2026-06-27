package io.teknek.deliverance.tensor.operations;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeGpuOperationsLoadTest {

    @Test
    public void nativeGpuOperationsLoadsAndReportsProviderName() {
        NativeGPUTensorOperations operations = new NativeGPUTensorOperations();

        assertEquals("Native GPU Operations", operations.name());
        assertEquals(1, operations.parallelSplitSize());
    }
}
