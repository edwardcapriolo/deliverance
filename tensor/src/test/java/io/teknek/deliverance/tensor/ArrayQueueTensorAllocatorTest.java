package io.teknek.deliverance.tensor;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

class ArrayQueueTensorAllocatorTest {
    @Test
    void capacityIsMeasuredInBytesNotTensorElements() {
        MetricRegistry metrics = new MetricRegistry();
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(5, metrics);
        TensorShape oneFloat = TensorShape.of(1);

        AbstractTensor first = allocator.getDirty(DType.F32, oneFloat);
        AbstractTensor second = allocator.getDirty(DType.F32, oneFloat);
        first.close();
        second.close();

        AbstractTensor reused = allocator.getDirty(DType.F32, oneFloat);
        AbstractTensor notCached = allocator.getDirty(DType.F32, oneFloat);
        try {
            assertSame(first, reused);
            assertNotSame(second, notCached);
        } finally {
            reused.close();
            notCached.close();
        }
    }
}
