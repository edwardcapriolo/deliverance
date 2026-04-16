package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestArmPanamaDirectly {
    @Test
    public void testKernel() {
        var ops = new PanamaTensorOperations(MachineSpec.Type.ARM_128,
                new ArrayQueueTensorAllocator(new MetricRegistry()),
                new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        FloatBufferTensor a = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            a.set(i + 1, 0, i);
        }
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(a, DType.Q4, true);
        FloatBufferTensor result = (FloatBufferTensor) PanamaTensorOperationsTest.allZeros(1);
        ops.batchDotProduct(result, a, q4, 0, 0, 32, 0, 0, 1);
        assertEquals("[0][0]=11200.0000".trim(), TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }

    @Test
    public void testKernel86() {
        var ops = new PanamaTensorOperations(MachineSpec.Type.AVX_256,
                new ArrayQueueTensorAllocator(new MetricRegistry()),
                new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
        FloatBufferTensor a = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            a.set(i + 1, 0, i);
        }
        Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(a, DType.Q4, true);
        FloatBufferTensor result = (FloatBufferTensor) PanamaTensorOperationsTest.allZeros(1);
        ops.batchDotProduct(result, a, q4, 0, 0, 32, 0, 0, 1);
        assertEquals("[0][0]=11200.0000".trim(), TensorDisplayUtil.pretty2dDisplayAll(result).trim());
    }
}
