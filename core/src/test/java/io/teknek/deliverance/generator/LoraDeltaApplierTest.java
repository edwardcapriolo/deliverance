package io.teknek.deliverance.generator;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.LoraLayerDelta;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

/**
 * Verifies {@link LoraDeltaApplier}'s two-matmul-plus-accumulate math against a small
 * hand-computed reference -- see step 4 plan Section 8's "prove the math on edge cases with a
 * naive/reference calculation" testing convention (same one steps 1-3 used).
 */
public class LoraDeltaApplierTest {

    @Test
    void appliesTwoMatmulPlusAccumulateAgainstHandComputedReference() {
        AbstractModel model = Mockito.mock(AbstractModel.class);
        when(model.primaryTensorOperations()).thenReturn(new NaiveTensorOperations());
        when(model.getWorkingDType()).thenReturn(DType.F32);
        TensorAllocator allocator = Mockito.mock(TensorAllocator.class);
        when(allocator.getDirty(Mockito.eq(DType.F32), Mockito.any(TensorShape.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((TensorShape) invocation.getArgument(1)));
        when(model.getTensorAllocator()).thenReturn(allocator);

        // input = [1, 2, 3] (batch=1, inFeatures=3)
        try (AbstractTensor input = new FloatBufferTensor(1, 3);
                AbstractTensor loraA = new FloatBufferTensor(2, 3); // [rank=2, inFeatures=3]
                AbstractTensor scaledLoraB = new FloatBufferTensor(2, 2); // [outFeatures=2, rank=2]
                AbstractTensor output = new FloatBufferTensor(1, 2)) {
            input.set(1.0f, 0, 0);
            input.set(2.0f, 0, 1);
            input.set(3.0f, 0, 2);

            // loraA row0 = [1,0,0], row1 = [0,1,0] -> rankResult = [1, 2]
            loraA.set(1.0f, 0, 0);
            loraA.set(0.0f, 0, 1);
            loraA.set(0.0f, 0, 2);
            loraA.set(0.0f, 1, 0);
            loraA.set(1.0f, 1, 1);
            loraA.set(0.0f, 1, 2);

            // scaledLoraB row0 = [1,1] -> 1*1+2*1=3, row1 = [2,0] -> 1*2+2*0=2 -> deltaResult = [3, 2]
            scaledLoraB.set(1.0f, 0, 0);
            scaledLoraB.set(1.0f, 0, 1);
            scaledLoraB.set(2.0f, 1, 0);
            scaledLoraB.set(0.0f, 1, 1);

            output.set(10.0f, 0, 0);
            output.set(20.0f, 0, 1);

            LoraLayerDelta delta = new LoraLayerDelta(loraA, scaledLoraB, 2);
            LoraDeltaApplier.apply(model, output, input, delta);

            assertEquals(13.0f, output.get(0, 0), 1e-6f);
            assertEquals(22.0f, output.get(0, 1), 1e-6f);
        }
    }
}
