package io.teknek.deliverance.model.gemma4;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Gemma4PleMathTest {
    @Test
    public void combinePerLayerInputsUsesOneOverRootTwoScale() {
        try (AbstractTensor projected = new FloatBufferTensor(1, 4);
             AbstractTensor tokenIdentity = new FloatBufferTensor(1, 4)) {
            projected.set(3.0f, 0, 0);
            projected.set(5.0f, 0, 1);
            projected.set(7.0f, 0, 2);
            projected.set(9.0f, 0, 3);

            tokenIdentity.set(1.0f, 0, 0);
            tokenIdentity.set(1.0f, 0, 1);
            tokenIdentity.set(1.0f, 0, 2);
            tokenIdentity.set(1.0f, 0, 3);

            float scale = (float) (1.0 / Math.sqrt(2.0));
            Gemma4PleSupport.combinePerLayerInputs(new ConfigurableTensorProvider(new NaiveTensorOperations()),
                    projected, tokenIdentity, scale, 4);

            assertEquals(4.0f * scale, projected.get(0, 0), 1.0e-6f);
            assertEquals(6.0f * scale, projected.get(0, 1), 1.0e-6f);
            assertEquals(8.0f * scale, projected.get(0, 2), 1.0e-6f);
            assertEquals(10.0f * scale, projected.get(0, 3), 1.0e-6f);
        }
    }

    @Test
    public void splitPerLayerInputsUsesPackedLayerSlices() {
        try (AbstractTensor projected = new FloatBufferTensor(1, 6);
             AbstractTensor ignored = new FloatBufferTensor(1, 1)) {
            for (int i = 0; i < 6; i++) {
                projected.set(i + 1, 0, i);
            }

            AbstractTensor[] split = Gemma4PleSupport.splitPerLayerInputs(projected, 1, 3, 2, FloatBufferTensor::new);
            try {
                assertEquals(1.0f, split[0].get(0, 0), 1.0e-6f);
                assertEquals(2.0f, split[0].get(0, 1), 1.0e-6f);
                assertEquals(3.0f, split[1].get(0, 0), 1.0e-6f);
                assertEquals(4.0f, split[1].get(0, 1), 1.0e-6f);
                assertEquals(5.0f, split[2].get(0, 0), 1.0e-6f);
                assertEquals(6.0f, split[2].get(0, 1), 1.0e-6f);
            } finally {
                for (AbstractTensor tensor : split) {
                    tensor.close();
                }
            }
        }
    }
}
