package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.when;

public class RmsNormTest {
    @Test
    public void partialRangeUsesRangeLengthNotEmbeddingLength() {
        AbstractModel model = Mockito.mock(AbstractModel.class);
        Config config = new Config(
                16,
                8,
                16,
                2,
                1,
                1,
                1.0e-6f,
                32,
                2,
                List.of(1),
                io.teknek.deliverance.math.ActivationFunction.Type.GELU_PYTORCH_TANH,
                10000.0,
                null,
                null,
                4,
                null,
                null,
                null,
                null,
                null,
                null,
                List.of("LlamaForCausalLM")
        );
        when(model.getConfig()).thenReturn(config);
        when(model.makeDenseTensor(Mockito.any(io.teknek.deliverance.tensor.TensorShape.class)))
                .thenAnswer(invocation -> new FloatBufferTensor((io.teknek.deliverance.tensor.TensorShape) invocation.getArgument(0)));

        try (AbstractTensor input = new FloatBufferTensor(1, 8);
             AbstractTensor weights = new FloatBufferTensor(1, 8)) {
            for (int i = 0; i < 8; i++) {
                weights.set(1.0f, 0, i);
            }
            input.set(3.0f, 0, 0);
            input.set(4.0f, 0, 1);

            RmsNorm norm = new RmsNorm(model, weights, new MetricRegistry());
            try (AbstractTensor output = norm.forward(input, 0, 2)) {
                assertEquals(0.848528f, output.get(0, 0), 1.0e-5f);
                assertEquals(1.131370f, output.get(0, 1), 1.0e-5f);
            }
        }
    }
}
