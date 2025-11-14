package io.teknek.deliverance.geneator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperationsTest;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


/*
import torch
import torch.nn as nn

torch.manual_seed(10)

batch_size, seq_len, hidden_dim = 2, 5, 7
input_tensor = torch.randn( seq_len, hidden_dim)

print(input_tensor)
print(f"Original input tensor shape: {input_tensor.shape}")
print(f"Original input tensor (first sample): \n{input_tensor[0]}\n")

layer_norm = nn.LayerNorm(normalized_shape=7, elementwise_affine=False)
print(f" {layer_norm}")

# Apply LayerNorm to the input tensor
output_tensor = layer_norm(input_tensor)
print(f"result \n{output_tensor}\n")

edward@fedora:~/mlai-demo$ python3 audio_kit/src/slm/layernorm.py
tensor([[-0.8173, -0.5556, -0.8267, -1.2970, -0.1974, -0.9643, -0.5133],
        [ 2.6278, -0.7465,  1.0051, -0.2568,  0.4765, -0.6652, -0.3627],
        [-1.4504, -0.2496,  0.8298,  1.1209,  0.9999, -0.4344, -2.1806],
        [-1.1094, -2.0410,  0.0334,  1.6294, -2.1184,  0.7828,  0.5632],
        [-1.1151,  0.1490, -1.0923, -1.4551,  1.3358,  0.3075,  0.6278]])
Original input tensor shape: torch.Size([5, 7])
Original input tensor (first sample):
tensor([-0.8173, -0.5556, -0.8267, -1.2970, -0.1974, -0.9643, -0.5133])

 LayerNorm((7,), eps=1e-05, elementwise_affine=False)
result
tensor([[-0.2393,  0.5583, -0.2678, -1.7008,  1.6497, -0.6872,  0.6871],
        [ 2.0881, -0.9347,  0.6344, -0.4960,  0.1609, -0.8619, -0.5908],
        [-1.0617, -0.0462,  0.8665,  1.1127,  1.0104, -0.2026, -1.6792],
        [-0.5830, -1.2736,  0.2641,  1.4471, -1.3309,  0.8195,  0.6568],
        [-0.9635,  0.3355, -0.9401, -1.3129,  1.5551,  0.4984,  0.8275]])


 */
public class LayerNormTest {

    @Test
    void layerNormTest(){
        float [][] goldInput = {
                { 0.8173F, -0.5556F, -0.8267F, -1.2970F, -0.1974F, -0.9643F, -0.5133F },
                { 2.6278F, -0.7465F,  1.0051F, -0.2568F,  0.4765F, -0.6652F, -0.3627F },
                { -1.4504F, -0.2496F,  0.8298F,  1.1209F,  0.9999F, -0.4344F, -2.1806F },
                { -1.1094F, -2.0410F,  0.0334F,  1.6294F, -2.1184F,  0.7828F,  0.5632F },
                {-1.1151F,  0.1490F, -1.0923F, -1.4551F,  1.3358F,  0.3075F,  0.6278F}
        };
        AbstractTensor inputTensor = new FloatBufferTensor(5, 7);
        for (int i = 0 ; i< goldInput.length; i++){
            for (int j = 0; j< goldInput[i].length; j++){
                inputTensor.set(goldInput[i][j], i, j);
            }
        }
        TensorCache tc = new TensorCache(new MetricRegistry());
        AbstractTensor<?,?> output = tc.get(inputTensor.getDType(), inputTensor.shape());

        AbstractTensor<?,?> weights = PanamaTensorOperationsTest.allOnes(7);
        AbstractTensor<?,?> bias = PanamaTensorOperationsTest.allZeros(7);
        LayerNorm.performLayerNorm(inputTensor, output, weights, bias, Double.valueOf(1e-05).floatValue(), 0, 7, 7);
        System.out.println("out");
        for (int i = 0 ; i < goldInput.length; i++){
            for (int j = 0; j < goldInput[i].length; j++){
                System.out.print(output.get(i, j) + " ");
            }
            System.out.println();
        }
        //last row is great
        //[-0.9635,  0.3355, -0.9401, -1.3129,  1.5551,  0.4984,  0.8275]
        //-0.9635164 0.33550507 -0.9400866 -1.3129091 1.5550911 0.4983837 0.8275322
        Assertions.assertEquals(output.get(4,0), -0.9635, .0001);

        //first row isnt
        //2.0958734 -0.07973201 -0.5093384 -1.2546128 0.48789987 -0.72739017 -0.012700124
        //[-0.2393,  0.5583, -0.2678, -1.7008,  1.6497, -0.6872,  0.6871]
    }
}



